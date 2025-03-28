import os
import numpy as np
import pandas as pd
import rasterio
import cv2
import rasterio
import math
from PIL import Image
import time
import shutil
import random

# ===========================
# PART 1 - filtering scenes
# ===========================

# path
base_path = "/Volumes/Extreme Pro/SARFish_Data"

# input and output directory for filtered data
scene_directory_input = os.path.join(base_path, "data_clean")
scene_directory_output = os.path.join(base_path, "data_clean_speckle_new")
os.makedirs(scene_directory_output, exist_ok=True)

def lee_speckle_filter(img, window_size=5):
    """
    -Apply the Lee filter for speckle noise reduction.
    -img: 2D numpy array
    -window_size: window size for the box filter
    -return: 2D numpy array filtered
    """
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(window_size, window_size))
    mean_squared = cv2.boxFilter(img**2, ddepth=-1, ksize=(window_size, window_size))
    variance = mean_squared - mean**2
    variance = np.maximum(variance, 1e-6)
    overall_variance = np.mean(variance)
    b = variance / (variance + overall_variance)
    return mean + b * (img - mean)

def contrast_filter(image8, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    -local contrast enhancement wit h CLAHE to make easier visual interpretation easier.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image8)

def filter_entire_scene(scene_name):
    """
    -load the VV channel of a scene, apply Lee filter, then CLAHE.
    -save the result as a single-band in scene_directory_output.
    """
    print(f"starting filtering for scene: {scene_name}")
    scene_input_path = os.path.join(scene_directory_input, scene_name + ".SAFE")
    vv_files = [f for f in os.listdir(scene_input_path) if "vv" in f.lower() and f.endswith(".tiff")]
    if not vv_files:
        print("No VV file in", scene_input_path)
        return
    vv_path = os.path.join(scene_input_path, vv_files[0])
    
    with rasterio.open(vv_path) as src:
        sar = src.read(1).astype(np.float32)
        profile = src.profile
    
    # normalize to [0,1]
    sar_minimum, sar_maximus = sar.min(), sar.max()
    if sar_maximus - sar_minimum < 1e-9:
        print("Scene is empty or invalid range:", vv_path)
        return
    sar_normalized = (sar - sar_minimum) / (sar_maximus - sar_minimum)
    
    # apply lee speckle filter
    print("applying Lee filter...")
    sar_filtered = lee_speckle_filter(sar_normalized, window_size=5)
    
    # convert to 8bit and apply local contras t
    sar_8_bit = np.clip(sar_filtered * 255, 0, 255).astype(np.uint8)
    print("applying CLAHE for contrast boost...")
    sar_contrast = contrast_filter(sar_8_bit, clip_limit=2.0, tile_grid_size=(8,8))
    
    # save the filtered result
    scene_output_path = os.path.join(scene_directory_output, scene_name + ".SAFE")
    os.makedirs(scene_output_path, exist_ok=True)
    tiff_output = os.path.join(scene_output_path, vv_files[0].replace(".tiff", "_speckle.tiff"))
    new_profile = profile.copy()
    new_profile.update(dtype=rasterio.uint8, count=1)
    
    with rasterio.open(tiff_output, 'w', **new_profile) as dst:
        dst.write(sar_contrast, 1)
    
    print(f"filtered: {scene_name}")

if __name__ == "__main__":
    # Process each scene in data_clean
    all_scenes = [
        f.replace(".SAFE", "") for f in os.listdir(scene_directory_input)
        if f.endswith(".SAFE") and not f.startswith("._")
    ]
    print(f"Found {len(all_scenes)} scenes to filter.")
    for scene_id in all_scenes:
        filter_entire_scene(scene_id)
    print("filtering complete.")



# ==================================================
#PART 2 - tiling, iteration 6
# ==================================================

Image.MAX_IMAGE_PIXELS = None

# paths
csv_path = os.path.join(base_path, "GRD_validation.csv")
scene_dir = os.path.join(base_path, "data_clean_speckle_new")

output_directory = os.path.join(base_path, "yolo_tiled_speckle_new")
images_output = os.path.join(output_directory, "images")
labels_output = os.path.join(output_directory, "labels")
os.makedirs(images_output, exist_ok=True)
os.makedirs(labels_output, exist_ok=True)

# tiling parameters 
TILE_SIZE = 256
OVERLAP = 64
MIN_BOX_SIZE = 2

df = pd.read_csv(csv_path)
df['is_vessel'] = df['is_vessel'].fillna(False).infer_objects().astype(bool)
df = df[df['is_vessel'] == True].copy()

def sort_coordinates(row):
    left, right = sorted([row['left'], row['right']])
    top, bottom = sorted([row['top'], row['bottom']])
    return left, top, right, bottom

df[['left', 'top', 'right', 'bottom']] = df.apply(lambda r: pd.Series(sort_coordinates(r)), axis=1)
df = df[(df['right'] - df['left'] >= 1) & (df['bottom'] - df['top'] >= 1)]
used_scene_ids = [f.replace(".SAFE", "") for f in os.listdir(scene_dir) if f.endswith(".SAFE")]

def convert_to_yolo_normalization(xmin, ymin, xmax, ymax, tile_w, tile_h):
    """
    -convert bounding box to YOLO format
    """
    x_c = (xmin + xmax) / 2.0 / tile_w
    y_c = (ymin + ymax) / 2.0 / tile_h
    w = (xmax - xmin) / tile_w
    h = (ymax - ymin) / tile_h
    return x_c, y_c, w, h

def tile_image(scene_id, tile_size=TILE_SIZE, overlap=OVERLAP):
    """
    -read the filtered 8bit processed tiff and tile it.
    -tiles are saved in subdirectories to reduce folder overhead,
    -YOLO .txt labels are generated for overlapping bounding boxes.
    """
    print(f"starting tiling for scene: {scene_id}")
    scene_path = os.path.join(scene_dir, scene_id + ".SAFE")
    vv_files = [f for f in os.listdir(scene_path) if "vv" in f.lower() and f.endswith(".tiff")]
    if not vv_files:
        print(f"No VV for {scene_id}")
        return
    tiff_path = os.path.join(scene_path, vv_files[0])
    try:
        with rasterio.open(tiff_path) as src:
            full_image = src.read(1)
    except Exception as e:
        print(f"error reading {tiff_path}: {e}")
        return

    h, w = full_image.shape
    scene_df = df[df['GRD_product_identifier'] == scene_id]

    # create subdirectories for this scene
    scene_image_directory = os.path.join(images_output, scene_id)
    scene_label_directory = os.path.join(labels_output, scene_id)
    os.makedirs(scene_image_directory, exist_ok=True)
    os.makedirs(scene_label_directory, exist_ok=True)

    # calculate tile steps
    x_steps = range(0, w, tile_size - overlap)
    y_steps = range(0, h, tile_size - overlap)
    tile_count = 0

    for y0 in y_steps:
        for x0 in x_steps:
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            tile_w = x1 - x0
            tile_h = y1 - y0
            if tile_w < 100 or tile_h < 100:
                continue

            tile = full_image[y0:y1, x0:x1]
            tile_name = f"{scene_id}_x{x0}_y{y0}.png"
            tile_path = os.path.join(scene_image_directory, tile_name)
            Image.fromarray(tile).save(tile_path)

            # process bounding boxes
            df_subtemp = []
            for _, row in scene_df.iterrows():
                xmin, ymin, xmax, ymax = row['left'], row['top'], row['right'], row['bottom']
                if xmax < x0 or xmin > x1 or ymax < y0 or ymin > y1:
                    continue
                nxmin = max(x0, xmin) - x0
                nymin = max(y0, ymin) - y0
                nxmax = min(x1, xmax) - x0
                nymax = min(y1, ymax) - y0

                if (nxmax - nxmin) < MIN_BOX_SIZE or (nymax - nymin) < MIN_BOX_SIZE:
                    continue
                
                x_c, y_c, bw, bh = convert_to_yolo_normalization(nxmin, nymin, nxmax, nymax, tile_w, tile_h)

                if any([x_c <= 0, y_c <= 0, bw <= 0, bh <= 0, x_c >= 1, y_c >= 1]):
                    continue
                df_subtemp.append(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
            label_name = tile_name.replace(".png", ".txt")
            label_path = os.path.join(scene_label_directory, label_name)

            with open(label_path, "w") as f:
                for line in df_subtemp:
                    f.write(line + "\n")
            tile_count += 1

            if tile_count % 500 == 0:
                print(f"scene {scene_id}: processed {tile_count} tiles ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"scene {scene_id}: created {tile_count} tiles.")

# ppocess each scene for tiling
for sid in used_scene_ids:
    tile_image(sid, TILE_SIZE, OVERLAP)
print("FINALLY TILING COMPLETE!!! :D ")



# ==============================
# PART 3 - oversampling positive tiles
# ====================================

def oversample_vessel_tiles(images_directory, labels_dir, factor=2):
    """
    -for each tile with at least one bounding box, replicate it 'factor' times
    - the goal isto boost vessel representation in training data.
    """
    for scene_sub_directory in os.listdir(images_directory):
        full_scene_directory = os.path.join(images_directory, scene_sub_directory)
        if not os.path.isdir(full_scene_directory):
            continue
        all_images = [f for f in os.listdir(full_scene_directory) if f.endswith('.png')]
        for image_name in all_images:
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_scene_dir = os.path.join(labels_dir, scene_sub_directory)
            label_path = os.path.join(label_scene_dir, label_name)
            if not os.path.exists(label_path):
                continue
            if os.path.getsize(label_path) > 0:
                for i in range(factor - 1):
                    base, ext = os.path.splitext(image_name)
                    new_image_name = f"{base}_dup{i}{ext}"
                    new_label_name = f"{base}_dup{i}.txt"
                    shutil.copy(os.path.join(full_scene_directory, image_name),
                                os.path.join(full_scene_directory, new_image_name))
                    shutil.copy(label_path,
                                os.path.join(label_scene_dir, new_label_name))


# ================+================
# PART 4 - splitting train and validation
# =====================================

# path to the scene subfolders with tiled images
base_images_dir = "/Volumes/Extreme Pro/SARFish_Data/yolo_tiled_speckle_new/images"

# output text file paths for train and val splits
train_txt = "/Volumes/Extreme Pro/SARFish_Data/yolo_tiled_speckle_new/train.txt"
val_txt = "/Volumes/Extreme Pro/SARFish_Data/yolo_tiled_speckle_new/val.txt"

def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                count += 1
    return count

# count total images
total_images = count_images(base_images_dir)
print(f"Total images found: {total_images}")

# gather all png image paths
all_image_paths = []
for root, dirs, files in os.walk(base_images_dir):
    for file in files:
        if file.lower().endswith(".png"):
            full_path = os.path.join(root, file)
            all_image_paths.append(full_path)

print(f"collected {len(all_image_paths)} image paths.")

# Shuffle and split into 80% train, 20% validation
random.seed(42)
random.shuffle(all_image_paths)
split_idx = int(0.8 * len(all_image_paths))
train_paths = all_image_paths[:split_idx]
val_paths = all_image_paths[split_idx:]

# Write the list of paths to text files
with open(train_txt, "w") as f:
    for path in train_paths:
        f.write(path + "\n")

with open(val_txt, "w") as f:
    for path in val_paths:
        f.write(path + "\n")

print("train and validation text files created:")
print(f"train list: {train_txt} ({len(train_paths)} images)")
print(f"validation list: {val_txt} ({len(val_paths)} images)")

# ==================================================
# PART 5 - run oversampling # sometimes this part need to be ran separatly
# ==================================================
if __name__ == "__main__":
     train_img_dir = "/Volumes/Extreme Pro/SARFish_Data/yolo_tiled_speckle_new/train/images"
     train_label_directory = "/Volumes/Extreme Pro/SARFish_Data/yolo_tiled_speckle_new/train/labels"
     oversample_vessel_tiles(train_img_dir, train_label_directory, factor=3)


# ==================================================
# PART 6 - K-Means Anchors
# ==================================================

from sklearn.cluster import KMeans

labels_directory_anchors = "/Volumes/Extreme Pro/SARFish_Data/yolo_tiled_speckle_new/labels/train"
box_dimentions = []
for label_file in os.listdir(labels_directory_anchors):
    if label_file.endswith(".txt"):
        with open(os.path.join(labels_directory_anchors, label_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x_center, y_center, width, height = map(float, parts)
                    box_dimentions.append([width, height])
box_dimentions = np.array(box_dimentions)
print("collected box dimensions:", box_dimentions.shape)
k = 9
kmeans = KMeans(n_clusters=k, random_state=0).fit(box_dimentions)
anchors = kmeans.cluster_centers_
print("custom anchors (width and height):")
print(anchors)

