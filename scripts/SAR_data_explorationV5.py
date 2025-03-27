import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#====================
# DATA EXPLORATION
#====================

# paths
csv_path = "/Volumes/Extreme Pro/SARFish_Data/GRD_validation.csv"
scene_directory = "/Volumes/Extreme Pro/SARFish_Data/data_clean_speckle_new"
base_path = "/Volumes/Extreme Pro/SARFish_Data"
tiff_directory = os.path.join(base_path, "data_clean_speckle_new")

# load csv
df = pd.read_csv(csv_path)
print(f"GRD validation shape: {df.shape}")

# get scene folders
scene_ids = [
    folder.replace(".SAFE", "")
    for folder in os.listdir(scene_directory)
    if folder.endswith(".SAFE")
]

print(f"Found {len(scene_ids)} .SAFE scene folders in data_clean")

# filter csv using actual scene ids
df_scenes = df[df['GRD_product_identifier'].isin(scene_ids)].copy()
print(f"Filtered CSV shape: {df_scenes.shape}")

# clean booleans
df_scenes['is_vessel'] = df_scenes['is_vessel'].fillna(False).astype(bool)
df_scenes['is_fishing'] = df_scenes['is_fishing'].fillna(False).astype(bool)

# summary stats
print("\nsummary statistics for selected scenes")
print("total detections:", len(df_scenes))
print("unique scenes:", df_scenes['GRD_product_identifier'].nunique())
print("vessels detected:", df_scenes['is_vessel'].sum())
print("fishing vessels:", df_scenes['is_fishing'].sum())
print("non-Fishing vessels:", (df_scenes['is_vessel'] & ~df_scenes['is_fishing']).sum())
print("confidence distribution:")
print(df_scenes['confidence'].value_counts())

# plot vessel types
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(6, 4))
df_plot = df_scenes[df_scenes['is_vessel']]
df_plot['vessel_type'] = df_plot['is_fishing'].map({True: "Fishing", False: "Non-Fishing"})
sns.countplot(data=df_plot, x='vessel_type', palette='Set2', ax=ax)
ax.set_title("Vessel Type Counts in Selected Scenes")
plt.tight_layout()
plt.show()

# plot confi dence histogram
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df_scenes['confidence'], bins=10, kde=True, ax=ax)
ax.set_title("Detection Confidence Histogram")
plt.tight_layout()
plt.show()


# pick the first valid scene
scene_id = scene_ids[0]
scene_df = df_scenes[df_scenes['GRD_product_identifier'] == scene_id].copy()
scene_path = os.path.join(tiff_directory, scene_id + ".SAFE")

# load VV polarization SAR image
vv_files = [i for i in os.listdir(scene_path) if "vv" in i.lower() and i.endswith(".tiff")]
assert vv_files, f"No VV .tiff image found in {scene_path}"
tiff_path = os.path.join(scene_path, vv_files[0])

with rasterio.open(tiff_path) as source:
    sar_image = source.read(1)
    meta = source.meta

print("image shape:", sar_image.shape)
print("value range:", sar_image.min(), "to", sar_image.max())

# normalize SAR image for display
def normalize_image(image):
    image = np.clip(image, np.percentile(image, 2), np.percentile(image, 98))
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

normalized_image = normalize_image(sar_image)

# show full image with vessel boxes
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(normalized_image, cmap="gray")
ax.set_title(f"SAR Image with Vessels - Scene: {scene_id}")

for _, row in scene_df.iterrows():
    if row['is_vessel']:
        color = 'red' if row['is_fishing'] else 'lime'
        rectangle = patches.Rectangle(
            (row['left'], row['top']),
            row['right'] - row['left'],
            row['bottom'] - row['top'],
            linewidth=1.5,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rectangle)

plt.axis('off')
plt.tight_layout()
plt.show()

# bar chart for fishing vs non-fishing vessels
fig, ax = plt.subplots(figsize=(5, 4))
df_temporal = scene_df[scene_df['is_vessel']].copy()
df_temporal['vessel_type'] = df_temporal['is_fishing'].map({True: "Fishing", False: "Non-Fishing"})

sns.countplot(data=df_temporal, x='vessel_type', palette='Set2', ax=ax)
ax.set_title("Vessel Type Counts")
plt.show()

# crop and zoom in
def get_crop(image, row, padding=30):
    try:
        top, bottom = int(row['top']), int(row['bottom'])
        left, right = int(row['left']), int(row['right'])
    except:
        return None

    top, bottom = sorted([top, bottom])
    left, right = sorted([left, right])

    top = max(0, top - padding)
    bottom = min(image.shape[0], bottom + padding)
    left = max(0, left - padding)
    right = min(image.shape[1], right + padding)

    if bottom - top < 5 or right - left < 5:
        return None

    return image[top:bottom, left:right]

# zoom on fishing vessels
print("\nzoom-in fishing vessels")
for _, row in scene_df[scene_df['is_fishing']].head(3).iterrows():
    crop = get_crop(sar_image, row)
    if crop is not None:
        plt.figure()
        plt.imshow(normalize_image(crop), cmap='gray')
        plt.title(f"ffishing vessel - confidence: {row['confidence']}")
        plt.axis('off')
        plt.show()

# zoom on non-fishing vessels
print("\nzoom-in non fishing Vessels")
for _, row in scene_df[(scene_df['is_vessel']) & (~scene_df['is_fishing'])].head(3).iterrows():
    crop = get_crop(sar_image, row)
    if crop is not None:
        plt.figure()
        plt.imshow(normalize_image(crop), cmap='gray')
        plt.title(f"non fishing vessel - confidence : {row['confidence']}")
        plt.axis('off')
        plt.show()


# plotting style
sns.set_style("whitegrid")

# geospatial distribution of vessels
fig, ax = plt.subplots(figsize=(7, 6))
df_vessel = df_scenes[df_scenes['is_vessel']].copy()
df_vessel['vessel_type'] = df_vessel['is_fishing'].map({True: "Fishing", False: "Non-Fishing"})

sns.scatterplot(
    data=df_vessel,
    x='detect_lon', y='detect_lat',
    hue='vessel_type', palette='Set2', alpha=0.6, s=30, ax=ax
)
ax.set_title("geospatial distribution of vessels")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.legend()
plt.tight_layout()
plt.show()


# bounding box size histogram s
df_vessel['width'] = df_vessel['right'] - df_vessel['left']
df_vessel['height'] = df_vessel['bottom'] - df_vessel['top']

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_vessel['width'], bins=30, ax=axs[0], color='skyblue')
axs[0].set_title("bounding box widths")

sns.histplot(df_vessel['height'], bins=30, ax=axs[1], color='salmon')
axs[1].set_title("bounding box hheights")

plt.suptitle("vessel bounding box dimensions")
plt.tight_layout()
plt.show()


# aspect ratio histograms
df_vessel['aspect_ratio'] = df_vessel['width'] / df_vessel['height'].replace(0, np.nan)

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df_vessel, x='aspect_ratio', hue='vessel_type', bins=30, palette='Set2', alpha=0.6)
ax.set_title("aspect ratio of vessels")
plt.tight_layout()
plt.show()