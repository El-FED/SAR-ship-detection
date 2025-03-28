# validate.py

import os
import random
import textwrap
import subprocess

# paths
base_path = "/Volumes/Extreme Pro/SARFish_Data/yolo_tiled_speckle_new"
yolo_directory = "/Users/facundofed/yolov5"
validation_text = os.path.join(base_path, "val.txt")
mini_validation_text = os.path.join(base_path, "mini_val_2000.txt")
data_yaml = os.path.join(base_path, "data.yaml")
weights = os.path.join(yolo_directory, "runs/train/sar_tiled_speckle2/weights/best.pt")
val_script = os.path.join(yolo_directory, "val.py")

# sample 2000 random tiles
with open(validation_text, "r") as f:
    all_paths = [line.strip() for line in f if line.strip().endswith(".png")]

mini_sample = random.sample(all_paths, min(2000, len(all_paths)))

with open(mini_validation_text, "w") as f:
    for path in mini_sample:
        f.write(path + "\n")

print(f"created mini_val_2000.txt with {len(mini_sample)} images")

# write data.yaml
data_yaml_content = textwrap.dedent(f"""\
train: {os.path.join(base_path, "train.txt")}
val: {mini_validation_text}
nc: 1
names: ["vessel"]
""")
with open(data_yaml, "w") as f:
    f.write(data_yaml_content)
print("data.yaml updated")

# run val.py
val_cmd = [
    "python", val_script,
    "--weights", weights,
    "--data", data_yaml,
    "--img", "256",
    "--conf", "0.001",
    "--iou", "0.65",
    "--save-txt", "--save-conf",
    "--project", os.path.join(yolo_directory, "runs", "val"),
    "--name", "eval_2k",
    "--exist-ok"
]

print("\nRunning evaluation on 2,000 tiles...\n")
subprocess.run(val_cmd, check=True)
