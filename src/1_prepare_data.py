import os
from PIL import Image
from tqdm import tqdm

SOURCE = "../data/raw"
CLEAN_DIR = "../data/cleaned"

os.makedirs(CLEAN_DIR, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png")

image_paths = []

for root, _, files in os.walk(SOURCE):
    for f in files:
        if f.lower().endswith(valid_ext):
            image_paths.append(os.path.join(root, f))

print("Total images found:", len(image_paths))

count = 0

for path in tqdm(image_paths):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((224, 224))

        save_path = os.path.join(CLEAN_DIR, f"{count}.jpg")
        img.save(save_path, "JPEG", quality=90)

        count += 1
    except:
        continue

print("Cleaned images:", count)