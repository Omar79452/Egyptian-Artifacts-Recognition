import os
import zipfile
from PIL import Image
from tqdm import tqdm

ZIP_DIR = "../data/zips"              # حط هنا كل الـ zip
OUTPUT_DIR = "../data/raw"      # كل الصور هتروح هنا

os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png")

# =========================
# STEP 1: Extract ZIPs
# =========================
print("📦 Extracting ZIP files...")

for file in os.listdir(ZIP_DIR):
    if file.endswith(".zip"):
        zip_path = os.path.join(ZIP_DIR, file)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ZIP_DIR)

print("✅ Extraction Done")

# =========================
# STEP 2: Collect Images
# =========================
print("📸 Collecting images...")

count = 0

for root, _, files in os.walk(ZIP_DIR):
    for f in files:
        if f.lower().endswith(valid_ext):
            src_path = os.path.join(root, f)

            try:
                img = Image.open(src_path).convert("RGB")

                save_path = os.path.join(OUTPUT_DIR, f"{count}.jpg")
                img.save(save_path, "JPEG", quality=90)

                count += 1
            except:
                continue

print(f"✅ Total images collected: {count}")