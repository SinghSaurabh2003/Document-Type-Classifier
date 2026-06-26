import zipfile
from pathlib import Path

# ============================================================
# CHANGE THESE PATHS
# ============================================================

ZIP_FILE = Path(r"E:\project\archive.zip")

TRAIN_HALF = Path(r"D:\Project 2\archive (3)\Training_half")

LABEL_FILE = "labels/train.txt"

MEMO_LABEL = "15"

# ============================================================

TRAIN_HALF.mkdir(parents=True, exist_ok=True)

memo_index = 0

print("=" * 60)
print("Extracting Memo Images...")
print("=" * 60)

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:

    with zip_ref.open(LABEL_FILE) as train_file:

        for line in train_file:

            line = line.decode("utf-8").strip()

            if not line:
                continue

            image_path, label = line.split()

            # Only Memo images
            if label != MEMO_LABEL:
                continue

            destination = TRAIN_HALF / f"memo{memo_index}.tif"

            zip_image_path = f"images/{image_path}"
            with zip_ref.open(zip_image_path) as src:
                destination.write_bytes(src.read())

            memo_index += 1

            if memo_index % 1000 == 0:
                print(f"{memo_index} images extracted...")

print("\n" + "=" * 60)
print("Extraction Completed Successfully")
print("=" * 60)
print(f"Total Memo Images Extracted : {memo_index}")
print(f"Saved To                   : {TRAIN_HALF}")
print("=" * 60)
