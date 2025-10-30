from pathlib import Path
import shutil, random

# Root dataset path
root = Path("data/kaggle/Guitar-Detection-2")
neg_src = root / "negatives_raw"

# Define where train/valid/test live
splits = {
    "train": {"images": root / "train/images", "labels": root / "train/labels"},
    "valid": {"images": root / "valid/images", "labels": root / "valid/labels"},
    "test":  {"images": root / "test/images",  "labels": root / "test/labels"},
}

# Make sure the directories exist
for s in splits.values():
    s["images"].mkdir(parents=True, exist_ok=True)
    s["labels"].mkdir(parents=True, exist_ok=True)

# Collect all .jpg/.jpeg/.png negatives
imgs = [p for p in neg_src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file()]
random.shuffle(imgs)

n = len(imgs)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)

# Split into train/valid/test
parts = {
    "train": imgs[:n_train],
    "valid": imgs[n_train:n_train + n_valid],
    "test": imgs[n_train + n_valid:]
}

# Copy and make empty label files
for split, items in parts.items():
    for p in items:
        dst_img = splits[split]["images"] / p.name
        shutil.copy2(p, dst_img)
        (splits[split]["labels"] / (dst_img.stem + ".txt")).write_text("")  # empty label

print("Negatives distributed as follows:")
for split, items in parts.items():
    print(f"  {split}: {len(items)} images")
