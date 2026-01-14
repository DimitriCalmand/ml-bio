import os
import shutil
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Configuration
SOURCE_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/split")
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # Train (increased), Val (reduced), Test (reduced)
SEED = 42

def split_dataset():
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    # Set seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Create output directories
    for split in ['train', 'val', 'test']:
        split_dir = OUTPUT_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)

    # Get classes
    classes = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes: {classes}")

    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        class_dir = SOURCE_DIR / class_name
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIOS[0])
        n_val = int(n_total * SPLIT_RATIOS[1])
        # Remainig for test
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        print(f"  Total: {n_total} -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
        
        for split, imgs in splits.items():
            dest_dir = OUTPUT_DIR / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in tqdm(imgs, desc=f"  Copying to {split}", leave=False):
                shutil.copy2(img_path, dest_dir / img_path.name)

    print("\n" + "="*50)
    print("Dataset splitting complete!")
    print(f"Resulting data structure created in: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    split_dataset()
