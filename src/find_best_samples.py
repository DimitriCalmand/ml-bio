import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import heapq

# Configuration
DATA_DIR = Path("data/split")
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "best_samples.json"

IMAGE_SIZE = (224, 224)
SAMPLES_PER_CLASS = 5

# Custom Object Support
try:
    from tensorflow.keras.losses import CategoricalFocalCrossentropy
except ImportError:
    class CategoricalFocalCrossentropy(keras.losses.Loss):
         def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
            super().__init__(**kwargs)
         def call(self, y_true, y_pred): return tf.reduce_sum(y_true)

def load_model_and_classes():
    model_path = MODEL_DIR / "best_model_finetuned.keras"
    if not model_path.exists():
         model_path = MODEL_DIR / "best_model.keras"
    
    print(f"Loading model: {model_path}")
    try:
        model = keras.models.load_model(model_path, custom_objects={'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy})
    except:
        model = keras.models.load_model(model_path, compile=False)
    
    with open(MODEL_DIR / "class_mapping.json", 'r') as f:
        mapping = json.load(f)
        # Convert keys to int, values are class names
        class_names = [mapping[str(i)] for i in range(len(mapping))]
    
    return model, class_names

def find_best_samples():
    model, class_names = load_model_and_classes()
    
    # Structure to hold heaps for each class
    # Heap stores tuples: (confidence, file_path)
    # We want max heap, but python has min heap. So we store confidence.
    # Actually, let's just collect all correct ones and sort list for simplicity.
    class_correct_candidates = {c: [] for c in class_names}
    
    print("Scanning test set for best performant samples...")
    
    # Iterate through all test images
    image_paths = list(TEST_DIR.glob("*/*.jpg"))
    
    # Prepare batch processing for speed
    batch_size = 32
    
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_labels = []
        batch_filenames = []
        
        valid_indices = []
        
        for idx, p in enumerate(batch_paths):
            try:
                # Load and preprocess
                img = keras.utils.load_img(p, target_size=IMAGE_SIZE)
                img_arr = keras.utils.img_to_array(img)
                
                # Get true label from folder name
                true_label_name = p.parent.name
                if true_label_name not in class_names:
                    continue
                    
                batch_images.append(img_arr)
                batch_labels.append(true_label_name)
                batch_filenames.append(str(p))
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {p}: {e}")
        
        if not batch_images:
            continue
            
        # Inference
        batch_stack = np.array(batch_images)
        # Use predict with verbose=0
        preds = model.predict(batch_stack, verbose=0)
        
        # Analyze results
        for j, pred_probs in enumerate(preds):
            pred_idx = np.argmax(pred_probs)
            confidence = float(pred_probs[pred_idx])
            pred_class = class_names[pred_idx]
            true_class = batch_labels[j]
            filename = batch_filenames[j]
            
            # Check if correct
            if pred_class == true_class:
                class_correct_candidates[true_class].append({
                    "path": filename,
                    "confidence": confidence,
                    "true_class": true_class
                })

    # Select top K for each class
    best_samples = {}
    print("\nSelection Results:")
    for cls in class_names:
        candidates = class_correct_candidates[cls]
        # Sort by confidence descending
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Take top K
        selected = candidates[:SAMPLES_PER_CLASS]
        best_samples[cls] = [s['path'] for s in selected]
        
        # Log
        avg_conf = np.mean([s['confidence'] for s in selected]) if selected else 0
        print(f"  {cls}: Found {len(candidates)} candidates. Top {len(selected)} Avg Conf: {avg_conf:.4f}")
    
    # Save to disk
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(best_samples, f, indent=2)
    
    print(f"\nSaved {sum(len(v) for v in best_samples.values())} best samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    find_best_samples()
