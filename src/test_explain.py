"""
Test suite for model explainability features.

This script tests Grad-CAM, Grad-CAM++, and LIME explanations
on sample images from the dataset.

Usage:
    python src/test_explain.py                    # Test on random images
    python src/test_explain.py --image <path>     # Test on specific image
    python src/test_explain.py --all              # Generate explanations for all classes
    python src/test_explain.py --clinical <path>  # Generate clinical report
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
import argparse
from typing import List, Optional

# Import from explain module
from explain import (
    GradCAM,
    GradCAMPlusPlus,
    LIMEExplainer,
    ExplanationGenerator,
    create_clinical_explanation,
    load_and_preprocess_image,
    get_img_array,
    make_gradcam_heatmap,
    save_and_display_gradcam,
    explain_with_lime
)


# Configuration
MODEL_DIR = Path("models")
DATA_DIR = Path("data/split/test") if Path("data/split/test").exists() else Path("data/processed")
RESULTS_DIR = Path("results")
EXPLANATIONS_DIR = RESULTS_DIR / "explanations"
IMAGE_SIZE = (224, 224)


def load_model_and_classes():
    """Load the trained model and class names."""
    model_path = MODEL_DIR / "best_model_finetuned.keras"
    
    if not model_path.exists():
        # Try alternative model path
        model_path = MODEL_DIR / "best_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found. Please train the model first with: python src/train.py"
            )
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load class mapping
    with open(MODEL_DIR / "class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    
    # Convert to ordered list
    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    
    return model, class_names


def get_sample_images(class_names: List[str], num_per_class: int = 1) -> List[Path]:
    """Get sample images from each class."""
    sample_images = []
    
    for class_name in class_names:
        class_dir = DATA_DIR / class_name
        if class_dir.exists():
            images = list(class_dir.glob("*.jpg"))
            if images:
                # Select random samples
                num_samples = min(num_per_class, len(images))
                samples = random.sample(images, num_samples)
                sample_images.extend(samples)
    
    return sample_images


def test_gradcam(model, class_names: List[str], img_path: Path):
    """Test Grad-CAM implementation."""
    print("\n" + "=" * 60)
    print("TESTING GRAD-CAM")
    print("=" * 60)
    
    print(f"\nImage: {img_path}")
    
    # Load image
    img_array, original = load_and_preprocess_image(str(img_path), IMAGE_SIZE)
    
    # Get prediction
    predictions = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    pred_class = class_names[pred_idx]
    confidence = predictions[0][pred_idx]
    
    print(f"Prediction: {pred_class} ({confidence:.2%})")
    
    # Initialize Grad-CAM
    try:
        gradcam = GradCAM(model)
        print(f"Target layer: {gradcam.layer_name}")
        
        # Compute heatmap
        heatmap = gradcam.compute_heatmap(img_array, pred_idx)
        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        
        # Create overlay
        overlay = gradcam.overlay_heatmap(heatmap, original)
        
        # Save result
        EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = EXPLANATIONS_DIR / f"gradcam_{img_path.stem}.png"
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original / 255.0)
        axes[0].set_title(f"Original\nTrue: {img_path.parent.name}")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nPred: {pred_class} ({confidence:.1%})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Grad-CAM saved to: {save_path}")
        return True
        
    except Exception as e:
        print(f"✗ Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradcam_plusplus(model, class_names: List[str], img_path: Path):
    """Test Grad-CAM++ implementation."""
    print("\n" + "=" * 60)
    print("TESTING GRAD-CAM++")
    print("=" * 60)
    
    print(f"\nImage: {img_path}")
    
    # Load image
    img_array, original = load_and_preprocess_image(str(img_path), IMAGE_SIZE)
    
    # Get prediction
    predictions = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    pred_class = class_names[pred_idx]
    confidence = predictions[0][pred_idx]
    
    print(f"Prediction: {pred_class} ({confidence:.2%})")
    
    try:
        gradcam_pp = GradCAMPlusPlus(model)
        print(f"Target layer: {gradcam_pp.layer_name}")
        
        # Compute heatmap
        heatmap = gradcam_pp.compute_heatmap(img_array, pred_idx)
        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        
        # Create overlay
        overlay = gradcam_pp.overlay_heatmap(heatmap, original)
        
        # Save result
        save_path = EXPLANATIONS_DIR / f"gradcam_pp_{img_path.stem}.png"
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original / 255.0)
        axes[0].set_title(f"Original\nTrue: {img_path.parent.name}")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM++ Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nPred: {pred_class} ({confidence:.1%})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Grad-CAM++ saved to: {save_path}")
        return True
        
    except Exception as e:
        print(f"✗ Grad-CAM++ failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lime(model, class_names: List[str], img_path: Path, num_samples: int = 300):
    """Test LIME implementation."""
    print("\n" + "=" * 60)
    print("TESTING LIME")
    print("=" * 60)
    
    print(f"\nImage: {img_path}")
    print(f"Using {num_samples} samples (increase for better quality)")
    
    # Load image
    img_array, original = load_and_preprocess_image(str(img_path), IMAGE_SIZE)
    
    # Get prediction
    predictions = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    pred_class = class_names[pred_idx]
    confidence = predictions[0][pred_idx]
    
    print(f"Prediction: {pred_class} ({confidence:.2%})")
    
    try:
        from skimage.segmentation import mark_boundaries
        
        lime_explainer = LIMEExplainer(model, class_names)
        
        print("Generating LIME explanation...")
        explanation = lime_explainer.explain(
            img_array,
            num_samples=num_samples,
            num_features=10
        )
        
        print(f"Top labels: {[class_names[l] for l in explanation.top_labels]}")
        
        # Get visualizations
        temp_pos, mask_pos = explanation.get_image_and_mask(
            label=pred_idx,
            positive_only=True,
            num_features=5
        )
        
        temp_all, mask_all = explanation.get_image_and_mask(
            label=pred_idx,
            positive_only=False,
            num_features=5
        )
        
        temp_focus, mask_focus = explanation.get_image_and_mask(
            label=pred_idx,
            positive_only=True,
            num_features=5,
            hide_rest=True
        )
        
        # Save result
        save_path = EXPLANATIONS_DIR / f"lime_{img_path.stem}.png"
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original / 255.0)
        axes[0].set_title(f"Original\nTrue: {img_path.parent.name}")
        axes[0].axis('off')
        
        axes[1].imshow(mark_boundaries(temp_pos, mask_pos))
        axes[1].set_title(f"Positive Features\nFor: {pred_class}")
        axes[1].axis('off')
        
        axes[2].imshow(mark_boundaries(temp_all, mask_all))
        axes[2].set_title("All Features")
        axes[2].axis('off')
        
        axes[3].imshow(temp_focus)
        axes[3].set_title(f"Key Regions\nConfidence: {confidence:.1%}")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ LIME saved to: {save_path}")
        
        # Print feature importance
        importance = explanation.get_feature_importance(pred_idx, num_features=5)
        print("\nTop 5 feature importances:")
        for feat_id, weight in importance:
            direction = "+" if weight > 0 else "-"
            print(f"  Superpixel {feat_id}: {direction}{abs(weight):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ LIME failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_explanation(model, class_names: List[str], img_path: Path):
    """Test the comprehensive explanation generator."""
    print("\n" + "=" * 60)
    print("TESTING COMPREHENSIVE EXPLANATION GENERATOR")
    print("=" * 60)
    
    print(f"\nImage: {img_path}")
    
    try:
        generator = ExplanationGenerator(model, class_names)
        print(f"Target layer: {generator.target_layer}")
        
        print("\nGenerating explanations (Grad-CAM + Grad-CAM++ + LIME)...")
        result = generator.explain_image(
            str(img_path),
            target_size=IMAGE_SIZE,
            methods=['gradcam', 'gradcam++', 'lime'],
            lime_samples=200
        )
        
        pred = result['prediction']
        print(f"\nPrediction: {pred['class_name']} ({pred['confidence']:.2%})")
        print("\nAll probabilities:")
        for cls, prob in sorted(pred['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            print(f"  {cls:8s}: {bar:20s} {prob:.2%}")
        
        # Save comprehensive figure
        save_path = EXPLANATIONS_DIR / f"comprehensive_{img_path.stem}.png"
        generator.create_explanation_figure(result, save_path=save_path)
        plt.close()
        
        print(f"\n✓ Comprehensive explanation saved to: {save_path}")
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive explanation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clinical_report(model, class_names: List[str], img_path: Path):
    """Test clinical report generation."""
    print("\n" + "=" * 60)
    print("TESTING CLINICAL REPORT GENERATION")
    print("=" * 60)
    
    print(f"\nImage: {img_path}")
    
    try:
        save_path = EXPLANATIONS_DIR / f"clinical_{img_path.stem}.png"
        
        print("Generating clinical explanation report...")
        create_clinical_explanation(
            model,
            str(img_path),
            class_names,
            target_size=IMAGE_SIZE,
            output_path=save_path
        )
        plt.close()
        
        print(f"\n✓ Clinical report saved to: {save_path}")
        return True
        
    except Exception as e:
        print(f"✗ Clinical report failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_legacy_functions(model, class_names: List[str], img_path: Path):
    """Test legacy backward-compatible functions."""
    print("\n" + "=" * 60)
    print("TESTING LEGACY FUNCTIONS (Backward Compatibility)")
    print("=" * 60)
    
    print(f"\nImage: {img_path}")
    
    success = True
    img_array = None
    heatmap = None
    
    # Test get_img_array
    try:
        img_array = get_img_array(str(img_path), IMAGE_SIZE)
        print(f"✓ get_img_array: shape={img_array.shape}")
    except Exception as e:
        print(f"✗ get_img_array failed: {e}")
        success = False
        return success  # Can't continue without image
    
    # Test make_gradcam_heatmap
    try:
        heatmap = make_gradcam_heatmap(img_array, model)
        print(f"✓ make_gradcam_heatmap: shape={heatmap.shape}")
    except Exception as e:
        print(f"✗ make_gradcam_heatmap failed: {e}")
        success = False
    
    # Test save_and_display_gradcam
    if heatmap is not None:
        try:
            save_path = EXPLANATIONS_DIR / f"legacy_gradcam_{img_path.stem}.jpg"
            result = save_and_display_gradcam(str(img_path), heatmap, str(save_path))
            print(f"✓ save_and_display_gradcam: saved to {save_path}")
        except Exception as e:
            print(f"✗ save_and_display_gradcam failed: {e}")
            success = False
    else:
        print("✗ save_and_display_gradcam skipped: no heatmap")
        success = False
    
    # Test explain_with_lime
    try:
        boundary_img, label = explain_with_lime(
            img_array, model, class_names, 
            num_features=5, num_samples=100
        )
        print(f"✓ explain_with_lime: label={class_names[label]}")
    except Exception as e:
        print(f"✗ explain_with_lime failed: {e}")
        success = False
    
    return success


def run_all_tests(image_path: Optional[str] = None, test_all_classes: bool = False):
    """Run all explanation tests."""
    print("=" * 60)
    print("EXPLANATION MODULE TEST SUITE")
    print("=" * 60)
    
    # Create output directory
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model and classes
    model, class_names = load_model_and_classes()
    print(f"\nClasses: {class_names}")
    
    # Get test images
    if image_path:
        test_images = [Path(image_path)]
    elif test_all_classes:
        test_images = get_sample_images(class_names, num_per_class=1)
    else:
        # Get one random image
        all_images = get_sample_images(class_names, num_per_class=2)
        if all_images:
            test_images = [random.choice(all_images)]
        else:
            print("ERROR: No images found in data/processed/")
            return
    
    if not test_images:
        print("ERROR: No test images found")
        return
    
    print(f"\nTest images: {len(test_images)}")
    for img in test_images[:5]:
        print(f"  - {img}")
    if len(test_images) > 5:
        print(f"  ... and {len(test_images) - 5} more")
    
    # Run tests
    results = {
        'gradcam': [],
        'gradcam++': [],
        'lime': [],
        'comprehensive': [],
        'clinical': [],
        'legacy': []
    }
    
    for img_path in test_images:
        print(f"\n{'='*60}")
        print(f"Testing on: {img_path.name}")
        print(f"True class: {img_path.parent.name}")
        print("=" * 60)
        
        results['gradcam'].append(test_gradcam(model, class_names, img_path))
        results['gradcam++'].append(test_gradcam_plusplus(model, class_names, img_path))
        results['lime'].append(test_lime(model, class_names, img_path, num_samples=200))
        results['comprehensive'].append(test_comprehensive_explanation(model, class_names, img_path))
        
        # Only generate clinical report for first image
        if img_path == test_images[0]:
            results['clinical'].append(test_clinical_report(model, class_names, img_path))
            results['legacy'].append(test_legacy_functions(model, class_names, img_path))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, test_results in results.items():
        if test_results:
            passed = sum(test_results)
            total = len(test_results)
            status = "✓" if passed == total else "✗"
            print(f"{status} {test_name}: {passed}/{total} passed")
    
    print(f"\nAll explanations saved to: {EXPLANATIONS_DIR}")
    print("=" * 60)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test explainability features for skin disease classification"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a specific image to explain"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test on one image from each class"
    )
    parser.add_argument(
        "--clinical",
        type=str,
        default=None,
        help="Generate clinical report for a specific image"
    )
    
    args = parser.parse_args()
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {len(gpus)} device(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("Using CPU")
    
    if args.clinical:
        # Only generate clinical report
        model, class_names = load_model_and_classes()
        EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
        test_clinical_report(model, class_names, Path(args.clinical))
    else:
        # Run all tests
        run_all_tests(
            image_path=args.image,
            test_all_classes=args.all
        )


if __name__ == "__main__":
    main()
