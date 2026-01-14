#!/usr/bin/env python3
"""
Command-line interface for generating model explanations.

This script provides an easy-to-use interface for generating
Grad-CAM, Grad-CAM++, and LIME explanations for skin lesion images.

Usage Examples:
    # Basic explanation
    python src/generate_explanation.py path/to/image.jpg
    
    # Clinical report
    python src/generate_explanation.py path/to/image.jpg --clinical
    
    # Specific methods
    python src/generate_explanation.py path/to/image.jpg --methods gradcam lime
    
    # Batch processing
    python src/generate_explanation.py path/to/folder/ --batch
    
    # High quality LIME
    python src/generate_explanation.py path/to/image.jpg --lime-samples 2000
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
from pathlib import Path
import json
from typing import List

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def setup_gpu():
    """Configure GPU memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU detected: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("ℹ Using CPU")


def load_model_and_config(model_path: str = None):
    """Load the trained model and configuration."""
    model_dir = Path("models")
    
    if model_path:
        model_file = Path(model_path)
    else:
        # Try finetuned model first
        model_file = model_dir / "best_model_finetuned.keras"
        if not model_file.exists():
            model_file = model_dir / "best_model.keras"
    
    if not model_file.exists():
        print(f"ERROR: Model not found at {model_file}")
        print("Please train the model first with: python src/train.py")
        sys.exit(1)
    
    print(f"Loading model: {model_file}")
    model = keras.models.load_model(model_file)
    
    # Load class mapping
    class_mapping_file = model_dir / "class_mapping.json"
    if not class_mapping_file.exists():
        print(f"ERROR: Class mapping not found at {class_mapping_file}")
        sys.exit(1)
    
    with open(class_mapping_file, 'r') as f:
        class_mapping = json.load(f)
    
    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    
    return model, class_names


def get_images_from_path(input_path: str) -> List[Path]:
    """Get list of image files from path (file or directory)."""
    path = Path(input_path)
    
    if path.is_file():
        return [path]
    elif path.is_dir():
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(path.glob(ext))
            images.extend(path.glob(ext.upper()))
        return sorted(images)
    else:
        print(f"ERROR: Path not found: {input_path}")
        sys.exit(1)


def generate_single_explanation(
    model,
    class_names: List[str],
    image_path: Path,
    output_dir: Path,
    methods: List[str],
    lime_samples: int,
    clinical: bool
):
    """Generate explanation for a single image."""
    from explain import ExplanationGenerator, create_clinical_explanation
    
    output_file = output_dir / f"explanation_{image_path.stem}.png"
    
    if clinical:
        print(f"\n  Generating clinical report...")
        create_clinical_explanation(
            model,
            str(image_path),
            class_names,
            output_path=output_file
        )
    else:
        print(f"\n  Generating explanations with: {', '.join(methods)}")
        generator = ExplanationGenerator(model, class_names)
        result = generator.explain_image(
            str(image_path),
            methods=methods,
            lime_samples=lime_samples
        )
        generator.create_explanation_figure(result, save_path=output_file)
        
        # Print prediction summary
        pred = result['prediction']
        print(f"  Prediction: {pred['class_name']} ({pred['confidence']:.1%})")
    
    plt.close('all')
    print(f"  ✓ Saved: {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate model explanations for skin lesion images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg                    # Basic explanation
  %(prog)s image.jpg --clinical         # Clinical report
  %(prog)s folder/ --batch              # Process all images in folder
  %(prog)s image.jpg -o output.png      # Custom output path
  %(prog)s image.jpg --methods gradcam  # Only Grad-CAM
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to image file or directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (file for single image, directory for batch)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (default: models/best_model_finetuned.keras)"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        default=['gradcam', 'lime'],
        choices=['gradcam', 'gradcam++', 'lime'],
        help="Explanation methods to use (default: gradcam lime)"
    )
    
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=500,
        help="Number of samples for LIME (default: 500, higher = better quality)"
    )
    
    parser.add_argument(
        "--clinical",
        action="store_true",
        help="Generate clinical-grade explanation report"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in directory"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_gpu()
    
    # Load model
    model, class_names = load_model_and_config(args.model)
    print(f"Classes: {class_names}")
    
    # Get images
    images = get_images_from_path(args.input)
    
    if not images:
        print("ERROR: No images found")
        sys.exit(1)
    
    if len(images) > 1 and not args.batch:
        print(f"Found {len(images)} images. Use --batch to process all.")
        images = images[:1]
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
        if len(images) == 1 and not output_dir.suffix:
            output_dir.mkdir(parents=True, exist_ok=True)
        elif len(images) == 1:
            output_dir = output_dir.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("results/explanations")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {len(images)} image(s)...")
    print(f"Output directory: {output_dir}")
    
    # Process images
    results = []
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{len(images)}] {img_path.name}")
        
        try:
            output_file = generate_single_explanation(
                model=model,
                class_names=class_names,
                image_path=img_path,
                output_dir=output_dir,
                methods=args.methods,
                lime_samples=args.lime_samples,
                clinical=args.clinical
            )
            results.append({'image': str(img_path), 'output': str(output_file), 'status': 'success'})
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({'image': str(img_path), 'error': str(e), 'status': 'failed'})
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"Processed: {len(results)} images")
    print(f"Success: {success}")
    print(f"Failed: {len(results) - success}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
