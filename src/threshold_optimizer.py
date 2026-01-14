"""
Threshold Optimizer for Skin Disease Classification.

This module provides tools to optimize classification thresholds,
particularly for safety-critical classes like melanoma where
high recall (sensitivity) is crucial to avoid missing diagnoses.

In medical diagnosis:
- False Negative (missing a melanoma) = VERY DANGEROUS
- False Positive (extra biopsy) = Acceptable (minor inconvenience)

Therefore, we optimize thresholds to maximize recall for dangerous classes.
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    f1_score,
    recall_score,
    precision_score
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


# Define which classes are safety-critical (high recall priority)
# These are pre-cancerous or cancerous lesions where missing diagnosis is dangerous
CRITICAL_CLASSES = {
    'mel': 0.85,   # Melanoma - most dangerous
    'bcc': 0.80,   # Basal cell carcinoma
    'akiec': 0.80, # Actinic keratosis
}

# Benign or less critical classes (balanced precision/recall)
NON_CRITICAL_CLASSES = ['nv', 'bkl', 'df', 'vasc']

# Configuration paths
DATA_DIR = Path("data/split")
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# Import Keras for standalone execution
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class ThresholdOptimizer:
    """
    Optimizes classification thresholds for multi-class problems
    with special focus on safety-critical classes.
    """
    
    def __init__(
        self,
        class_names: List[str],
        critical_classes: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the threshold optimizer.
        
        Args:
            class_names: List of class names in order of class indices.
            critical_classes: Dict mapping class names to minimum recall targets.
                            If None, uses default CRITICAL_CLASSES.
        """
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.critical_classes = critical_classes or CRITICAL_CLASSES
        
        # Default thresholds (equal for all classes)
        self.thresholds = {name: 0.5 for name in class_names}
        
        # Optimized thresholds will be stored here
        self.optimized_thresholds = None
        
    def find_threshold_for_target_recall(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_idx: int,
        target_recall: float,
        min_precision: float = 0.10
    ) -> Tuple[float, float, float]:
        """
        Find the threshold that achieves a target recall for a specific class.
        
        Args:
            y_true: True labels (one-hot encoded or class indices).
            y_proba: Predicted probabilities for all classes.
            class_idx: Index of the class to optimize.
            target_recall: Minimum recall to achieve.
            min_precision: Minimum acceptable precision (to avoid trivial solutions).
            
        Returns:
            Tuple of (optimal_threshold, achieved_recall, achieved_precision)
        """
        # Convert y_true to binary for this class
        if y_true.ndim == 1:
            y_binary = (y_true == class_idx).astype(int)
        else:
            y_binary = y_true[:, class_idx]
        
        # Get probabilities for this class
        class_proba = y_proba[:, class_idx]
        
        # Compute precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            y_binary, class_proba
        )
        
        # Find thresholds that meet the target recall
        valid_indices = np.where(
            (recalls[:-1] >= target_recall) & 
            (precisions[:-1] >= min_precision)
        )[0]
        
        if len(valid_indices) == 0:
            # Cannot achieve target recall with minimum precision
            # Find the best recall we can achieve
            best_idx = np.argmax(recalls[:-1])
            return thresholds[best_idx], recalls[best_idx], precisions[best_idx]
        
        # Among valid thresholds, pick the one with highest precision
        best_idx = valid_indices[np.argmax(precisions[:-1][valid_indices])]
        
        return thresholds[best_idx], recalls[best_idx], precisions[best_idx]
    
    def find_threshold_for_max_f1(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_idx: int
    ) -> Tuple[float, float]:
        """
        Find the threshold that maximizes F1 score for a specific class.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            class_idx: Index of the class to optimize.
            
        Returns:
            Tuple of (optimal_threshold, best_f1)
        """
        # Convert y_true to binary for this class
        if y_true.ndim == 1:
            y_binary = (y_true == class_idx).astype(int)
        else:
            y_binary = y_true[:, class_idx]
        
        class_proba = y_proba[:, class_idx]
        
        # Compute precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(
            y_binary, class_proba
        )
        
        # Calculate F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores[:-1])
        
        return thresholds[best_idx], f1_scores[best_idx]
    
    def optimize_all_thresholds(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Optimize thresholds for all classes.
        
        For critical classes: Optimize for target recall.
        For other classes: Optimize for maximum F1.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            verbose: Whether to print optimization results.
            
        Returns:
            Dictionary mapping class names to optimized thresholds.
        """
        if verbose:
            print("\n" + "=" * 60)
            print("THRESHOLD OPTIMIZATION")
            print("=" * 60)
        
        optimized = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            if class_name in self.critical_classes:
                # Safety-critical class: optimize for recall
                target_recall = self.critical_classes[class_name]
                threshold, recall, precision = self.find_threshold_for_target_recall(
                    y_true, y_proba, class_idx, target_recall
                )
                
                if verbose:
                    print(f"\n🔴 {class_name.upper()} (CRITICAL - Target Recall: {target_recall:.0%})")
                    print(f"   Optimized Threshold: {threshold:.4f}")
                    print(f"   Achieved Recall:     {recall:.4f} ({'✓' if recall >= target_recall else '✗'})")
                    print(f"   Achieved Precision:  {precision:.4f}")
                    
            else:
                # Non-critical class: optimize for F1
                threshold, f1 = self.find_threshold_for_max_f1(
                    y_true, y_proba, class_idx
                )
                
                if verbose:
                    print(f"\n🟢 {class_name} (Standard - Max F1)")
                    print(f"   Optimized Threshold: {threshold:.4f}")
                    print(f"   Best F1 Score:       {f1:.4f}")
            
            optimized[class_name] = float(threshold)
        
        self.optimized_thresholds = optimized
        return optimized
    
    def predict_with_thresholds(
        self,
        y_proba: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
        prioritize_critical: bool = True
    ) -> np.ndarray:
        """
        Make predictions using optimized thresholds.
        
        Args:
            y_proba: Predicted probabilities.
            thresholds: Thresholds to use. If None, uses optimized thresholds.
            prioritize_critical: If True and multiple classes exceed threshold,
                               prefer critical classes.
        
        Returns:
            Array of predicted class indices.
        """
        if thresholds is None:
            thresholds = self.optimized_thresholds or self.thresholds
        
        n_samples = y_proba.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            probs = y_proba[i]
            
            # Check which classes exceed their thresholds
            exceeds_threshold = {}
            for class_idx, class_name in enumerate(self.class_names):
                if probs[class_idx] >= thresholds[class_name]:
                    exceeds_threshold[class_idx] = probs[class_idx]
            
            if not exceeds_threshold:
                # No class exceeds threshold, use argmax
                predictions[i] = np.argmax(probs)
            elif prioritize_critical:
                # Check if any critical class exceeds threshold
                critical_exceeds = {
                    idx: prob for idx, prob in exceeds_threshold.items()
                    if self.class_names[idx] in self.critical_classes
                }
                
                if critical_exceeds:
                    # Pick the critical class with highest probability
                    predictions[i] = max(critical_exceeds, key=critical_exceeds.get)
                else:
                    # Pick the class with highest probability among those exceeding
                    predictions[i] = max(exceeds_threshold, key=exceeds_threshold.get)
            else:
                # Pick the class with highest probability among those exceeding
                predictions[i] = max(exceeds_threshold, key=exceeds_threshold.get)
        
        return predictions
    
    def save_thresholds(self, filepath: str):
        """Save optimized thresholds to a JSON file."""
        thresholds = self.optimized_thresholds or self.thresholds
        
        data = {
            'class_names': self.class_names,
            'thresholds': thresholds,
            'critical_classes': {k: v for k, v in self.critical_classes.items()},
            'description': 'Optimized thresholds for safety-critical classification'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Thresholds saved to: {filepath}")
    
    def load_thresholds(self, filepath: str):
        """Load thresholds from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.optimized_thresholds = data['thresholds']
        print(f"Thresholds loaded from: {filepath}")
        
        return self.optimized_thresholds


def compute_safety_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    critical_classes: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Compute safety-focused metrics for medical classification.
    
    These metrics emphasize recall (sensitivity) for dangerous conditions
    and provide clinical decision support information.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Prediction probabilities.
        class_names: List of class names.
        critical_classes: Dict of critical classes and target recalls.
        
    Returns:
        Dictionary containing safety metrics.
    """
    critical_classes = critical_classes or CRITICAL_CLASSES
    
    metrics = {
        'overall': {},
        'per_class': {},
        'safety_summary': {}
    }
    
    # Per-class metrics
    for class_idx, class_name in enumerate(class_names):
        y_binary_true = (y_true == class_idx).astype(int)
        y_binary_pred = (y_pred == class_idx).astype(int)
        
        tp = np.sum((y_binary_true == 1) & (y_binary_pred == 1))
        fn = np.sum((y_binary_true == 1) & (y_binary_pred == 0))
        fp = np.sum((y_binary_true == 0) & (y_binary_pred == 1))
        tn = np.sum((y_binary_true == 0) & (y_binary_pred == 0))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        is_critical = class_name in critical_classes
        target_recall = critical_classes.get(class_name, None)
        meets_target = sensitivity >= target_recall if target_recall else None
        
        metrics['per_class'][class_name] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'ppv': float(ppv),
            'npv': float(npv),
            'true_positives': int(tp),
            'false_negatives': int(fn),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'is_critical': is_critical,
            'target_sensitivity': target_recall,
            'meets_safety_target': meets_target
        }
    
    # Safety summary
    critical_recalls = []
    critical_meets_target = []
    
    for class_name, target in critical_classes.items():
        if class_name in metrics['per_class']:
            sensitivity = metrics['per_class'][class_name]['sensitivity']
            critical_recalls.append(sensitivity)
            critical_meets_target.append(sensitivity >= target)
    
    metrics['safety_summary'] = {
        'all_critical_targets_met': all(critical_meets_target) if critical_meets_target else True,
        'critical_classes_avg_sensitivity': float(np.mean(critical_recalls)) if critical_recalls else None,
        'min_critical_sensitivity': float(np.min(critical_recalls)) if critical_recalls else None,
        'critical_classes_status': {
            class_name: {
                'target': target,
                'achieved': metrics['per_class'].get(class_name, {}).get('sensitivity', 0),
                'gap': metrics['per_class'].get(class_name, {}).get('sensitivity', 0) - target
            }
            for class_name, target in critical_classes.items()
            if class_name in metrics['per_class']
        }
    }
    
    return metrics


def print_safety_report(metrics: Dict, class_names: List[str]):
    """
    Print a formatted safety report for clinical review.
    
    Args:
        metrics: Safety metrics dictionary.
        class_names: List of class names.
    """
    print("\n" + "=" * 70)
    print("🏥 CLINICAL SAFETY REPORT")
    print("=" * 70)
    
    # Critical classes first
    print("\n📊 SAFETY-CRITICAL CLASSES (High Sensitivity Required)")
    print("-" * 70)
    print(f"{'Class':<10} {'Sensitivity':<12} {'Target':<10} {'Status':<10} {'FN (Missed)':<12}")
    print("-" * 70)
    
    for class_name in class_names:
        if class_name in CRITICAL_CLASSES:
            m = metrics['per_class'][class_name]
            status = "✅ PASS" if m['meets_safety_target'] else "❌ FAIL"
            print(f"{class_name:<10} {m['sensitivity']:<12.2%} {m['target_sensitivity']:<10.0%} {status:<10} {m['false_negatives']:<12}")
    
    # Other classes
    print("\n📊 OTHER CLASSES (Balanced Metrics)")
    print("-" * 70)
    print(f"{'Class':<10} {'Sensitivity':<12} {'Specificity':<12} {'PPV':<10} {'NPV':<10}")
    print("-" * 70)
    
    for class_name in class_names:
        if class_name not in CRITICAL_CLASSES:
            m = metrics['per_class'][class_name]
            print(f"{class_name:<10} {m['sensitivity']:<12.2%} {m['specificity']:<12.2%} {m['ppv']:<10.2%} {m['npv']:<10.2%}")
    
    # Summary
    summary = metrics['safety_summary']
    print("\n" + "=" * 70)
    print("📋 SAFETY SUMMARY")
    print("=" * 70)
    
    overall_status = "✅ ALL TARGETS MET" if summary['all_critical_targets_met'] else "⚠️ TARGETS NOT MET"
    print(f"\nOverall Safety Status: {overall_status}")
    print(f"Average Critical Sensitivity: {summary['critical_classes_avg_sensitivity']:.2%}")
    print(f"Minimum Critical Sensitivity: {summary['min_critical_sensitivity']:.2%}")
    
    if not summary['all_critical_targets_met']:
        print("\n⚠️ ATTENTION: The following critical classes need improvement:")
        for class_name, status in summary['critical_classes_status'].items():
            if status['gap'] < 0:
                print(f"   - {class_name}: {status['achieved']:.2%} (need +{abs(status['gap']):.2%} to reach {status['target']:.0%})")
    
    print("\n" + "=" * 70)


# =============================================================================
# STANDALONE EXECUTION UTILITIES
# =============================================================================

def load_data_and_model():
    print("Loading validation data and model...")
    # Load Model with custom objects if needed
    try:
        from tensorflow.keras.losses import CategoricalFocalCrossentropy
    except ImportError:
        class CategoricalFocalCrossentropy(keras.losses.Loss):
            def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
                super().__init__(**kwargs)
            def call(self, y_true, y_pred): return tf.reduce_sum(y_true)

    model_path = MODEL_DIR / "best_model_finetuned.keras"
    if not model_path.exists(): model_path = MODEL_DIR / "best_model.keras"
    
    try:
        model = keras.models.load_model(model_path, custom_objects={'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy})
    except:
        model = keras.models.load_model(model_path, compile=False)

    # Load Classes
    with open(MODEL_DIR / "class_mapping.json") as f:
        mapping = json.load(f)
        classes = [mapping[str(i)] for i in range(len(mapping))]
    
    # Load Validation Data (for optimization)
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=(224, 224), batch_size=32, shuffle=False)
    
    # Load Test Data (for final evaluation)
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=(224, 224), batch_size=32, shuffle=False)
        
    return model, classes, val_ds, test_ds

def predict_dataset(model, ds):
    print("Generating predictions...")
    y_probs = model.predict(ds, verbose=1)
    y_true = np.concatenate([y for x, y in ds], axis=0)
    # Convert one-hot to sparse if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    return y_probs, y_true

if __name__ == "__main__":
    # 1. Setup
    model, class_names, val_ds, test_ds = load_data_and_model()
    
    # 2. Get Validation Predictions
    print("\n--- Optimizing Thresholds on Validation Set ---")
    val_probs, val_labels = predict_dataset(model, val_ds)
    
    # 3. Optimize
    optimizer = ThresholdOptimizer(class_names)
    opt_thresholds = optimizer.optimize_all_thresholds(val_labels, val_probs)
        
    # Save thresholds
    optimizer.save_thresholds(RESULTS_DIR / "optimized_thresholds.json")
        
    # 4. Evaluate on Test Set
    print("\n" + "="*60)
    print("EVALUATION ON TEST SET WITH OPTIMIZED THRESHOLDS")
    print("="*60)
    
    test_probs, test_labels = predict_dataset(model, test_ds)
    
    # Standard (Argmax)
    pred_std = np.argmax(test_probs, axis=1)
    acc_std = np.mean(pred_std == test_labels)
    f1_std = f1_score(test_labels, pred_std, average='weighted')
    print(f"\nStandard Accuracy (Argmax):      {acc_std:.4f}")
    print(f"Standard F1 Score (Weighted):    {f1_std:.4f}")
    
    # Optimized
    pred_opt = optimizer.predict_with_thresholds(test_probs, opt_thresholds)
    acc_opt = np.mean(pred_opt == test_labels)
    f1_opt = f1_score(test_labels, pred_opt, average='weighted')
    print(f"Optimized Accuracy (Scaled):     {acc_opt:.4f}")
    print(f"Optimized F1 Score (Weighted):   {f1_opt:.4f}")
    
    print("\n--- Optimized Classification Report ---")
    print(classification_report(test_labels, pred_opt, target_names=class_names))
    
    # Safety Report
    metrics = compute_safety_metrics(test_labels, pred_opt, test_probs, class_names)
    print_safety_report(metrics, class_names)
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, pred_opt)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
    plt.title("Optimized Confusion Matrix")
    plt.savefig(RESULTS_DIR / "confusion_matrix_optimized.png")
    print(f"\nSaved matrix to {RESULTS_DIR}/confusion_matrix_optimized.png")
