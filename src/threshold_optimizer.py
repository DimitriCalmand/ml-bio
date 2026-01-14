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
    'mel': 0.90,   # Melanoma - most dangerous, need 90%+ recall
    'bcc': 0.85,   # Basal cell carcinoma - cancerous
    'akiec': 0.85, # Actinic keratosis - pre-cancerous
}

# Benign or less critical classes (balanced precision/recall)
NON_CRITICAL_CLASSES = ['nv', 'bkl', 'df', 'vasc']


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


if __name__ == "__main__":
    # Example usage
    print("Threshold Optimizer Module")
    print("-" * 40)
    print("This module provides tools for optimizing classification thresholds")
    print("with special focus on safety-critical medical classes.")
    print("\nCritical classes with target recalls:")
    for cls, target in CRITICAL_CLASSES.items():
        print(f"  - {cls}: {target:.0%}")
