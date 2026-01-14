"""
Model Explainability Module for Skin Disease Classification.

This module provides comprehensive explainability tools including:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- Grad-CAM++: Improved Grad-CAM with better localization
- LIME: Local Interpretable Model-agnostic Explanations
- Integrated visualization utilities

Compatible with Keras 3 and TensorFlow 2.16+

Author: ML-Bio Project
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union, Callable
import warnings

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def load_and_preprocess_image(
    img_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess an image for model inference.
    
    Args:
        img_path: Path to the image file.
        target_size: Target size (height, width) for the model.
        
    Returns:
        Tuple of (preprocessed_array, original_image):
            - preprocessed_array: Shape (1, H, W, 3), ready for model
            - original_image: Shape (H, W, 3), original pixel values [0, 255]
    """
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    
    # Keep original for visualization
    original = img_array.copy()
    
    # Add batch dimension
    preprocessed = np.expand_dims(img_array, axis=0)
    
    return preprocessed, original


def get_img_array(img_path: Union[str, Path], size: Tuple[int, int]) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Load and preprocess an image for model inference.
    
    Args:
        img_path: Path to the image file.
        size: Target size (height, width) for the model.
        
    Returns:
        Preprocessed image array with shape (1, H, W, 3).
    """
    preprocessed, _ = load_and_preprocess_image(img_path, size)
    return preprocessed


# =============================================================================
# GRAD-CAM IMPLEMENTATION (Keras 3 Compatible)
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    This class provides a robust implementation of Grad-CAM that works with
    Keras 3 and nested models (e.g., transfer learning models with a backbone).
    
    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks 
        via Gradient-based Localization", ICCV 2017.
    """
    
    def __init__(
        self, 
        model: keras.Model,
        layer_name: Optional[str] = None
    ):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: Keras model to explain.
            layer_name: Name of the convolutional layer to use for CAM.
                       If None, automatically finds the last conv layer.
        """
        self.model = model
        self._analyze_model_structure()
        self.layer_name = layer_name or self._find_target_layer()
        
    def _analyze_model_structure(self):
        """Analyze the model structure to find backbone and layers."""
        self.backbone = None
        self.backbone_idx = None
        self.pre_backbone_layers = []
        self.post_backbone_layers = []
        self._backbone_grad_model = None
        
        for idx, layer in enumerate(self.model.layers):
            # Check if this is a nested model (backbone like MobileNetV2)
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                self.backbone = layer
                self.backbone_idx = idx
                self.pre_backbone_layers = list(range(idx))
                self.post_backbone_layers = list(range(idx + 1, len(self.model.layers)))
                break
    
    def _find_target_layer(self) -> str:
        """
        Automatically find the last convolutional layer.
        
        Returns:
            Name of the last convolutional layer found.
        """
        # First, try to find in the backbone
        if self.backbone is not None:
            for sub_layer in reversed(self.backbone.layers):
                if self._is_conv_layer(sub_layer):
                    return f"{self.backbone.name}/{sub_layer.name}"
        
        # If not found in backbone, search in main model
        for layer in reversed(self.model.layers):
            if self._is_conv_layer(layer):
                return layer.name
                
        raise ValueError("No convolutional layer found in model")
    
    def _is_conv_layer(self, layer) -> bool:
        """Check if a layer is a convolutional layer."""
        layer_type = type(layer).__name__
        return layer_type in ['Conv2D', 'DepthwiseConv2D', 'SeparableConv2D']
    
    def _build_backbone_grad_model(self, sublayer_name: str) -> keras.Model:
        """
        Build a sub-model from backbone that outputs both conv layer and final output.
        
        This is the key to Keras 3 compatibility - we create a model directly
        from the backbone, not from the parent model.
        
        Args:
            sublayer_name: Name of the target conv layer within the backbone.
            
        Returns:
            Keras Model that outputs [conv_output, backbone_output].
        """
        if self._backbone_grad_model is not None:
            return self._backbone_grad_model
            
        target_layer = self.backbone.get_layer(sublayer_name)
        self._backbone_grad_model = keras.Model(
            inputs=self.backbone.input,
            outputs=[target_layer.output, self.backbone.output]
        )
        return self._backbone_grad_model
    
    def _get_conv_layer_output(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get the convolutional layer output and model predictions.
        
        This method handles nested models properly for Keras 3 by:
        1. Creating a sub-model from the backbone (not the parent model)
        2. Processing pre-backbone layers (skipping InputLayer)
        3. Running the backbone sub-model
        4. Processing post-backbone layers
        
        Args:
            inputs: Input tensor.
            
        Returns:
            Tuple of (conv_output, predictions).
        """
        if '/' in self.layer_name:
            # Nested model case (e.g., "mobilenetv2_1.00_224/Conv_1")
            backbone_name, sublayer_name = self.layer_name.split('/', 1)
            
            # Build the backbone sub-model (cached after first call)
            backbone_grad_model = self._build_backbone_grad_model(sublayer_name)
            
            # Process through layers before backbone (skip InputLayer)
            x = inputs
            for idx in self.pre_backbone_layers:
                layer = self.model.layers[idx]
                # Skip InputLayer - it's just a placeholder, not callable in Keras 3
                if type(layer).__name__ == 'InputLayer':
                    continue
                x = layer(x)
            
            # Get conv outputs and backbone outputs from sub-model
            conv_outputs, backbone_outputs = backbone_grad_model(x)
            
            # Continue through remaining layers (GAP, Dropout, Dense, etc.)
            x = backbone_outputs
            for idx in self.post_backbone_layers:
                x = self.model.layers[idx](x, training=False)
            
            predictions = x
            return conv_outputs, predictions
        else:
            # Simple case - layer is in main model (no nested backbone)
            # Build a single grad model
            target_layer = self.model.get_layer(self.layer_name)
            grad_model = keras.Model(
                inputs=self.model.input,
                outputs=[target_layer.output, self.model.output]
            )
            conv_outputs, predictions = grad_model(inputs)
            return conv_outputs, predictions
    
    def compute_heatmap(
        self,
        img_array: np.ndarray,
        pred_index: Optional[int] = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            img_array: Preprocessed image array with shape (1, H, W, 3).
            pred_index: Index of the class to explain. If None, uses the
                       predicted class (argmax).
            eps: Small constant for numerical stability.
            
        Returns:
            Heatmap as a 2D numpy array with values in [0, 1].
        """
        # Cast to float32 for gradient computation
        img_tensor = tf.cast(img_array, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            
            # Get conv output and predictions
            conv_outputs, predictions = self._get_conv_layer_output(img_tensor)
            
            if conv_outputs is None:
                warnings.warn(f"Could not find conv output for layer {self.layer_name}")
                return np.zeros((7, 7))
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Compute gradients of the class output with respect to conv outputs
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            warnings.warn(
                f"Gradients are None for layer {self.layer_name}. "
                "This might indicate the layer is not connected properly."
            )
            return np.zeros((7, 7))
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy()
    
    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet"
    ) -> np.ndarray:
        """
        Overlay heatmap on the original image.
        
        Args:
            heatmap: 2D heatmap array with values in [0, 1].
            original_image: Original image array (H, W, 3) in [0, 255].
            alpha: Opacity of the heatmap overlay.
            colormap: Matplotlib colormap name.
            
        Returns:
            Superimposed image as uint8 array (H, W, 3).
        """
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(
            heatmap, 
            (original_image.shape[1], original_image.shape[0])
        )
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Superimpose
        original_uint8 = original_image.astype(np.uint8)
        superimposed = cv2.addWeighted(
            original_uint8, 1 - alpha,
            heatmap_colored, alpha,
            0
        )
        
        return superimposed


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ implementation with improved localization.
    
    Grad-CAM++ uses a weighted combination of positive partial derivatives
    as weights instead of global average pooled gradients, providing better
    localization for multiple occurrences of objects.
    
    Reference:
        Chattopadhyay et al., "Grad-CAM++: Generalized Gradient-based Visual 
        Explanations for Deep Convolutional Networks", WACV 2018.
    """
    
    def compute_heatmap(
        self,
        img_array: np.ndarray,
        pred_index: Optional[int] = None,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Grad-CAM++ heatmap.
        
        Args:
            img_array: Preprocessed image array with shape (1, H, W, 3).
            pred_index: Index of the class to explain.
            eps: Small constant for numerical stability.
            
        Returns:
            Heatmap as a 2D numpy array with values in [0, 1].
        """
        img_tensor = tf.cast(img_array, tf.float32)
        
        # For Grad-CAM++, we need second and third order gradients
        # But this is complex with nested models, so we use an approximation
        # that works well in practice
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(img_tensor)
            
            conv_outputs, predictions = self._get_conv_layer_output(img_tensor)
            
            if conv_outputs is None:
                return np.zeros((7, 7))
            
            # Watch conv_outputs for second-order gradients
            tape.watch(conv_outputs)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # First derivative
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            return np.zeros((7, 7))
        
        # For Grad-CAM++, we weight positive gradients by their squared values
        # This is an approximation that avoids the need for higher-order derivatives
        
        # ReLU on gradients
        grads_relu = tf.maximum(grads, 0)
        
        # Square of gradients for weighting
        grads_squared = grads ** 2
        grads_cubed = grads ** 3
        
        # Compute alpha (simplified version)
        sum_activations = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
        alpha_num = grads_squared
        alpha_denom = 2.0 * grads_squared + sum_activations * grads_cubed + eps
        alpha = alpha_num / alpha_denom
        
        # Where gradients are zero, alpha should be zero
        alpha = tf.where(grads != 0, alpha, tf.zeros_like(alpha))
        
        # Weights
        weights = tf.reduce_sum(alpha * grads_relu, axis=(1, 2))
        
        # Generate heatmap
        heatmap = tf.reduce_sum(conv_outputs[0] * weights[0], axis=-1)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        del tape  # Release the persistent tape
        
        return heatmap.numpy()


# =============================================================================
# LEGACY GRAD-CAM FUNCTIONS (Backward compatibility)
# =============================================================================

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: str = "out_relu",
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Legacy function for Grad-CAM heatmap generation.
    
    This function provides backward compatibility with the original interface.
    For new code, use the GradCAM class instead.
    
    Args:
        img_array: Preprocessed image array with shape (1, H, W, 3).
        model: Keras model to explain.
        last_conv_layer_name: Name of the last convolutional layer.
        pred_index: Index of the class to explain.
        
    Returns:
        Heatmap as a 2D numpy array with values in [0, 1].
    """
    try:
        gradcam = GradCAM(model, None)  # Auto-detect layer
        return gradcam.compute_heatmap(img_array, pred_index)
    except Exception as e:
        warnings.warn(f"GradCAM failed: {e}")
        return np.zeros((7, 7))


def save_and_display_gradcam(
    img_path: Union[str, Path],
    heatmap: np.ndarray,
    cam_path: str = "cam.jpg",
    alpha: float = 0.4
) -> np.ndarray:
    """
    Legacy function to save and display Grad-CAM overlay.
    
    Args:
        img_path: Path to the original image.
        heatmap: Grad-CAM heatmap.
        cam_path: Path to save the result.
        alpha: Opacity of the heatmap overlay.
        
    Returns:
        Superimposed image as PIL Image.
    """
    # Load original image
    img = keras.preprocessing.image.load_img(img_path)
    img_array = keras.preprocessing.image.img_to_array(img)
    
    # Create overlay
    gradcam = GradCAM.__new__(GradCAM)
    result = gradcam.overlay_heatmap(heatmap, img_array, alpha)
    
    # Save
    result_img = keras.preprocessing.image.array_to_img(result)
    result_img.save(cam_path)
    
    return result_img


# =============================================================================
# LIME IMPLEMENTATION
# =============================================================================

class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for images.
    
    This class provides a comprehensive LIME implementation with various
    visualization options for medical imaging interpretability.
    
    Reference:
        Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions 
        of Any Classifier", KDD 2016.
    """
    
    def __init__(
        self,
        model: keras.Model,
        class_names: List[str],
        preprocess_fn: Optional[Callable] = None
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Keras model to explain.
            class_names: List of class names in order of class indices.
            preprocess_fn: Optional preprocessing function for images.
        """
        self.model = model
        self.class_names = class_names
        self.preprocess_fn = preprocess_fn
        
        # Import lime here to avoid import errors if not installed
        try:
            from lime import lime_image
            self.lime_image = lime_image
        except ImportError:
            raise ImportError(
                "LIME is not installed. Install it with: pip install lime"
            )
        
        self.explainer = self.lime_image.LimeImageExplainer()
    
    def _predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function wrapper for LIME.
        
        Args:
            images: Batch of images (N, H, W, 3) in float format.
            
        Returns:
            Prediction probabilities (N, num_classes).
        """
        # LIME may pass images in different formats
        if images.max() > 1.0:
            images = images.astype(np.float32)
        else:
            images = (images * 255).astype(np.float32)
        
        if self.preprocess_fn:
            images = self.preprocess_fn(images)
        
        return self.model.predict(images, verbose=0)
    
    def explain(
        self,
        img_array: np.ndarray,
        num_samples: int = 1000,
        num_features: int = 10,
        top_labels: int = 3,
        hide_color: Optional[int] = 0,
        segmentation_fn: Optional[Callable] = None
    ) -> 'LIMEExplanation':
        """
        Generate LIME explanation for an image.
        
        Args:
            img_array: Image array with shape (1, H, W, 3) or (H, W, 3).
            num_samples: Number of perturbed samples to generate.
            num_features: Number of superpixels to include in explanation.
            top_labels: Number of top predicted labels to explain.
            hide_color: Color value for hidden superpixels (0=black, 1=white).
            segmentation_fn: Custom segmentation function.
            
        Returns:
            LIMEExplanation object with visualization methods.
        """
        # Ensure correct shape
        if img_array.ndim == 4:
            img_array = img_array[0]
        
        # Normalize to [0, 1] for LIME
        if img_array.max() > 1.0:
            img_normalized = img_array / 255.0
        else:
            img_normalized = img_array
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            img_normalized,
            self._predict_fn,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=num_samples,
            segmentation_fn=segmentation_fn
        )
        
        return LIMEExplanation(explanation, self.class_names, img_normalized)


class LIMEExplanation:
    """
    Container for LIME explanation with visualization methods.
    """
    
    def __init__(
        self,
        explanation,
        class_names: List[str],
        original_image: np.ndarray
    ):
        """
        Initialize LIME explanation container.
        
        Args:
            explanation: LIME explanation object.
            class_names: List of class names.
            original_image: Original image array (H, W, 3) normalized to [0, 1].
        """
        self.explanation = explanation
        self.class_names = class_names
        self.original_image = original_image
        self.top_labels = explanation.top_labels
    
    def get_image_and_mask(
        self,
        label: Optional[int] = None,
        positive_only: bool = True,
        num_features: int = 5,
        hide_rest: bool = False,
        negative_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the explanation image and mask for a specific class.
        
        Args:
            label: Class index to explain. If None, uses top predicted class.
            positive_only: If True, only show features that support the class.
            num_features: Number of top features to show.
            hide_rest: If True, hide areas not in explanation.
            negative_only: If True, only show features against the class.
            
        Returns:
            Tuple of (explained_image, mask).
        """
        if label is None:
            label = self.top_labels[0]
        
        # Handle negative_only parameter for older lime versions
        try:
            return self.explanation.get_image_and_mask(
                label,
                positive_only=positive_only,
                num_features=num_features,
                hide_rest=hide_rest,
                negative_only=negative_only
            )
        except TypeError:
            # Older lime version without negative_only
            return self.explanation.get_image_and_mask(
                label,
                positive_only=positive_only,
                num_features=num_features,
                hide_rest=hide_rest
            )
    
    def visualize(
        self,
        label: Optional[int] = None,
        num_features: int = 5,
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """
        Create comprehensive LIME visualization.
        
        Args:
            label: Class index to explain.
            num_features: Number of features to highlight.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure.
        """
        from skimage.segmentation import mark_boundaries
        
        if label is None:
            label = self.top_labels[0]
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # 1. Original image
        axes[0].imshow(self.original_image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        # 2. Positive contributions only
        temp_pos, mask_pos = self.get_image_and_mask(
            label, positive_only=True, num_features=num_features
        )
        axes[1].imshow(mark_boundaries(temp_pos, mask_pos))
        axes[1].set_title(f"Pro '{self.class_names[label]}'", fontsize=12)
        axes[1].axis('off')
        
        # 3. All contributions (positive and negative)
        temp_all, mask_all = self.get_image_and_mask(
            label, positive_only=False, num_features=num_features
        )
        axes[2].imshow(mark_boundaries(temp_all, mask_all))
        axes[2].set_title(f"All contributions", fontsize=12)
        axes[2].axis('off')
        
        # 4. Highlighted regions only
        temp_focus, mask_focus = self.get_image_and_mask(
            label, positive_only=True, num_features=num_features, hide_rest=True
        )
        axes[3].imshow(temp_focus)
        axes[3].set_title("Key Regions", fontsize=12)
        axes[3].axis('off')
        
        plt.tight_layout()
        return fig
    
    def get_feature_importance(
        self,
        label: Optional[int] = None,
        num_features: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get feature importance scores for a class.
        
        Args:
            label: Class index.
            num_features: Number of features to return.
            
        Returns:
            List of (feature_id, importance) tuples sorted by absolute importance.
        """
        if label is None:
            label = self.top_labels[0]
        
        local_exp = self.explanation.local_exp[label]
        return sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:num_features]


# Legacy function for backward compatibility
def explain_with_lime(
    img_array: np.ndarray,
    model: keras.Model,
    class_names: List[str],
    num_features: int = 5,
    num_samples: int = 1000
) -> Tuple[np.ndarray, int]:
    """
    Legacy function for LIME explanation generation.
    
    Args:
        img_array: Preprocessed image array with shape (1, H, W, 3).
        model: Keras model.
        class_names: List of class names.
        num_features: Number of superpixels to highlight.
        num_samples: Number of perturbed samples.
        
    Returns:
        Tuple of (boundary_image, predicted_label_index).
    """
    from skimage.segmentation import mark_boundaries
    
    explainer = LIMEExplainer(model, class_names)
    explanation = explainer.explain(
        img_array,
        num_samples=num_samples,
        num_features=num_features
    )
    
    temp, mask = explanation.get_image_and_mask(
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    
    boundary_img = mark_boundaries(temp, mask)
    
    return boundary_img, explanation.top_labels[0]


# =============================================================================
# COMPREHENSIVE EXPLANATION GENERATOR
# =============================================================================

class ExplanationGenerator:
    """
    Comprehensive explanation generator combining multiple XAI methods.
    
    This class provides a unified interface for generating explanations
    using Grad-CAM, Grad-CAM++, and LIME, with various visualization options.
    """
    
    def __init__(
        self,
        model: keras.Model,
        class_names: List[str],
        target_layer: Optional[str] = None
    ):
        """
        Initialize the explanation generator.
        
        Args:
            model: Keras model to explain.
            class_names: List of class names.
            target_layer: Target layer for Grad-CAM. Auto-detected if None.
        """
        self.model = model
        self.class_names = class_names
        
        # Initialize explainers
        self.gradcam = GradCAM(model, target_layer)
        self.gradcam_pp = GradCAMPlusPlus(model, target_layer)
        self.lime_explainer = LIMEExplainer(model, class_names)
        
        self.target_layer = self.gradcam.layer_name
    
    def explain_image(
        self,
        img_path: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224),
        class_index: Optional[int] = None,
        methods: List[str] = ['gradcam', 'gradcam++', 'lime'],
        lime_samples: int = 500,
        lime_features: int = 5
    ) -> Dict:
        """
        Generate comprehensive explanations for an image.
        
        Args:
            img_path: Path to the image file.
            target_size: Target size for the model.
            class_index: Class to explain. If None, uses predicted class.
            methods: List of explanation methods to use.
            lime_samples: Number of samples for LIME.
            lime_features: Number of features for LIME.
            
        Returns:
            Dictionary containing:
                - 'prediction': Predicted class info
                - 'gradcam': Grad-CAM heatmap and overlay
                - 'gradcam++': Grad-CAM++ heatmap and overlay
                - 'lime': LIME explanation object
                - 'original': Original image
        """
        # Load and preprocess image
        img_array, original = load_and_preprocess_image(img_path, target_size)
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        pred_index = int(np.argmax(predictions[0]))
        pred_confidence = float(predictions[0][pred_index])
        
        if class_index is None:
            class_index = pred_index
        
        result = {
            'prediction': {
                'class_index': int(pred_index),
                'class_name': self.class_names[pred_index],
                'confidence': float(pred_confidence),
                'all_probabilities': {
                    self.class_names[i]: float(p) 
                    for i, p in enumerate(predictions[0])
                }
            },
            'original': original,
            'explained_class': {
                'index': class_index,
                'name': self.class_names[class_index]
            }
        }
        
        # Generate Grad-CAM
        if 'gradcam' in methods:
            heatmap = self.gradcam.compute_heatmap(img_array, class_index)
            overlay = self.gradcam.overlay_heatmap(heatmap, original)
            result['gradcam'] = {
                'heatmap': heatmap,
                'overlay': overlay
            }
        
        # Generate Grad-CAM++
        if 'gradcam++' in methods:
            heatmap_pp = self.gradcam_pp.compute_heatmap(img_array, class_index)
            overlay_pp = self.gradcam_pp.overlay_heatmap(heatmap_pp, original)
            result['gradcam++'] = {
                'heatmap': heatmap_pp,
                'overlay': overlay_pp
            }
        
        # Generate LIME
        if 'lime' in methods:
            lime_exp = self.lime_explainer.explain(
                img_array,
                num_samples=lime_samples,
                num_features=lime_features
            )
            result['lime'] = lime_exp
        
        return result
    
    def create_explanation_figure(
        self,
        explanation_result: Dict,
        figsize: Tuple[int, int] = (20, 10),
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Create a comprehensive visualization figure.
        
        Args:
            explanation_result: Result from explain_image().
            figsize: Figure size.
            save_path: Optional path to save the figure.
            
        Returns:
            Matplotlib figure.
        """
        from skimage.segmentation import mark_boundaries
        
        # Determine number of columns based on available methods
        n_cols = 1  # Original
        if 'gradcam' in explanation_result:
            n_cols += 1
        if 'gradcam++' in explanation_result:
            n_cols += 1
        if 'lime' in explanation_result:
            n_cols += 2  # LIME pro + LIME regions
        
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        col = 0
        pred = explanation_result['prediction']
        explained = explanation_result['explained_class']
        
        # Original image with prediction
        original = explanation_result['original'] / 255.0
        axes[col].imshow(original)
        axes[col].set_title(
            f"Original\n"
            f"Predicted: {pred['class_name']} ({pred['confidence']:.1%})",
            fontsize=11
        )
        axes[col].axis('off')
        col += 1
        
        # Grad-CAM
        if 'gradcam' in explanation_result:
            axes[col].imshow(explanation_result['gradcam']['overlay'])
            axes[col].set_title(
                f"Grad-CAM\n"
                f"Explaining: {explained['name']}",
                fontsize=11
            )
            axes[col].axis('off')
            col += 1
        
        # Grad-CAM++
        if 'gradcam++' in explanation_result:
            axes[col].imshow(explanation_result['gradcam++']['overlay'])
            axes[col].set_title(
                f"Grad-CAM++\n"
                f"Explaining: {explained['name']}",
                fontsize=11
            )
            axes[col].axis('off')
            col += 1
        
        # LIME
        if 'lime' in explanation_result:
            lime_exp = explanation_result['lime']
            
            # LIME positive features
            temp, mask = lime_exp.get_image_and_mask(
                label=explained['index'],
                positive_only=True,
                num_features=5
            )
            axes[col].imshow(mark_boundaries(temp, mask))
            axes[col].set_title(
                f"LIME Pro\n"
                f"For: {explained['name']}",
                fontsize=11
            )
            axes[col].axis('off')
            col += 1
            
            # LIME key regions
            temp_focus, _ = lime_exp.get_image_and_mask(
                label=explained['index'],
                positive_only=True,
                num_features=5,
                hide_rest=True
            )
            axes[col].imshow(temp_focus)
            axes[col].set_title(
                f"LIME Key Regions\n"
                f"For: {explained['name']}",
                fontsize=11
            )
            axes[col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation saved to: {save_path}")
        
        return fig
    
    def generate_batch_explanations(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224),
        methods: List[str] = ['gradcam', 'lime']
    ) -> List[Dict]:
        """
        Generate explanations for multiple images.
        
        Args:
            image_paths: List of image paths.
            output_dir: Directory to save results.
            target_size: Target size for the model.
            methods: Explanation methods to use.
            
        Returns:
            List of explanation results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, img_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {img_path}")
            
            try:
                result = self.explain_image(
                    img_path,
                    target_size=target_size,
                    methods=methods
                )
                
                # Save figure
                fig_path = output_dir / f"explanation_{Path(img_path).stem}.png"
                self.create_explanation_figure(result, save_path=fig_path)
                plt.close()
                
                result['image_path'] = str(img_path)
                result['figure_path'] = str(fig_path)
                results.append(result)
                
            except Exception as e:
                print(f"  Error: {e}")
                results.append({'image_path': str(img_path), 'error': str(e)})
        
        return results


# =============================================================================
# MEDICAL-SPECIFIC VISUALIZATION
# =============================================================================

def create_clinical_explanation(
    model: keras.Model,
    img_path: Union[str, Path],
    class_names: List[str],
    target_size: Tuple[int, int] = (224, 224),
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create a clinical-grade explanation figure suitable for medical reports.
    
    This function generates a comprehensive visualization with:
    - Original image
    - Prediction probabilities for all classes
    - Grad-CAM heatmap
    - LIME explanation
    - Risk assessment for critical classes
    
    Args:
        model: Trained Keras model.
        img_path: Path to the skin lesion image.
        class_names: List of class names.
        target_size: Target size for the model.
        output_path: Optional path to save the figure.
        
    Returns:
        Matplotlib figure.
    """
    from skimage.segmentation import mark_boundaries
    
    # Critical classes for skin cancer
    critical_classes = {'mel', 'bcc', 'akiec'}
    
    # Generate explanations
    generator = ExplanationGenerator(model, class_names)
    result = generator.explain_image(
        img_path,
        target_size=target_size,
        methods=['gradcam', 'lime'],
        lime_samples=300
    )
    
    # Create figure with specific layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    pred = result['prediction']
    
    # Row 1: Images
    # Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(result['original'] / 255.0)
    ax1.set_title("Original Lesion", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Grad-CAM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(result['gradcam']['overlay'])
    ax2.set_title("Grad-CAM Attention", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # LIME positive
    lime_exp = result['lime']
    temp, mask = lime_exp.get_image_and_mask(
        label=pred['class_index'],
        positive_only=True,
        num_features=5
    )
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mark_boundaries(temp, mask))
    ax3.set_title("LIME Key Features", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # LIME focus
    temp_focus, _ = lime_exp.get_image_and_mask(
        label=pred['class_index'],
        positive_only=True,
        num_features=5,
        hide_rest=True
    )
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(temp_focus)
    ax4.set_title("Isolated Features", fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Row 2: Probability distribution
    ax5 = fig.add_subplot(gs[1, :2])
    probs = pred['all_probabilities']
    colors = ['#ff6b6b' if cn in critical_classes else '#4ecdc4' for cn in class_names]
    bars = ax5.barh(list(probs.keys()), list(probs.values()), color=colors)
    ax5.set_xlim(0, 1)
    ax5.set_xlabel('Probability', fontsize=11)
    ax5.set_title('Classification Probabilities', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar, prob in zip(bars, probs.values()):
        ax5.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', va='center', fontsize=10)
    
    # Row 2 right: Risk assessment
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')
    
    # Calculate risk scores
    critical_risk = sum(probs.get(c, 0) for c in critical_classes if c in probs)
    
    risk_text = f"""
    ═══════════════════════════════════════
    DIAGNOSTIC SUMMARY
    ═══════════════════════════════════════
    
    Primary Prediction: {pred['class_name'].upper()}
    Confidence: {pred['confidence']:.1%}
    
    ─────────────────────────────────────
    RISK ASSESSMENT
    ─────────────────────────────────────
    
    Critical Classes Risk Score: {critical_risk:.1%}
    
    • Melanoma (mel): {probs.get('mel', 0):.1%}
    • Basal Cell Carcinoma (bcc): {probs.get('bcc', 0):.1%}
    • Actinic Keratosis (akiec): {probs.get('akiec', 0):.1%}
    
    ─────────────────────────────────────
    """
    
    if critical_risk > 0.3:
        risk_text += "\n    ⚠️  ELEVATED RISK: Further examination recommended"
    elif pred['class_name'] in critical_classes:
        risk_text += "\n    ⚠️  POSITIVE FOR CRITICAL CLASS: Biopsy recommended"
    else:
        risk_text += "\n    ✓  LOW RISK: Routine monitoring suggested"
    
    ax6.text(0.1, 0.9, risk_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Row 3: Interpretation guide
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    interpretation = """
    ═══════════════════════════════════════════════════════════════════════════════════════════════════════
    INTERPRETATION GUIDE
    ═══════════════════════════════════════════════════════════════════════════════════════════════════════
    
    • Grad-CAM: Shows which regions of the image the model focuses on. Warm colors (red/yellow) indicate 
      high attention areas that most influenced the classification decision.
      
    • LIME: Highlights superpixels that contributed to the prediction. Green boundaries show positive 
      contributions (supporting the predicted class), red boundaries show negative contributions.
      
    • Critical Classes (red in probability chart): mel (Melanoma), bcc (Basal Cell Carcinoma), 
      akiec (Actinic Keratosis) - these require careful clinical attention.
    
    ⚠️  DISCLAIMER: This is an AI-assisted analysis tool. All diagnoses should be confirmed by a 
        qualified dermatologist through clinical examination and histopathological analysis.
    """
    
    ax7.text(0.02, 0.95, interpretation, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Clinical explanation saved to: {output_path}")
    
    return fig


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate explanations for skin disease classification model"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image to explain"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model_finetuned.keras",
        help="Path to the trained model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/explanation.png",
        help="Output path for the explanation figure"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        default=['gradcam', 'lime'],
        choices=['gradcam', 'gradcam++', 'lime'],
        help="Explanation methods to use"
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=500,
        help="Number of samples for LIME"
    )
    parser.add_argument(
        "--clinical",
        action="store_true",
        help="Generate clinical-grade explanation report"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)
    
    # Load class mapping
    import json
    with open("models/class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    
    if args.clinical:
        # Generate clinical report
        create_clinical_explanation(
            model, args.image_path, class_names,
            output_path=args.output
        )
    else:
        # Generate standard explanation
        generator = ExplanationGenerator(model, class_names)
        result = generator.explain_image(
            args.image_path,
            methods=args.methods,
            lime_samples=args.lime_samples
        )
        
        # Save figure
        generator.create_explanation_figure(result, save_path=args.output)
    
    print(f"\nExplanation saved to: {args.output}")


if __name__ == "__main__":
    main()
