import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Try to import Focal Loss (Keras 3+) or implement it
try:
    from tensorflow.keras.losses import CategoricalFocalCrossentropy
except ImportError:
    # Fallback for older Keras versions
    class CategoricalFocalCrossentropy(keras.losses.Loss):
        def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.gamma = gamma
            self.from_logits = from_logits

        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            if self.from_logits:
                y_pred = tf.nn.softmax(y_pred)
            
            # Clip probabilities to prevent log(0)
            epsilon = keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            # Cross entropy
            cross_entropy = -y_true * tf.math.log(y_pred)
            
            # Focal loss factor
            weight = self.alpha * y_true * tf.math.pow((1 - y_pred), self.gamma)
            
            return tf.reduce_sum(weight * cross_entropy, axis=-1)

def create_model(num_classes=7, input_shape=(224, 224, 3), trainable_layers=30):
    """
    Creates a robust skin disease classification model using EfficientNetB1.
    Includes built-in data augmentation layers.
    """
    # 1. Input Layer
    inputs = keras.Input(shape=input_shape)
    
    # 2. Data Augmentation (Active only during training)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.3)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
    x = layers.RandomContrast(0.2)(x)
    x = layers.RandomBrightness(0.2)(x)
    # Add Gaussian Noise for robustness
    x = layers.GaussianNoise(0.1)(x)
    
    # 3. Backbone: EfficientNetB1
    base_model = keras.applications.EfficientNetB1(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )
    
    # Freeze the backbone initially
    base_model.trainable = False
    
    # 4. Classification Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x) # Increased dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB1_Skin")
    
    return model


def compile_model(model, learning_rate=0.001, weight_decay=0.01):
    # AdamW with stronger weight decay
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Use Focal Loss to handle imbalance dynamically
    loss = CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def get_callbacks(model_dir, append=False):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce LR when needed
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Logging
        keras.callbacks.CSVLogger(
            str(model_dir / "training_log.csv"),
            append=append
        )
    ]
    
    return callbacks


def unfreeze_base_model(model, num_layers_unfreeze=50):
    """
    Unfreezes the top N layers of the backbone for fine-tuning.
    """
    # Find the backbone (EfficientNet)
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or "efficientnet" in layer.name.lower():
            backbone = layer
            break
            
    if backbone is None:
        print("Warning: Backbone not found for unfreezing.")
        return model

    backbone.trainable = True
    
    # Freeze all layers except the last N
    for layer in backbone.layers[:-num_layers_unfreeze]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    print(f"\nModel unfreezed. Last {num_layers_unfreeze} layers of backbone are trainable.")
    return model
