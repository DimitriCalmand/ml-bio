import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


def create_model(num_classes=7, input_shape=(224, 224, 3), trainable_layers=20):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    # Remplacer preprocess_input par Rescaling pour éviter les erreurs de sérialisation (TrueDivide)
    # MobileNetV2 attend des pixels entre -1 et 1
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def get_callbacks(model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        
        keras.callbacks.CSVLogger(
            str(model_dir / "training_log.csv")
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    print("Création du modèle de test...")
    model = create_model()
    model = compile_model(model)
    
    print("\nArchitecture du modèle:")
    model.summary()
    
    print(f"\nNombre total de paramètres: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Paramètres entraînables: {trainable_params:,}")


def unfreeze_base_model(model, num_layers_unfreeze=50):
    """
    Décongèle les dernières couches du modèle de base pour le fine-tuning.
    """
    # Trouver la couche MobileNetV2
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or "mobilenet" in layer.name:
            base_model = layer
            break
            
    if base_model is None:
        print("Attention: Modèle de base non trouvé pour le fine-tuning")
        return model

    # Décongeler le modèle de base
    base_model.trainable = True
    
    # Congeler toutes les couches sauf les N dernières
    # Note: Il est important de laisser les couches BatchNormalization gelées
    # lors du fine-tuning pour ne pas casser les statistiques apprises
    for layer in base_model.layers[:-num_layers_unfreeze]:
        layer.trainable = False
        
    print(f"\nModèle décongelé. Les {num_layers_unfreeze} dernières couches du backbone sont entraînables.")
    return model
