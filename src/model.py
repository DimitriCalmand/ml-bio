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
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
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
            filepath=str(model_dir / "best_model.h5"),
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
            patience=7,
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
