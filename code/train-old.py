## Geo Image Classification - Model training
# Note: A system with GPU is preferred for DL model training


# Import required libraries
print("Importing required libraries...")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import (
    Xception,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Configuration parameters
input_size = 299  # Xception's default input size
learning_rate = 0.0005
size_inner = 100
droprate = 0.2
epochs = 10
batch_size = 32


def make_model(input_size=299, learning_rate=0.01, size_inner=100,
               droprate=0.5, num_classes=6):
    """
    Create and compile the model.
    
    Args:
        input_size (int): Input image size
        learning_rate (float): Learning rate for optimizer
        size_inner (int): Number of neurons in dense layer
        droprate (float): Dropout rate
        num_classes (int): Number of output classes
    """
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    # Freeze the base model
    base_model.trainable = False

    # Build the model
    inputs = keras.Input(shape=(input_size, input_size, 3))
    x = inputs
    
    # Add preprocessing layer
    x = tf.keras.layers.Rescaling(1./255)(x)
    
    # Base model
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Add regularization
    x = keras.layers.Dense(size_inner, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(droprate)(x)
    
    outputs = keras.layers.Dense(num_classes)(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def create_data_generators(input_size, batch_size):
    """Create training and validation data generators."""
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2  # Add validation split
    )

    # Training dataset
    train_ds = train_gen.flow_from_directory(
        '../data/seg_train/seg_train',
        target_size=(input_size, input_size),
        batch_size=batch_size,
        subset='training'  # Specify training subset
    )

    # Validation dataset
    val_ds = train_gen.flow_from_directory(
        '../data/seg_train/seg_train',
        target_size=(input_size, input_size),
        batch_size=batch_size,
        subset='validation',  # Specify validation subset
        shuffle=False
    )

    return train_ds, val_ds


def main():
    # Create data generators
    train_ds, val_ds = create_data_generators(input_size, batch_size)
    
    # Get number of classes
    num_classes = len(train_ds.class_indices)
    
    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'geo-model.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=2,
        #     min_lr=1e-6
        # )
    ]

    # Create and train model
    model = make_model(
        input_size=input_size,
        learning_rate=learning_rate,
        size_inner=size_inner,
        droprate=droprate,
        num_classes=num_classes
    )

    print(f"Starting model training with {epochs} epochs. This may take some time...")
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    print("Model training completed.")
    return history


if __name__ == "__main__":
    main()
