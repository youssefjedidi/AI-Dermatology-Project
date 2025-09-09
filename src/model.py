# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from src import config # Import our central configuration file

def build_model(for_finetuning: bool = False):
    """
    Builds, compiles, and returns the Xception-based Keras model.

    Args:
        for_finetuning (bool): If True, prepares the model for fine-tuning by
        making more layers trainable.

    Returns:
        A compiled Keras model.
    """
    print(f"--- Building Xception model (Fine-tuning mode: {for_finetuning}) ---")

    # 1. Load the pre-trained Xception base
    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False, # Exclude the final classification layer
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    )

    # 2. Set the trainability of the base model layers
    if for_finetuning:
        # Unfreeze the last 35 layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-35]:
            layer.trainable = False
        print(f"Fine-tuning enabled: Last {len(base_model.layers[-35:])} layers of Xception are now trainable.")
    else:
        # Freeze the entire base model for initial feature extraction
        base_model.trainable = False
        print("Initial training: All layers of the Xception base model are frozen.")

    # 3. Create the new classification head
    # We use the Keras Functional API for more flexibility
    inputs = tf.keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    x = base_model(inputs, training=False if not for_finetuning else True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # 4. Compile the model with an appropriate learning rate
    learning_rate = config.FINETUNE_LEARNING_RATE if for_finetuning else config.INITIAL_LEARNING_RATE
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            'sparse_categorical_accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    print(f"Model compiled with learning rate: {learning_rate}")
    model.summary()
    return model

if __name__ == '__main__':
    # This block allows you to test the model building script directly
    print("--- Testing Initial Model Build ---")
    initial_model = build_model(for_finetuning=False)

    print("\n" + "="*50 + "\n")

    print("--- Testing Fine-tuning Model Build ---")
    finetune_model = build_model(for_finetuning=True)