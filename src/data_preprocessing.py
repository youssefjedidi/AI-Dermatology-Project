# src/data_preprocessing.py
import tensorflow as tf
from src import config # Import our central configuration file

def preprocess_image(image, label):
    """Applies the Xception-specific normalization to an image."""
    return tf.keras.applications.xception.preprocess_input(image), label

def get_data_generators():
    """
    Creates and returns the training, validation, and test data generators.

    This function uses the parameters from the config file to load images,
    create a validation split, apply data augmentation, and optimize the
    data pipeline for performance.

    Returns:
        A tuple containing (train_ds, val_ds, test_ds, class_names).
    """
    print("--- Creating Data Generators ---")

    # The modern way to create a validation split is directly in the loader
    # This ensures no data leakage between training and validation sets.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.EXTRACTED_DATA_PATH / "train",
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=123, # Use a seed for reproducibility
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode='int' # Best for SparseCategoricalCrossentropy loss
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.EXTRACTED_DATA_PATH / "train",
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode='int'
    )

    # The test dataset does not need a validation split
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.EXTRACTED_DATA_PATH / "test",
        shuffle=False, # It's critical not to shuffle the test set for consistent evaluation
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        interpolation='bilinear', # Explicitly set to match standard practice.
        label_mode='int'
    )

    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")

    # --- Define Augmentation and Preprocessing Layers ---

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ], name="data_augmentation")


    # --- Apply Transformations to the Datasets ---

    # Augment and preprocess the training data
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Only preprocess validation and test data (DO NOT augment them)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # --- Optimize for Performance ---
    # Cache data in memory and prefetch batches to overlap data prep and model execution
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print("--- Data Generators Created and Optimized ---")

    return train_ds, val_ds, test_ds, class_names

if __name__ == '__main__':
    # This block allows you to run this script directly to test the data loaders
    # It will create the generators and print the shape of one batch.
    print("Testing the data generator script...")
    train_generator, val_generator, test_generator, classes = get_data_generators()
    
    for image_batch, label_batch in train_generator.take(1):
        print("Shape of one batch of training images:", image_batch.shape)
        print("Shape of one batch of training labels:", label_batch.shape)