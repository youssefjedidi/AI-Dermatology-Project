# tests/test_data_preprocessing.py
import pytest
import tensorflow as tf
import numpy as np
from src import config

# This import now works because you moved the function to the global scope
from src.data_preprocessing import preprocess_image 

# The test below was removed because the `preprocess_image` function is not
# responsible for resizing images. Its only job is to normalize pixel values.
# Resizing is handled by `tf.keras.utils.image_dataset_from_directory`.
#
# def test_preprocess_image_resizes_correctly():
#     ...

def test_preprocess_image_normalizes_values_and_preserves_shape():
    """
    Tests if the preprocess_image function correctly:
    1. Normalizes pixel values to the [-1, 1] range.
    2. Does NOT change the shape of the input image.
    """
    # Arrange: Create a fake image with pixel values from 0 to 255
    # The shape should match what the function expects, as no resizing happens here.
    image_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    fake_image = tf.cast(np.random.randint(0, 256, size=image_shape), dtype=tf.float32)
    dummy_label = tf.constant(0, dtype=tf.int32)

    # Act: Run the function we are testing
    processed_image, _ = preprocess_image(fake_image, dummy_label)

    # Assert (Part 1): Check if all pixel values are within the expected range
    assert tf.reduce_min(processed_image) >= -1.0, "Minimum pixel value is less than -1.0"
    assert tf.reduce_max(processed_image) <= 1.0, "Maximum pixel value is greater than 1.0"
    
    # Assert (Part 2): Check that the shape has NOT changed
    assert processed_image.shape == image_shape, "Function should not change the image shape"