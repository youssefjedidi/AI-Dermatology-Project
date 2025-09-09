# tests/test_data_preprocessing_logic.py
import pytest
import tensorflow as tf
from unittest.mock import MagicMock

# We import the module itself to patch objects within it
from src import data_preprocessing

def test_get_data_generators_applies_augmentation_correctly(monkeypatch):
    """
    Tests that data augmentation is applied ONLY to the training dataset by
    spying on the .map() method's call order and arguments.
    """
    # Arrange:
    # 1. Create a single mock object that will represent our dataset.
    mock_dataset = MagicMock()

    # 2. To allow chaining (e.g., ds.map(...).cache(...)), we configure
    #    each method to return the mock object itself.
    mock_dataset.map.return_value = mock_dataset
    mock_dataset.cache.return_value = mock_dataset
    mock_dataset.prefetch.return_value = mock_dataset

    # 3. Add the class_names attribute needed for the function to run.
    mock_dataset.class_names = ['class_a', 'class_b']

    # 4. Mock the data loader to return this single, chainable spy object every time.
    monkeypatch.setattr(tf.keras.utils, "image_dataset_from_directory", lambda *args, **kwargs: mock_dataset)

    # Act:
    # Run the function. All method calls will be recorded by our spy object.
    data_preprocessing.get_data_generators()

    # Assert:
    # Get a direct reference to the map method's mock.
    mock_map_method = mock_dataset.map

    # 1. Check that .map() was called 4 times in total.
    assert mock_map_method.call_count == 4, f"Expected .map() to be called 4 times, but was called {mock_map_method.call_count} times."

    # 2. Get the function that was passed to .map() in the VERY FIRST call.
    first_call_function = mock_map_method.call_args_list[0].args[0]
    
    # 3. The first call should be for augmentation, so its function must NOT be the named preprocess_image function.
    assert first_call_function is not data_preprocessing.preprocess_image, \
        "The first .map() call should have been the augmentation lambda, but it was preprocess_image."

    # 4. Check that the subsequent three calls WERE for preprocessing.
    for i in range(1, 4):
        subsequent_call_function = mock_map_method.call_args_list[i].args[0]
        assert subsequent_call_function is data_preprocessing.preprocess_image, \
            f"Call #{i+1} to .map() should have used preprocess_image, but it did not."