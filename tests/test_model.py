# tests/test_model.py
import pytest
import tensorflow as tf
import numpy as np  
from src import config
from src.model import build_model

def test_build_model_creates_keras_model():
    """
    Smoke test to ensure the build_model function returns a valid Keras model.
    """
    # Act
    model = build_model(for_finetuning=False)
    
    # Assert
    assert isinstance(model, tf.keras.Model)

def test_build_model_output_shape_is_correct():
    """
    Tests if the model's final output layer has the correct shape for the number of classes.
    """
    # Act
    model = build_model(for_finetuning=False)
    
    # Assert
    # The output shape will be (None, NUM_CLASSES). 'None' is for the batch size.
    expected_shape = (None, config.NUM_CLASSES)
    assert model.output_shape == expected_shape

def test_build_model_uses_correct_learning_rate():
    """
    Tests if the model is compiled with the correct learning rate based on the flag.
    """
    # Act
    initial_model = build_model(for_finetuning=False)
    finetune_model = build_model(for_finetuning=True)

    # Assert
    assert np.isclose(initial_model.optimizer.learning_rate.numpy(), config.INITIAL_LEARNING_RATE)
    assert np.isclose(finetune_model.optimizer.learning_rate.numpy(), config.FINETUNE_LEARNING_RATE)