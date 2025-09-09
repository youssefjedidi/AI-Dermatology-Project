# tests/test_evaluate.py
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from unittest.mock import MagicMock

# Import the functions we want to test
from src.evaluate import plot_and_save_confusion_matrix, evaluate_model

def test_plot_and_save_confusion_matrix(tmp_path):
    """
    Tests that the confusion matrix function creates an output file.
    We don't check the plot's content, just that it runs and saves.
    """
    # Arrange
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    class_names = ['Class A', 'Class B']
    save_path = tmp_path / "cm.png"

    # Act
    # We run the actual function because it doesn't have heavy dependencies.
    plot_and_save_confusion_matrix(y_true, y_pred, class_names, str(save_path))

    # Assert
    # The only thing we need to verify is that the file was created.
    assert save_path.exists()


@pytest.fixture
def mock_model_pipeline(monkeypatch):
    """
    A fixture to completely mock the model building, loading, and evaluation process.
    """
    # 1. Create a fake model object
    mock_model = MagicMock()
    # 2. Configure its .evaluate() method to return some dummy scores
    mock_model.evaluate.return_value = [0.12345, 0.98765, 0.99987] # [loss, top-1, top-3]
    
    # 3. Create a mock for the build_model function that returns our fake model
    mock_build_model = MagicMock(return_value=mock_model)
    
    # 4. Use monkeypatch to replace the real build_model in the evaluate module
    monkeypatch.setattr("src.evaluate.build_model", mock_build_model)
    
    # Return the individual mocks so our test can make assertions on them
    return mock_build_model, mock_model

def test_evaluate_model_happy_path(mock_model_pipeline, capsys):
    """
    Tests the main evaluation logic, ensuring all mocked components are called correctly.
    'capsys' is a pytest fixture to capture print statements (stdout).
    """
    # Arrange
    mock_build_model, mock_model = mock_model_pipeline
    fake_weights_path = "fake/path/model.weights.h5"
    # Create a dummy dataset (the content doesn't matter, only the object itself)
    fake_test_ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    fake_class_names = ['A', 'B', 'C']

    # Act
    evaluate_model(fake_weights_path, fake_test_ds, fake_class_names)

    # Assert
    # 1. Was the model built correctly for fine-tuning?
    mock_build_model.assert_called_once_with(for_finetuning=True)
    # 2. Were the weights loaded?
    mock_model.load_weights.assert_called_once_with(fake_weights_path)
    # 3. Was the model evaluated on the test set?
    mock_model.evaluate.assert_called_once_with(fake_test_ds, verbose=1)

    # 4. Check if the results were printed correctly to the console
    captured = capsys.readouterr()
    assert "Test Loss: 0.1235" in captured.out
    assert "Test Accuracy (Top-1): 0.9877" in captured.out
    assert "Test Accuracy (Top-3): 0.9999" in captured.out

def test_evaluate_model_handles_load_error(mock_model_pipeline, capsys):
    """
    Tests that the function handles and reports an error if loading weights fails.
    """
    # Arrange
    mock_build_model, mock_model = mock_model_pipeline
    # Configure the load_weights method to raise an error when called
    mock_model.load_weights.side_effect = Exception("File not found!")

    # Act
    evaluate_model("bad/path.h5", None, None)

    # Assert
    # Check that the error message was printed to the console
    captured = capsys.readouterr()
    assert "FATAL: Error loading model weights." in captured.out
    # Check that model.evaluate was NOT called, because the function should exit early.
    mock_model.evaluate.assert_not_called()