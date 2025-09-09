# tests/test_quantization.py
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock, mock_open, patch

from src.quantization import quantize_model, evaluate_tflite_model

@pytest.fixture
def mock_quantization_pipeline(monkeypatch):
    """Mocks the Keras and TFLite conversion pipeline."""
    mock_model = MagicMock()
    mock_build_model = MagicMock(return_value=mock_model)
    monkeypatch.setattr("src.quantization.build_model", mock_build_model)
    
    mock_converter_instance = MagicMock()
    mock_converter_instance.convert.return_value = b"quantized_model_bytes"
    mock_converter_class = MagicMock()
    mock_converter_class.from_keras_model.return_value = mock_converter_instance
    monkeypatch.setattr("src.quantization.tf.lite.TFLiteConverter", mock_converter_class)
    
    return mock_build_model, mock_model, mock_converter_class

def test_quantize_model(mock_quantization_pipeline, tmp_path):
    """Tests the quantization function's orchestration."""
    mock_build_model, mock_model, mock_converter_class = mock_quantization_pipeline
    weights_path = "fake/weights.h5"
    quantized_path = tmp_path / "model.tflite"
    
    with patch("builtins.open", mock_open()) as mocked_file:
        result = quantize_model(weights_path, str(quantized_path))

        mock_build_model.assert_called_once_with(for_finetuning=True)
        mock_model.load_weights.assert_called_once_with(weights_path)
        mock_converter_class.from_keras_model.assert_called_once_with(mock_model)
        mocked_file.assert_called_once_with(str(quantized_path), 'wb')
        mocked_file().write.assert_called_once_with(b"quantized_model_bytes")
        assert result == b"quantized_model_bytes"

@pytest.fixture
def mock_tflite_interpreter(monkeypatch):
    """Mocks the TFLite interpreter to return predictable values for a 3-class model."""
    mock_interpreter_instance = MagicMock()
    
    input_details = [{'index': 0, 'shape': np.array([3, 299, 299, 3])}] # Batch size of 3
    output_details = [{'index': 1}]
    mock_interpreter_instance.get_input_details.return_value = input_details
    mock_interpreter_instance.get_output_details.return_value = output_details
    
    # --- THE FIX IS HERE ---
    # Mock predictions for 3 samples with 3 classes each.
    mock_predictions = np.array([
        [0.1, 0.8, 0.1],  # Prediction: 1, True: 1. Top-1 Correct.
        [0.7, 0.2, 0.1],  # Prediction: 0, True: 0. Top-1 Correct.
        [0.1, 0.2, 0.7],  # Prediction: 2, True: 2. Top-1 Correct.
    ], dtype=np.float32)
    mock_interpreter_instance.get_tensor.return_value = mock_predictions
    
    mock_interpreter_class = MagicMock(return_value=mock_interpreter_instance)
    monkeypatch.setattr("src.quantization.tf.lite.Interpreter", mock_interpreter_class)
    
    return mock_interpreter_instance

def test_evaluate_tflite_model(mock_tflite_interpreter, capsys):
    """
    Tests that the TFLite evaluation function correctly calculates
    Top-1 and Top-3 accuracy from mock interpreter output.
    """
    # Arrange
    # --- AND THE FIX IS HERE ---
    # True labels now match the 3-class scenario.
    true_labels = np.array([1, 0, 2])
    # Dummy dataset with 3 samples.
    fake_images = np.zeros((3, 299, 299, 3), dtype=np.float32)
    test_ds = tf.data.Dataset.from_tensor_slices((fake_images, true_labels)).batch(3)
    
    class_names = ['Class A', 'Class B', 'Class C']
    
    # Act
    evaluate_tflite_model("fake/model.tflite", test_ds, class_names)
    
    # Assert
    # All 3 predictions are correct for Top-1 and Top-3.
    # Expected Top-1 Accuracy: 3/3 = 1.0
    # Expected Top-3 Accuracy: 3/3 = 1.0
    
    captured = capsys.readouterr()
    assert "Quantized Model Top-1 Accuracy: 1.0000" in captured.out
    assert "Quantized Model Top-3 Accuracy: 1.0000" in captured.out
    # Also, check that the classification report is now printed without error.
    assert "Class A" in captured.out
    assert "Class C" in captured.out