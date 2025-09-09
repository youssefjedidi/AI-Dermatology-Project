# tests/test_train.py
import pytest
from unittest.mock import MagicMock, call
from pathlib import Path

# Import the module we are testing
from src import train
from src import config

@pytest.fixture
def mock_training_dependencies(monkeypatch):
    """
    Mocks all external Keras dependencies by directly patching the attributes
    on the train module object.
    """
    mock_ModelCheckpoint = MagicMock()
    mock_EarlyStopping = MagicMock()
    mock_ReduceLROnPlateau = MagicMock()
    mock_CSVLogger = MagicMock()

    monkeypatch.setattr(train, "ModelCheckpoint", mock_ModelCheckpoint)
    monkeypatch.setattr(train, "EarlyStopping", mock_EarlyStopping)
    monkeypatch.setattr(train, "ReduceLROnPlateau", mock_ReduceLROnPlateau)
    monkeypatch.setattr(train, "CSVLogger", mock_CSVLogger)
    monkeypatch.setattr(Path, "mkdir", MagicMock())

    return {
        "ModelCheckpoint": mock_ModelCheckpoint,
        "EarlyStopping": mock_EarlyStopping,
        "ReduceLROnPlateau": mock_ReduceLROnPlateau,
        "CSVLogger": mock_CSVLogger,
    }

def test_train_model_orchestration(mock_training_dependencies, capsys):
    """
    Tests that train_model correctly configures callbacks and calls model.fit().
    """
    # --- Arrange ---
    callbacks = mock_training_dependencies
    mock_model = MagicMock()
    mock_history = MagicMock()
    mock_model.fit.return_value = mock_history
    
    model_name = config.INITIAL_MODEL_NAME
    epochs = 15
    
    expected_weights_filename = model_name.replace('.keras', '.weights.h5')
    expected_weights_filepath = config.MODELS_DIR / expected_weights_filename
    expected_log_path = config.LOGS_DIR / f"{model_name.replace('.keras', '')}_log.csv"

    # --- Act ---
    returned_model, returned_history = train.train_model(
        model=mock_model,
        train_ds="fake_train_data",
        val_ds="fake_val_data",
        epochs=epochs,
        model_name=model_name
    )

    # --- Assert ---

    # 1. Assert ALL callbacks were instantiated with the exact, correct parameters.
    callbacks["ModelCheckpoint"].assert_called_once_with(
        filepath=expected_weights_filepath,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    callbacks["EarlyStopping"].assert_called_once_with(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
    callbacks["ReduceLROnPlateau"].assert_called_once_with(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=0.00001 # This must match train.py exactly
    )
    callbacks["CSVLogger"].assert_called_once_with(expected_log_path)

    # 2. Assert model.fit was called
    mock_model.fit.assert_called_once()
    
    # 3. Assert the best weights were loaded at the end
    mock_model.load_weights.assert_called_once_with(expected_weights_filepath)

    # 4. Assert that the function returned the correct objects
    assert returned_model is mock_model
    assert returned_history is mock_history