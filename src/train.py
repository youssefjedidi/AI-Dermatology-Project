# src/train.py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from src import config # Import our central configuration file

def train_model(model: tf.keras.Model, train_ds, val_ds, epochs: int, model_name: str, is_finetuning: bool = False):
    """
    Trains the Keras model with specified callbacks.

    Args:
        model (tf.keras.Model): The compiled Keras model to train.
        train_ds: The training dataset.
        val_ds: The validation dataset.
        epochs (int): The number of epochs for training.
        model_name (str): The filename for the saved model.
        is_finetuning (bool): Flag to adjust callback monitoring for fine-tuning.

    Returns:
        The trained Keras model and its training history.
    """
    # Create the directory for logs if it doesn't exist
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOGS_DIR / f"{model_name.replace('.keras', '')}_log.csv"

    # Define callbacks for model training
    # We use ModelCheckpoint to save the best version of the model during training
    model_checkpoint = ModelCheckpoint(
        filepath=config.MODELS_DIR / model_name,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=6, # Stop if val_loss doesn't improve for 6 epochs
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Reduce LR by a factor of 5
        patience=3,
        verbose=1,
        min_lr=0.00001
    )

    csv_logger = CSVLogger(log_path)
    
    callbacks = [model_checkpoint, early_stopping, reduce_lr, csv_logger]

    stage = "Fine-Tuning" if is_finetuning else "Initial Training"
    print(f"\n--- Starting Model {stage} for {epochs} epochs ---")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    print(f"\n--- {stage} Finished ---")
    
    # After training, we explicitly load the best model saved by ModelCheckpoint
    print(f"Loading best model from {config.MODELS_DIR / model_name}")
    best_model = tf.keras.models.load_model(config.MODELS_DIR / model_name)
    
    return best_model, history

if __name__ == '__main__':
    # This block is for testing purposes. It will not run a full training session.
    print("--- Testing Train Module ---")
    # To run a test, we would need to build a model and load data first.
    # For now, we confirm the file can be imported and functions are defined.
    # A real test would involve creating dummy data and a dummy model.
    print("Train module is syntactically correct.")