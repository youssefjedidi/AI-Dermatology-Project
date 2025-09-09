# src/train.py
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from src import config

def train_model(model: tf.keras.Model, train_ds, val_ds, epochs: int, model_name: str):
    """
    Trains the Keras model, saving only the best weights. This is the robust method.
    """
    # Ensure directories for logs and models exist
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define filenames for weights and logs based on the base model_name
    weights_filename = model_name.replace('.keras', '.weights.h5')
    weights_filepath = config.MODELS_DIR / weights_filename
    log_path = config.LOGS_DIR / f"{model_name.replace('.keras', '')}_log.csv"

    # --- CRITICAL CHANGE: Save only the weights of the best model ---
    model_checkpoint = ModelCheckpoint(
        filepath=weights_filepath,
        save_weights_only=True, # This is the key
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=0.00001)
    csv_logger = CSVLogger(log_path)
    
    callbacks = [model_checkpoint, early_stopping, reduce_lr, csv_logger]

    stage = "Fine-Tuning" if "finetuned" in model_name else "Initial Training"
    print(f"\n--- Starting Model {stage} for {epochs} epochs ---")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    print(f"\n--- {stage} Finished ---")
    
    # Load the best weights back into the model to ensure it's in the best state
    print(f"Loading best weights from {weights_filepath} into the final model object.")
    model.load_weights(weights_filepath)
    
    return model, history