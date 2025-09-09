# main.py
import tensorflow as tf
from src import config
from src.data_ingestion import download_and_extract_data
from src.data_preprocessing import get_data_generators
from src.model import build_model
from src.train import train_model
# We will create the evaluate.py and quantization.py logic later
# from src.evaluate import evaluate_model
# from src.quantization import convert_to_tflite

def run_pipeline():
    """
    Orchestrates the entire ML pipeline from data ingestion to model training.
    """
    print("--- Starting Full ML Pipeline ---")

    # --- Phase 1: Data Ingestion ---
    # Ensure the raw data is downloaded and extracted.
    download_and_extract_data(
        dataset_id=config.KAGGLE_DATASET_ID,
        download_path=config.KAGGLE_DOWNLOAD_PATH,
        extract_path=config.EXTRACTED_DATA_PATH
    )

    # --- Phase 2: Data Preprocessing ---
    # Create the data generators for training, validation, and testing.
    train_ds, val_ds, test_ds, class_names = get_data_generators()

    # --- Phase 3: Initial Model Training (Transfer Learning) ---
    # Build the model with the frozen base for initial training.
    initial_model = build_model(for_finetuning=False)
    
    # Train the initial model
    trained_initial_model, history_initial = train_model(
        model=initial_model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config.INITIAL_TRAINING_EPOCHS,
        model_name=config.INITIAL_MODEL_NAME
    )
    
    # At this point, you would typically evaluate the initial model
    # print("\n--- Evaluating Initial Model ---")
    # evaluate_model(trained_initial_model, test_ds)

    # --- Phase 4: Model Fine-Tuning ---
    # Re-compile the model with unfrozen layers and a lower learning rate.
    # We use the 'trained_initial_model' as the starting point.
    finetune_model = build_model(for_finetuning=True)
    # The weights of the classification head are preserved. We need to load the full model weights.
    finetune_model.load_weights(config.MODELS_DIR / config.INITIAL_MODEL_NAME)
    
    # Train the fine-tuning model
    trained_finetune_model, history_finetune = train_model(
        model=finetune_model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config.FINETUNE_EPOCHS,
        model_name=config.FINETUNED_MODEL_NAME,
        is_finetuning=True
    )

    # --- Phase 5: Final Evaluation & Quantization (Future Steps) ---
    # print("\n--- Evaluating Final Fine-Tuned Model ---")
    # evaluate_model(trained_finetune_model, test_ds)
    
    # print("\n--- Converting Final Model to TFLite ---")
    # convert_to_tflite(trained_finetune_model)

    print("\n--- ML Pipeline Finished Successfully ---")
    print(f"Final models saved in: {config.MODELS_DIR}")

if __name__ == '__main__':
    # This makes the script executable.
    run_pipeline()