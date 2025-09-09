# main.py
import tensorflow as tf
from src import config
from src.data_ingestion import download_and_extract_data
from src.data_preprocessing import get_data_generators
from src.model import build_model
from src.train import train_model

def run_pipeline():
    """
    Orchestrates the entire ML pipeline from data ingestion to model training.
    """
    print("--- Starting Full ML Pipeline ---")

    # Phase 1: Data Ingestion
    download_and_extract_data(
        dataset_id=config.KAGGLE_DATASET_ID,
        download_path=config.KAGGLE_DOWNLOAD_PATH,
        extract_path=config.EXTRACTED_DATA_PATH
    )

    # Phase 2: Data Preprocessing
    train_ds, val_ds, test_ds, class_names = get_data_generators()

    # Phase 3: Initial Model Training (Transfer Learning)
    print("\n" + "="*50 + "\nPHASE 3: INITIAL TRAINING\n" + "="*50)
    initial_model = build_model(for_finetuning=False)
    
    train_model(
        model=initial_model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config.INITIAL_TRAINING_EPOCHS,
        model_name=config.INITIAL_MODEL_NAME
    )
    
    # Phase 4: Model Fine-Tuning
    print("\n" + "="*50 + "\nPHASE 4: FINE-TUNING\n" + "="*50)
    finetune_model = build_model(for_finetuning=True)
    
    # Load the weights from the best checkpoint of the initial training phase.
    # This is a critical step.
    initial_model_path = config.MODELS_DIR / config.INITIAL_MODEL_NAME
    print(f"Loading weights from {initial_model_path} for fine-tuning...")
    finetune_model.load_weights(initial_model_path)
    
    train_model(
        model=finetune_model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config.FINETUNE_EPOCHS,
        model_name=config.FINETUNED_MODEL_NAME # <-- THE FIX IS HERE (removed is_finetuning)
    )

    print("\n--- ML Pipeline Finished Successfully ---")
    print(f"Final fine-tuned model saved in: {config.MODELS_DIR}")

if __name__ == '__main__':
    run_pipeline()