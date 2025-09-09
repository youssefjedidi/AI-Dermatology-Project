# src/config.py
from pathlib import Path

# --- Core Paths ---
# Use Path() for OS-agnostic paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs" # Let's create a dedicated logs directory

# --- Kaggle Dataset Information ---
# This is the unique identifier for the dataset on Kaggle
KAGGLE_DATASET_ID = "shubhamgoel27/dermnet"
KAGGLE_DOWNLOAD_PATH = DATA_DIR / "dermnet-download"
EXTRACTED_DATA_PATH = DATA_DIR / "dermnet" # Where the 'train' and 'test' folders will be

# --- Data Preprocessing & Model Parameters ---
IMAGE_SIZE = 299
IMG_WIDTH = 299
IMG_HEIGHT = 299
BATCH_SIZE = 32
NUM_CLASSES = 23
VALIDATION_SPLIT = 0.2 # 20% of the training data will be used for validation

# --- Training Parameters ---
INITIAL_TRAINING_EPOCHS = 30
FINETUNE_EPOCHS = 20
INITIAL_LEARNING_RATE = 0.001
FINETUNE_LEARNING_RATE = 0.00005

# --- Model Naming ---
INITIAL_MODEL_NAME = f"initial_model_epochs_{INITIAL_TRAINING_EPOCHS}.keras"
# FINETUNED_MODEL_NAME = f"finetuned_model_epochs_{FINETUNE_EPOCHS}.keras"
FINETUNED_MODEL_NAME = "h62_Xc_model.h5" 
QUANTIZED_MODEL_NAME = "quantized_dynamic_range_model.tflite"