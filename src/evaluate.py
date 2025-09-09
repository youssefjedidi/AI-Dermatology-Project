# src/evaluate.py
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
# We need our local model builder to create a fresh architecture
from src.model import build_model

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Calculates, plots, and saves a confusion matrix."""
    # Ensure the parent directory for the log/plot exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
    con_mat_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(15, 12))
    sns.heatmap(con_mat_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def evaluate_model(weights_path: str, test_ds, class_names: list):
    """
    Builds a fresh model, loads saved weights from a .weights.h5 file, 
    and evaluates its performance.
    """
    print(f"\n--- Evaluating Model Using Weights: {weights_path} ---")

    # 1. Build a fresh, clean instance of the model architecture.
    # The saved weights are from a fine-tuned model, so we build it in that mode.
    model = build_model(for_finetuning=True)

    # 2. Load only the learned weights into this clean architecture.
    try:
        model.load_weights(weights_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading model weights.")
        import traceback
        traceback.print_exc()
        return

    # The model is already compiled by the build_model function.

    # Perform standard evaluation (loss, top-1, top-3)
    print("\nCalculating loss and accuracy metrics...")
    scores = model.evaluate(test_ds, verbose=1)
    print(f"\n--- EVALUATION RESULTS ---")
    print(f"Test Loss: {scores[0]:.4f}")
    print(f"Test Accuracy (Top-1): {scores[1]:.4f}")
    print(f"Test Accuracy (Top-3): {scores[2]:.4f}")

    # --- Generate Detailed Classification Report ---
    # print("\n--- Generating Classification Report ---")
    # y_true = np.concatenate([y for x, y in test_ds], axis=0)
    # y_pred_probs = model.predict(test_ds)
    # y_pred = np.argmax(y_pred_probs, axis=1)
    
    # print(classification_report(y_true, y_pred, target_names=class_names))
    
    # # --- Generate and Save Confusion Matrix ---
    # matrix_path = config.LOGS_DIR / f"{Path(weights_path).stem}_confusion_matrix.png"
    # plot_and_save_confusion_matrix(y_true, y_pred, class_names, matrix_path)

if __name__ == '__main__':
    print("--- Running Standalone Model Evaluation ---")
    
    # This import works because our project is installed in editable mode.
    from src.data_preprocessing import get_data_generators
    
    print("Preparing test data generator...")
    _, _, test_ds, class_names = get_data_generators()
    
    # Define the path to the weights file that you downloaded from Kaggle.
    model_weights_path = config.MODELS_DIR / "finetuned_model_weights.weights.h5"
    
    if model_weights_path.exists():
        print(f"Weights file found at: {model_weights_path}")
        evaluate_model(str(model_weights_path), test_ds, class_names)
    else:
        print(f"FATAL ERROR: Model weights not found at {model_weights_path}")
        print("Please ensure the finetuned_model_weights.weights.h5 file is in the models/ directory.")