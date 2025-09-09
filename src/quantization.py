# src/quantization.py
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm # Import the tqdm library for the progress bar

from src import config
from src.model import build_model # We need to build the model to load weights into it

def quantize_model(weights_path: str, quantized_model_path: str):
    """
    Loads a model with trained weights, applies dynamic range quantization,
    and saves the resulting TFLite model.
    """
    print(f"\n--- 1. Starting Model Quantization ---")
    print(f"Loading weights from: {weights_path}")
    Path(quantized_model_path).parent.mkdir(parents=True, exist_ok=True)

    model = build_model(for_finetuning=True)
    model.load_weights(weights_path)
    print("Weights loaded successfully.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("Converting model to TFLite with dynamic range quantization...")
    tflite_quantized_model = converter.convert()
    
    with open(quantized_model_path, 'wb') as f:
        f.write(tflite_quantized_model)
        
    print(f"Quantized TFLite model saved successfully to: {quantized_model_path}")
    return tflite_quantized_model


def evaluate_tflite_model(tflite_model_path: str, test_ds: tf.data.Dataset, class_names: list):
    """
    Evaluates the performance of a TFLite model on a test dataset, calculating
    Top-1 and Top-3 accuracy.
    """
    print(f"\n--- 2. Evaluating Quantized TFLite Model ---")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # --- MODIFICATION: Prepare for both Top-1 and Top-3 ---
    all_predictions_top1 = []
    all_true_labels = []
    top3_correct_predictions = 0

    num_batches = tf.data.experimental.cardinality(test_ds).numpy()
    
    for images, labels in tqdm(test_ds, total=num_batches, desc="Evaluating Quantized Model"):
        if images.shape[0] != input_details['shape'][0]:
            interpreter.resize_tensor_input(input_details['index'], images.shape)
            interpreter.allocate_tensors()
            
        interpreter.set_tensor(input_details['index'], images)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details['index'])
        
        # --- NEW LOGIC FOR TOP-K ACCURACY ---
        
        # 1. Get Top-1 predictions (for classification report and accuracy)
        predicted_labels_top1 = np.argmax(output, axis=1)
        all_predictions_top1.extend(predicted_labels_top1)
        true_labels_batch = labels.numpy()
        all_true_labels.extend(true_labels_batch)
        
        # 2. Get Top-3 predictions and check for correctness
        # argsort sorts from smallest to largest, so we take the last 3 indices
        top3_indices = np.argsort(output, axis=1)[:, -3:]
        
        # 3. Iterate through the batch to check if the true label is in the Top-3
        for i in range(len(true_labels_batch)):
            if true_labels_batch[i] in top3_indices[i]:
                top3_correct_predictions += 1

    # --- MODIFICATION: Calculate and Print Both Accuracies ---
    total_predictions = len(all_true_labels)
    top1_accuracy = accuracy_score(all_true_labels, all_predictions_top1)
    top3_accuracy = top3_correct_predictions / total_predictions
    
    print(f"\n--- Quantized Model Evaluation Results ---")
    print(f"Quantized Model Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Quantized Model Top-3 Accuracy: {top3_accuracy:.4f}")
    
    print("\n--- Classification Report (based on Top-1 predictions) ---")
    print(classification_report(all_true_labels, all_predictions_top1, target_names=class_names))


if __name__ == '__main__':
    print("--- Running Standalone Model Quantization and Evaluation ---")
    
    from src.data_preprocessing import get_data_generators
    
    _, _, test_ds, class_names = get_data_generators()

    weights_file = config.MODELS_DIR / "finetuned_model_weights.weights.h5"
    tflite_file = config.MODELS_DIR / config.QUANTIZED_MODEL_NAME
    
    quantize_model(str(weights_file), str(tflite_file))
    evaluate_tflite_model(str(tflite_file), test_ds, class_names)
# # src/quantization.py
# import tensorflow as tf
# import numpy as np
# from pathlib import Path
# from sklearn.metrics import classification_report, accuracy_score
# from tqdm import tqdm # Import the tqdm library for the progress bar

# from src import config
# from src.model import build_model # We need to build the model to load weights into it

# def quantize_model(weights_path: str, quantized_model_path: str):
#     """
#     Loads a model with trained weights, applies dynamic range quantization,
#     and saves the resulting TFLite model.

#     Args:
#         weights_path (str): Path to the saved Keras model weights (.weights.h5).
#         quantized_model_path (str): Path to save the output .tflite model.
#     """
#     print(f"\n--- 1. Starting Model Quantization ---")
#     print(f"Loading weights from: {weights_path}")
#     Path(quantized_model_path).parent.mkdir(parents=True, exist_ok=True)

#     # Build a fresh model instance in fine-tuning mode to match the weights' architecture
#     model = build_model(for_finetuning=True)
    
#     # Load the trained weights
#     model.load_weights(weights_path)
#     print("Weights loaded successfully.")

#     # Create a TFLite converter from the Keras model
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
#     # Apply dynamic range quantization (weights are quantized to 8-bit integers)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
#     # Convert the model
#     print("Converting model to TFLite with dynamic range quantization...")
#     tflite_quantized_model = converter.convert()
    
#     # Save the quantized model to a file
#     with open(quantized_model_path, 'wb') as f:
#         f.write(tflite_quantized_model)
        
#     print(f"Quantized TFLite model saved successfully to: {quantized_model_path}")
#     return tflite_quantized_model


# def evaluate_tflite_model(tflite_model_path: str, test_ds: tf.data.Dataset, class_names: list):
#     """
#     Evaluates the performance of a TFLite model on a test dataset using batch processing.

#     Args:
#         tflite_model_path (str): Path to the .tflite model file.
#         test_ds (tf.data.Dataset): The test dataset, preprocessed and batched.
#         class_names (list): A list of class names for the classification report.
#     """
#     print(f"\n--- 2. Evaluating Quantized TFLite Model ---")
    
#     # Load the TFLite model and allocate tensors
#     interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
#     interpreter.allocate_tensors()

#     # Get input and output tensor details
#     input_details = interpreter.get_input_details()[0]
#     output_details = interpreter.get_output_details()[0]

#     all_predictions = []
#     all_true_labels = []

#     # --- PROGRESS BAR INTEGRATION ---
#     # Get the total number of batches to show a proper progress bar
#     num_batches = tf.data.experimental.cardinality(test_ds).numpy()

#     # Wrap the dataset with tqdm to create a progress bar
#     for images, labels in tqdm(test_ds, total=num_batches, desc="Evaluating Quantized Model"):
#         # Check if the input tensor needs to be resized for the batch
#         if images.shape[0] != input_details['shape'][0]:
#             interpreter.resize_tensor_input(input_details['index'], images.shape)
#             interpreter.allocate_tensors()
            
#         # Set the tensor for the entire batch
#         interpreter.set_tensor(input_details['index'], images)
#         interpreter.invoke()
        
#         # Get the output and find the predicted label for each image in the batch
#         output = interpreter.get_tensor(output_details['index'])
#         predicted_labels = np.argmax(output, axis=1)
        
#         all_predictions.extend(predicted_labels)
#         all_true_labels.extend(labels.numpy())

#     # Calculate and print the final metrics
#     accuracy = accuracy_score(all_true_labels, all_predictions)
#     print(f"\n--- Quantized Model Evaluation Results ---")
#     print(f"Quantized Model Top-1 Accuracy: {accuracy:.4f}")
    
#     print("\n--- Classification Report ---")
#     print(classification_report(all_true_labels, all_predictions, target_names=class_names))


# if __name__ == '__main__':
#     print("--- Running Standalone Model Quantization and Evaluation ---")
    
#     from src.data_preprocessing import get_data_generators
    
#     # Get the test dataset
#     _, _, test_ds, class_names = get_data_generators()

#     # Define paths using the central config file
#     weights_file = config.MODELS_DIR / "finetuned_model_weights.weights.h5"
#     tflite_file = config.MODELS_DIR / config.QUANTIZED_MODEL_NAME
    
#     # Step 1: Quantize the Keras model and save it as a .tflite file
#     quantize_model(str(weights_file), str(tflite_file))
    
#     # Step 2: Evaluate the newly created .tflite model
#     evaluate_tflite_model(str(tflite_file), test_ds, class_names)