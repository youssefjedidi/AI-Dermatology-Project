import tensorflow as tf
from tensorflow.keras import layers, regularizers
from src import config
from tensorflow.keras.applications import xception

def build_model(for_finetuning: bool = True):
    """
    Builds the Keras model, EXACTLY replicating the FINAL, FINE-TUNED architecture
    from the original Kaggle notebooks to ensure perfect weight loading.
    """

    print(f"--- Building Xception model (Fine-tuning mode: {for_finetuning}) ---")

    base_model = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    )

    # --- ARCHITECTURE MATCHING LOGIC ---
    # This logic exactly replicates the state of the model when the weights were saved.
    # The base model itself is frozen, but we can unfreeze layers within it.
    base_model.trainable = False
    set_trainable = False
    for layer in base_model.layers:
        if "block11_sepconv1" in layer.name:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    inputs = tf.keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    x = tf.keras.layers.BatchNormalization()(inputs)
    # The training=False argument is crucial here for inference
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    learning_rate = config.FINETUNE_LEARNING_RATE if for_finetuning else config.INITIAL_LEARNING_RATE
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            'sparse_categorical_accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )

    print(f"Model compiled with learning rate: {learning_rate}")
    return model
# # src/model.py
# import tensorflow as tf
# from tensorflow.keras import layers, regularizers
# from src import config
# from tensorflow.keras.applications import xception


# # def build_model(for_finetuning: bool = False):
# #     """
# #     Builds, compiles, and returns the Xception-based Keras model.
# #     This version precisely matches the trainable layer logic needed for the saved weights.
# #     """
# #     print(f"--- Building Xception model (Fine-tuning mode: {for_finetuning}) ---")

# #     base_model = tf.keras.applications.Xception(
# #         weights='imagenet',
# #         include_top=False,
# #         input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
# #     )

# #     # --- ARCHITECTURE MATCHING LOGIC ---
# #     # This logic exactly replicates the state of the model when the weights were saved.
# #     # The base model itself is frozen, but we can unfreeze layers within it.
# #     base_model.trainable = False

# #     if for_finetuning:
# #         # Unfreeze the last 35 layers INSIDE the base model.
# #         # This is the state required to correctly load the fine-tuned weights.
# #         for layer in base_model.layers[-35:]:
# #             layer.trainable = True
# #         print(f"Fine-tuning mode: Last {len(base_model.layers[-35:])} layers of Xception are now trainable.")
# #     else:
# #         # For initial training, all internal layers are already frozen by base_model.trainable = False
# #         print("Initial training mode: All layers of the Xception base model are frozen.")

# #     # --- MODEL GRAPH DEFINITION (FUNCTIONAL API) ---
# #     inputs = tf.keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    
# #     # The 'training' argument is important for Batch Normalization layers.
# #     # It tells them whether to use learned statistics (inference) or update them (training).
# #     # We set to for_finetuning because we are loading weights from a fine-tuned model.
# #     x = base_model(inputs, training=for_finetuning)
    
# #     x = layers.GlobalAveragePooling2D()(x)
# #     x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
# #     x = layers.Dropout(0.5)(x)
# #     outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

# #     model = tf.keras.Model(inputs, outputs)

# #     # --- COMPILATION ---
# #     # We compile with the fine-tuning learning rate, as this is the state for evaluation.
# #     learning_rate = config.FINETUNE_LEARNING_RATE if for_finetuning else config.INITIAL_LEARNING_RATE
    
# #     model.compile(
# #         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
# #         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
# #         metrics=[
# #             'sparse_categorical_accuracy',
# #             tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
# #         ]
# #     )

# #     print(f"Model compiled with learning rate: {learning_rate}")
# #     return model
# def build_model(for_finetuning: bool = False):
#     """
#     Builds the Keras model, EXACTLY replicating the architecture from the Kaggle notebook blueprint.
#     """
#     print(f"--- Building model with Kaggle blueprint (Fine-tuning: {for_finetuning}) ---")

#     # --- Replicating Layer 0: The nested Sequential model for preprocessing ---
#     # This matches the `sequential_2` layer in your summary.
#     preprocessing_layer = tf.keras.Sequential([
#         layers.InputLayer(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)),
#         layers.Resizing(config.IMAGE_SIZE, config.IMAGE_SIZE),
#         layers.Lambda(xception.preprocess_input),
#     ])

#     # --- Replicating Layer 2: The Xception base model ---
#     base_model = tf.keras.applications.Xception(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
#     )
    
#     # --- Replicating the Trainable Status ---
#     # The blueprint shows the base model itself is NOT trainable, but layers within it are.
#     base_model.trainable = False
#     if for_finetuning:
#         for layer in base_model.layers[-35:]:
#             layer.trainable = True
    
#     # --- Assembling the Main Sequential Model (matching "sequential_3") ---
#     model = tf.keras.Sequential([
#         preprocessing_layer,
#         layers.BatchNormalization(), # This is layer 1 in your summary
#         base_model,                  # This is layer 2 (xception)
#         layers.GlobalAveragePooling2D(), # Layer 3
#         layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)), # Layer 4
#         layers.Dropout(0.5), # Layer 5
#         layers.Dense(config.NUM_CLASSES, activation='softmax') # Layer 6
#     ])

#     # --- Compiling the model ---
#     learning_rate = config.FINETUNE_LEARNING_RATE if for_finetuning else config.INITIAL_LEARNING_RATE
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=[
#             'sparse_categorical_accuracy',
#             tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
#         ]
#     )

#     print(f"Model compiled with learning rate: {learning_rate}")
#     return model

# if __name__ == '__main__':
#     # This block allows you to test the model building script directly
#     print("--- Testing Model Build for Evaluation/Fine-tuning ---")
#     # We build in fine-tuning mode to see the summary that our evaluation script will use.
#     finetune_model = build_model(for_finetuning=True)
#     finetune_model.summary()
# # # src/model.py
# # import tensorflow as tf
# # from tensorflow.keras import layers, regularizers
# # from src import config # Import our central configuration file

# # def build_model(for_finetuning: bool = False):
# #     """
# #     Builds, compiles, and returns the Xception-based Keras model.

# #     Args:
# #         for_finetuning (bool): If True, prepares the model for fine-tuning by
# #         making more layers trainable.

# #     Returns:
# #         A compiled Keras model.
# #     """
# #     print(f"--- Building Xception model (Fine-tuning mode: {for_finetuning}) ---")

# #     # 1. Load the pre-trained Xception base
# #     base_model = tf.keras.applications.Xception(
# #         weights='imagenet',
# #         include_top=False, # Exclude the final classification layer
# #         input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
# #     )

# #     # 2. Set the trainability of the base model layers
# #     if for_finetuning:
# #         # Unfreeze the last 35 layers for fine-tuning
# #         base_model.trainable = True
# #         for layer in base_model.layers[:-35]:
# #             layer.trainable = False
# #         print(f"Fine-tuning enabled: Last {len(base_model.layers[-35:])} layers of Xception are now trainable.")
# #     else:
# #         # Freeze the entire base model for initial feature extraction
# #         base_model.trainable = False
# #         print("Initial training: All layers of the Xception base model are frozen.")

# #     # 3. Create the new classification head
# #     # We use the Keras Functional API for more flexibility
# #     inputs = tf.keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
# #     x = base_model(inputs, training=False if not for_finetuning else True)
# #     x = layers.GlobalAveragePooling2D()(x)
# #     x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
# #     x = layers.Dropout(0.5)(x)
# #     outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

# #     model = tf.keras.Model(inputs, outputs)

# #     # 4. Compile the model with an appropriate learning rate
# #     learning_rate = config.FINETUNE_LEARNING_RATE if for_finetuning else config.INITIAL_LEARNING_RATE
    
# #     model.compile(
# #         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
# #         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
# #         metrics=[
# #             'sparse_categorical_accuracy',
# #             tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
# #         ]
# #     )

# #     print(f"Model compiled with learning rate: {learning_rate}")
# #     model.summary()
# #     return model

# # if __name__ == '__main__':
# #     # This block allows you to test the model building script directly
# #     print("--- Testing Initial Model Build ---")
# #     initial_model = build_model(for_finetuning=False)

# #     print("\n" + "="*50 + "\n")

# #     print("--- Testing Fine-tuning Model Build ---")
# #     finetune_model = build_model(for_finetuning=True)
# src/model.py
# src/model.py

    # print("Building corrected architecture...")
    # base_model = xception.Xception(weights='imagenet', include_top=False, input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    # base_model.trainable = False
    # set_trainable = False
    # for layer in base_model.layers:
    #     if "block11_sepconv1" in layer.name:
    #         set_trainable = True
    #     if set_trainable:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False
    # inputs = tf.keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    # x = tf.keras.layers.BatchNormalization()(inputs)
    # # The training=False argument is crucial here for inference
    # x = base_model(x, training=False)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), kernel_constraint=tf.keras.constraints.MaxNorm(3))(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    # model = tf.keras.Model(inputs, outputs)
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=['sparse_categorical_accuracy']
    # )
    # return model
    # # NOTE: This function now ONLY builds the fine-tuning architecture.
    # # The 'for_finetuning' flag is kept for consistency but will always be True for evaluation.
    # print(f"--- Building Ground Truth Fine-Tuned Kaggle Architecture ---")

    # # --- Replicating the Preprocessing Layer ---
    # preprocessing_layer = tf.keras.Sequential([
    #     layers.InputLayer(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)),
    #     layers.Resizing(config.IMAGE_SIZE, config.IMAGE_SIZE),
    #     layers.Lambda(xception.preprocess_input),
    # ], name="preprocessing_sequential")

    # # --- Replicating the Xception Base Model ---
    # base_model = tf.keras.applications.Xception(
    #     weights='imagenet',
    #     include_top=False,
    #     input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    # )
    
    # --- Replicating the EXACT Fine-Tuning Trainable Status ---
    # The fine-tuning script set the entire base_model to trainable first
    # base_model.trainable = True
    
    # Then it unfroze the last 35 layers (this is slightly redundant but we match it)
    # for layer in base_model.layers[-35:]:
    #     layer.trainable = True
    
    # --- Assembling the Main Sequential Model ---
    # model = tf.keras.Sequential([
    #     preprocessing_layer,
    #     layers.BatchNormalization(),
    #     base_model,
    #     layers.GlobalAveragePooling2D(),
    #     layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    #     layers.Dropout(0.5),
    #     layers.Dense(config.NUM_CLASSES, activation='softmax')
    # ])

    # # --- Replicating the Compilation Step for the Fine-Tuned Model ---
    # learning_rate = config.FINETUNE_LEARNING_RATE # Should be 0.00005
    
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=[
    #         'sparse_categorical_accuracy',
    #         tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    #     ]
    # )

    # print(f"Model compiled with fine-tuning learning rate: {learning_rate}")
    # return model