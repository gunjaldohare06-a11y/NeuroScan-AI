import tensorflow as tf

# ==============================
# DATASET PATHS
# ==============================

TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"

# ==============================
# IMAGE SETTINGS
# ==============================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# DATA GENERATORS
# ==============================

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

# Training data
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Validation data
validation_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Testing data
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==============================
# CNN MODEL
# ==============================

model = tf.keras.Sequential([

    # Input
    tf.keras.layers.Input(shape=(224, 224, 3)),

    # Convolution Block 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Feature extraction
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Prevent overfitting
    tf.keras.layers.Dropout(0.5),

    # Output Layer
    tf.keras.layers.Dense(
        4,
        activation="softmax"
    )
])

# ==============================
# COMPILE MODEL
# ==============================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# DISPLAY MODEL
# ==============================

model.summary()
# ==============================
# MODEL TRAINING
# ==============================

EPOCHS = 10

print("\n===== STARTING MODEL TRAINING =====\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

print("\n===== MODEL TRAINING COMPLETE =====")
# ==============================
# SAVE TRAINED MODEL
# ==============================

MODEL_PATH = "model/brain_tumor_model.keras"

model.save(MODEL_PATH)

print("\n===== MODEL SAVED SUCCESSFULLY =====")
print("Model saved at:", MODEL_PATH)