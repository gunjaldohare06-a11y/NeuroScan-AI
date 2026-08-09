import tensorflow as tf

# ==============================
# PATHS
# ==============================

MODEL_PATH = "model/brain_tumor_model.keras"
TEST_DIR = "dataset/Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# LOAD TRAINED MODEL
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==============================
# LOAD TEST DATA
# ==============================

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==============================
# EVALUATE MODEL
# ==============================

print("\n===== MODEL EVALUATION =====")

loss, accuracy = model.evaluate(test_data)

print("\n===== FINAL TEST RESULT =====")

print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")