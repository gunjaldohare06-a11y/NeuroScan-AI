import tensorflow as tf
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# PATHS
# ==============================

MODEL_PATH = "model/brain_tumor_model.keras"
TEST_DIR = "dataset/Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# LOAD MODEL
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
# PREDICTIONS
# ==============================

print("\nGenerating predictions...")

predictions = model.predict(test_data)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data.classes

class_names = list(test_data.class_indices.keys())

# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\n===== CONFUSION MATRIX =====")
print(cm)

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)

# ==============================
# DISPLAY CONFUSION MATRIX
# ==============================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Brain Tumor Detection - Confusion Matrix")

plt.tight_layout()
plt.show()