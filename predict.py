import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# MODEL
# ==============================

MODEL_PATH = "model/brain_tumor_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# ==============================
# CLASS NAMES
# ==============================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ==============================
# IMAGE PREDICTION FUNCTION
# ==============================

def predict_brain_tumor(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    image_path = input("Enter MRI image path: ")

    result, confidence = predict_brain_tumor(image_path)

    print("\n===== NEUROSCAN AI RESULT =====")

    print("Prediction:", result)

    print("Confidence:", round(confidence, 2), "%")