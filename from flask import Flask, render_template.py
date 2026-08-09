from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "model/brain_tumor_model.keras"
UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading NeuroScan AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether image was uploaded
    if "mri_image" not in request.files:
        return "No image uploaded", 400

    file = request.files["mri_image"]

    # Check filename
    if file.filename == "":
        return "No image selected", 400

    # Generate unique filename
    extension = os.path.splitext(file.filename)[1]

    filename = str(uuid.uuid4()) + extension

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    # Save image
    file.save(filepath)

    # ======================================
    # IMAGE PREPROCESSING
    # ======================================

    image = Image.open(filepath).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ======================================
    # MODEL PREDICTION
    # ======================================

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    # ======================================
    # SEND RESULT TO WEBSITE
    # ======================================

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=round(float(confidence), 2),
        image_path=filepath
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True
    )