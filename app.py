import os

# Disable GPU for Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import uuid
import tensorflow as tf


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "model/brain_tumor_model.tflite"
UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# LOAD TRAINED TFLITE MODEL
# ==========================================

print("Loading NeuroScan AI TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully!")


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
    return render_template("login.html")


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # ======================================
    # CHECK IMAGE
    # ======================================

    if "mri_image" not in request.files:
        return "No image uploaded", 400

    file = request.files["mri_image"]

    if file.filename == "":
        return "No image selected", 400


    # ======================================
    # SAVE IMAGE
    # ======================================

    extension = os.path.splitext(file.filename)[1]

    filename = str(uuid.uuid4()) + extension

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

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
    # TFLITE MODEL PREDICTION
    # ======================================

    interpreter.set_tensor(
        input_details[0]["index"],
        image_array.astype(np.float32)
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]["index"]
    )


    # ======================================
    # PREDICTED CLASS
    # ======================================

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100


    # ======================================
    # ALL CLASS PROBABILITIES
    # ======================================

    probabilities = {
        class_names[i]: round(
            float(predictions[0][i] * 100),
            2
        )
        for i in range(len(class_names))
    }


    # ======================================
    # RESULT
    # ======================================

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=round(float(confidence), 2),
        image_path=filepath,
        probabilities=probabilities
    )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )