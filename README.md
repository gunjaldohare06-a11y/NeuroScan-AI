# 🧠 NeuroScan AI — Brain Tumor Detection System

> **AI-powered Brain Tumor Detection from MRI Images using Deep Learning**

NeuroScan AI is a deep learning-based brain tumor detection system that analyzes **MRI brain images** and classifies them into four categories using a **Convolutional Neural Network (CNN)**.

The project is designed as an **AIML academic project** with a user-friendly web interface where users can upload an MRI image and receive the model's predicted class along with its confidence score.

---

## 🚀 Project Overview

Brain tumors are abnormal growths of cells in the brain that may require early diagnosis and treatment.

**NeuroScan AI** uses **Deep Learning and Computer Vision** techniques to automatically analyze MRI scans and classify them into:

* 🧠 **Glioma**
* 🧠 **Meningioma**
* 🧠 **Pituitary Tumor**
* ✅ **No Tumor**

The system is intended as an **educational and research project** demonstrating the application of Artificial Intelligence in medical image classification.

---

## ✨ Features

* 🧠 MRI Brain Image Classification
* 🤖 CNN-based Deep Learning Model
* 🔬 Four-class tumor classification
* 📤 Upload MRI images for prediction
* 📊 Prediction confidence score
* 🔐 User Login Interface
* 📈 Interactive Dashboard
* 🎨 Modern and responsive UI
* 🧪 Model testing and evaluation
* 💻 Local execution through VS Code
* 🐙 GitHub-ready project structure

---

## 🧠 Prediction Classes

| Class         | Description                                      |
| ------------- | ------------------------------------------------ |
| 🧠 Glioma     | A type of tumor originating from glial cells     |
| 🧠 Meningioma | A tumor that develops from the meninges          |
| 🧠 Pituitary  | Tumor occurring in or around the pituitary gland |
| ✅ No Tumor    | MRI image classified as having no detected tumor |

---

## 📸 Screenshots

### 🔐 Login Page

<img width="1920" height="1020" alt="Screenshot 2026-08-09 161454" src="https://github.com/user-attachments/assets/c0466abc-ff10-4522-a854-c3acf06ee3e6" />


### 📊 Dashboard

<img width="1920" height="1020" alt="Screenshot 2026-08-09 161554" src="https://github.com/user-attachments/assets/1a2ec85a-9523-4566-b6ee-e61d9e84826b" />



<img width="1920" height="1020" alt="Screenshot 2026-08-09 161619" src="https://github.com/user-attachments/assets/a4bb81ba-2dab-4cf0-b3ab-8263d1959f9f" />



### 🧠 MRI Prediction Result

<img width="1603" height="867" alt="Screenshot 2026-08-09 115949" src="https://github.com/user-attachments/assets/df3dd4c9-7eb1-45ee-8917-6a1ae62c8ad2" />


<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/cdfea8c0-a0b0-4ba3-bb2c-3893f5d2abd4" />


> Replace the screenshot files with your actual project screenshots.

---

## 🛠️ Tech Stack

### Programming Language

* 🐍 Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* NumPy
* Pandas
* Scikit-learn

### Image Processing

* OpenCV
* PIL / Pillow

### Data Visualization

* Matplotlib
* Seaborn

### Web Development

* HTML
* CSS
* JavaScript
* Flask / Streamlit

### Development Tools

* Visual Studio Code
* Git
* GitHub

---

## 🗂️ Dataset

The project uses a brain MRI image dataset containing four categories:

```text
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

The dataset is divided into:

* **Training Dataset** — used to train the CNN model
* **Testing Dataset** — used to evaluate model performance

---

## 🤖 Machine Learning Workflow

```text
MRI Image Dataset
       ↓
Data Preprocessing
       ↓
Image Resizing
       ↓
Normalization
       ↓
Data Augmentation
       ↓
CNN Model
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Trained Model
       ↓
MRI Image Upload
       ↓
Prediction
       ↓
Tumor Class + Confidence
```

---

## 🧠 CNN Model

The system uses a **Convolutional Neural Network (CNN)** for image classification.

A typical CNN architecture consists of:

```text
Input MRI Image
       ↓
Convolution Layer
       ↓
ReLU Activation
       ↓
Max Pooling
       ↓
Convolution Layer
       ↓
ReLU Activation
       ↓
Max Pooling
       ↓
Flatten
       ↓
Dense Layer
       ↓
Dropout
       ↓
Output Layer
       ↓
4 Class Prediction
```

The final output represents the probability of each class:

```text
Glioma       → XX%
Meningioma   → XX%
Pituitary    → XX%
No Tumor     → XX%
```

The class with the highest probability is selected as the model's prediction.

---

## 📁 Project Structure

```text
NeuroScan-AI/
│
├── dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   │
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── model/
│   └── brain_tumor_model.keras
│
├── screenshots/
│   ├── login.png
│   ├── dashboard.png
│   └── result.png
│
├── templates/
│   ├── login.html
│   ├── dashboard.html
│   └── result.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/NeuroScan-AI.git
```

### 2. Open the Project

```bash
cd NeuroScan-AI
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Train the Model

To train the CNN model:

```bash
python train.py
```

After training, the model will be saved inside:

```text
model/
└── brain_tumor_model.keras
```

---

## 🔍 Run Prediction

To test the model with an MRI image:

```bash
python predict.py
```

The system will process the image and return the predicted tumor category.

---

## 🌐 Run the Web Application

Start the application using:

```bash
python app.py
```

Then open the local URL shown in the terminal, for example:

```text
http://127.0.0.1:5000
```

---

## 📊 Model Evaluation

The trained model can be evaluated using:

* Accuracy
* Loss
* Confusion Matrix
* Classification Report
* Precision
* Recall
* F1-Score

Example:

```text
Model Evaluation
----------------
Accuracy  : XX%
Precision : XX%
Recall    : XX%
F1 Score  : XX%
```

> Replace `XX%` with the actual values obtained after training and evaluation.

---

## 🔐 Login System

NeuroScan AI includes a login interface to provide a structured user experience.

```text
Login
  ↓
Dashboard
  ↓
Upload MRI
  ↓
AI Prediction
  ↓
Result
```

---

## 📊 Dashboard

The dashboard provides access to the main functionality of the system.

Users can:

* Upload MRI images
* Start prediction
* View prediction results
* View confidence score
* Navigate through the application

---

## 👥 Group Members

| Sr. No. | Name                  | Role                        |
| ------: | --------------------- | --------------------------- |
|       1 | **Gunjal Dohare**     | Project Lead & ML Developer |
|       2 | **Aryan Shahare**     | Frontend & UI Developer     |
|       3 | **Hemant Rahangdale** | Backend & Testing           |

---

## 👨‍💻 Responsibilities

### 👨‍💻 Gunjal Dohare

**Project Lead & ML Developer**

* Dataset preparation
* Data preprocessing
* CNN model development
* Model training
* Model evaluation
* AI prediction implementation
* Project integration

### 🎨 Aryan Shahare

**Frontend & UI Developer**

* Login page
* Dashboard UI
* MRI upload interface
* Result page
* HTML/CSS/JavaScript
* User experience and responsive design

### ⚙️ Hemant Rahangdale

**Backend & Testing**

* Backend integration
* API/application logic
* Testing
* Error handling
* Model integration support
* Application validation

---

## 🔮 Future Scope

Future improvements may include:

* 📱 Mobile application
* ☁️ Cloud deployment
* 🧠 Advanced CNN architectures
* ⚡ Faster prediction
* 📊 Detailed medical image visualization
* 🔬 Explainable AI (XAI)
* 🩻 Grad-CAM visualization
* 🗃️ Prediction history
* 👨‍⚕️ Doctor-oriented dashboard
* 🔐 Improved authentication and security

---

## ⚠️ Disclaimer

**NeuroScan AI is an educational and research project.**

The predictions generated by this system **should not be considered a medical diagnosis**. The system is intended to demonstrate the use of Artificial Intelligence and Deep Learning for brain MRI image classification.

For actual medical diagnosis or treatment decisions, users should consult a qualified medical professional.

---

## 📌 Project Status

```text
🚧 Project Status: Under Development
```

Current development includes:

* [x] Dataset collection
* [x] Dataset organization
* [ ] Data preprocessing
* [ ] CNN model training
* [ ] Model evaluation
* [ ] Prediction system
* [ ] Dashboard integration
* [ ] Login system
* [ ] Final testing
* [ ] GitHub deployment

---

## ⭐ Contributing

Contributions, suggestions, and improvements are welcome.

If you want to contribute:

```bash
git fork
git clone
git checkout -b feature-name
git add .
git commit -m "Add new feature"
git push
```

Then create a Pull Request.

---

## 📄 License

This project is developed for **educational and academic purposes**.

---

# 🧠 NeuroScan AI

### *AI-Powered Brain Tumor Detection from MRI Images*

**Developed by Gunjal Dohare, Aryan Shahare & Hemant Rahangdale**

⭐ If you find this project useful, consider giving the repository a star!
