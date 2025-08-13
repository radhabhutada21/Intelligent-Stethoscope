# Intelligent Stethoscope – AI-Powered Respiratory Disease Detection

This project is a **CNN-based web application** that detects respiratory diseases from lung sound recordings using **MFCC features** extracted from audio.  
It provides a **cost-effective, non-invasive, and accurate diagnostic solution** for conditions like Bronchiectasis, Bronchiolitis, COPD, Pneumonia, and URTI.

---

## 📌 Features
- 🎯 **Disease Detection** from lung sounds using Deep Learning
- 🎨 **User-friendly Flask Web Interface**
- 🎵 **MFCC Audio Feature Extraction** using Librosa
- 📊 **CNN Model** built with TensorFlow/Keras
- 📈 Displays **confidence scores** for each disease

---

## 🛠 Tech Stack
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Jinja2 Templates
- **Deep Learning**: TensorFlow, Keras
- **Audio Processing**: Librosa, NumPy
- **Data Handling**: Pandas, Scikit-learn

---

## 📊 Dataset
This project uses the **ICBHI Respiratory Sound Database** from Kaggle:
[Kaggle Dataset Link](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)

The dataset contains **audio samples** and **patient diagnosis labels** for respiratory diseases.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
git clone https://github.com/username/intelligent-stethoscope.git
cd intelligent-stethoscope


2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Download the dataset
Use KaggleHub to download the dataset:


import kagglehub
path = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
print("Dataset downloaded to:", path)
Or download manually from the Kaggle link.

4️⃣ Train the model

python train.py
This will save the trained model as model.h5.

5️⃣ Run the Flask app

python app.py
Access the app at http://127.0.0.1:5000/

🧠 Model Architecture
Input: MFCC features extracted from audio

Layers:

Conv2D → MaxPooling2D → Dropout

Conv2D → MaxPooling2D → Dropout

Flatten → Dense (ReLU) → Dropout

Dense (Softmax)

Loss: Categorical Crossentropy

Optimizer: Adam

📌 Example Prediction

Finished feature extraction from audio.wav

Bronchiectasis : 0.00%

Bronchiolitis : 0.00%

COPD           : 100.00%

Healthy        : 0.00%

Pneumonia      : 0.00%

URTI           : 0.00%
