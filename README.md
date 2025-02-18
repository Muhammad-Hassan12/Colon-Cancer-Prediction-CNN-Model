# Colon Cancer Detection Using CNN
This repository contains the code and trained models for **Colon Cancer Detection** using **Convolutional Neural Networks (CNNs)**. The model is trained to classify histopathological images as **Colon Adenocarcinoma (Cancerous)** or **Colon Benign Tissue (Non-Cancerous)**.

## Model Download Links:
1. **C_normal_1.h5** --> https://drive.google.com/file/d/1QNL5HhEovKFXXxVYyCPWoGKc6DAQb0eT/view?usp=sharing
2. **C_large_1.h5**  --> https://drive.google.com/file/d/1cHY_g2W8CW-gCkKEt8hc9_WuYcb2epOK/view?usp=sharing
3. **C_WDA_1.h5**    --> https://drive.google.com/file/d/1GW6A_f0vpxP1xCzJi9KthyqlJs5W40tP/view?usp=sharing
4. **C_DA_1.h5**     --> https://drive.google.com/file/d/15K5rQ3_50YnuGbA3o6dfD3wZLDhOnk51/view?usp=sharing
5. **C_DA_2.h5**     --> https://drive.google.com/file/d/1_0gPIU-raz3xLglNd5eWjuoy076Fb3m4/view?usp=sharing

## Project Overview
This project aims to develop an **AI-based classifier** that helps in **early detection of colon cancer** by analyzing medical images. The deep learning model was trained on a dataset of **Colon Adenocarcinoma** and **Colon Benign Tissue** images, achieving high accuracy.

## Repository Structure
```console
📦 Colon-Cancer-Detection
│── 📂 notebooks/                     # Jupyter Notebooks for training and evaluation
    │── 📜 colon-c-training.ipynb               # Model Traing Notebook
    │── 📜 Report Testing.ipynb                 # Model Evaluation (Accuracy, loss, and confusion matrix analysis)
│── 📂 Report/                        # Trained models testing reports
│── 📂 Training Graph/                # Model performance graphs
│── 📜 requirements.txt               # Python dependencies
│── 📜 README.md                      # Project documentation (this file)
```

## Features
✅ Binary Classification: Detects whether an image is Cancerous or Benign
✅ Deep Learning Model: CNN-based architecture trained using TensorFlow/Keras
✅ Data Augmentation: Improves model generalization
✅ Model Evaluation: Accuracy, loss, and confusion matrix analysis
✅ Pre-trained Models: Ready to use for inference

## Models Performance
### 1. **C_normal_1.h5** (Normal Training)
   * Training Accuracy: **93.54%**
   * Validation Accuracy: **93.78%**
   * Test Accuracy: **94.89%**

### 2. **C_large_1.h5** (Data Augmentation with Droupout)
   * Training Accuracy: **97.70%**
   * Validation Accuracy: **93.78%**
   * Test Accuracy: **93.33%**

### 3. **C_WDA_1.h5** (Fast Feature Extraction Without Data Augmentation)
   * Training Accuracy: **100%**
   * Validation Accuracy: **99.56%**

### 4. **C_DA_1.h5** (Feature Extraction with Data Augmentation: 1st Try)
   * Training Accuracy: **100%**
   * Validation Accuracy: **99.89%**
   * Test Accuracy: **99.89%**

### 5. **C_DA_2.h5** (Feature Extraction with Data Augmentation: 2st Try)
   * Training Accuracy: **100%**
   * Validation Accuracy: **100%**
   * Test Accuracy: **100%**

**Model performance has been validated using accuracy/loss curves, confusion matrices, and classification reports!**

## Installation & Setup
### Clone the Repository
```console
git clone https://github.com/Muhammad-Hassan12/Colon-Cancer-Prediction-CNN-Model.git
cd Colon-Cancer-Prediction-CNN-Model
```

### Install Dependencies
```console
pip install -r requirements.txt
```

### Run the Training Notebook (If you want to train your self!)
```console
jupyter notebook colon-c-training.ipynb
```

### Run the Model on Test Data for Evaluation
```console
jupyter notebook Report Testing.ipynb
```

## Dataset
The Dataset I used is **"Lung and Colon Cancer Histopathological Images"** by **"Larxel"** from **Kaggle**!
You can download it yourself: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images
The dataset consists of Colon Adenocarcinoma and Colon Benign Tissue images. The images are resized to 256x256 before training.

## Model Training Details
* Framework: TensorFlow/Keras
* Optimizer: RMSprop
* Loss Function: Binary Cross-Entropy
* Evaluation Metrics: Accuracy, Precision, Recall, F1-score

## Results and Visualizations
### 1. *C_normal_1.h5*
1. **Training & Validation Accuracy & Loss:**
* ![C_normal_1](https://github.com/user-attachments/assets/42941503-f0ef-4646-becc-1bb170949d43)

2. **Confusion Matrix:**
* ![Report](https://github.com/user-attachments/assets/bf6c3883-33e0-4345-85e0-9709d7ec6276)

### 2. *C_large_1.h5*
1. **Training & Validation Accuracy & Loss:**
* ![C_large_1](https://github.com/user-attachments/assets/30a95714-498b-4e07-83e1-21d3689d3d75)

2. **Confusion Matrix:**
* ![Report](https://github.com/user-attachments/assets/f4ca1141-495c-4ae8-bae1-aad98ec97d04)

### 3. *C_WDA_1.h5*
1. **Training & Validation Accuracy & Loss:**
* ![C_WDA_1](https://github.com/user-attachments/assets/45443f87-b3bb-46e4-b85d-058a83873365)

### 4. *C_DA_1.h5*
1. **Training & Validation Accuracy & Loss:**
* ![C_DA_1](https://github.com/user-attachments/assets/739d5e9a-b72d-4e30-b5b4-caeefc317464)

2. **Confusion Matrix:**
* ![Report](https://github.com/user-attachments/assets/44a1dbe4-ab6d-4c35-a9c9-6904285bc88b)

### 5. *C_DA_2.h5*
1. **Training & Validation Accuracy & Loss:**
* ![C_DA_2](https://github.com/user-attachments/assets/67c606d9-21d2-466a-a8b7-c93ca07868de)

2. **Confusion Matrix:**
* ![Report](https://github.com/user-attachments/assets/3dcb1bb5-9d30-46e8-8105-4d8fab771c97)

## Example Predictions
* Input: Colon biopsy image
* Model Output: Probability score (Cancerous vs. Benign)

## Future Improvements
🔹 Improve dataset diversity
🔹 Experiment with different CNN architectures
🔹 Deploy the model as a web app for easy access

## License
This project is **open-source** under the **MIT License**. Feel free to contribute!

## Special Thanks To!
**Kaggle Cloud Jupyter Notebook!**
