Plant Disease Detection using CNN
Project Overview

This project focuses on detecting and classifying plant diseases using a Convolutional Neural Network (CNN) trained on the PlantVillage dataset.

The project was developed as part of the EDP Internship Project and covers CNN training, validation, model evaluation, per-class accuracy analysis, classification report, and confusion matrix generation.

Weeks Covered
Week 5 — CNN Model Training

The following tasks were completed:

Loaded the PlantVillage image dataset
Created training and validation datasets
Applied image preprocessing
Built a CNN-based image classification model
Trained the model for 5 epochs
Evaluated training and validation performance
Saved the trained model in .keras format
Week 6–7 — Model Evaluation

The following evaluation tasks were completed:

Loaded the trained CNN model
Evaluated the model using validation images
Calculated accuracy for each disease class
Generated a classification report
Generated a confusion matrix
Analyzed the model's performance across different diseases
Project Structure
edpweek5/
│
├── week5_training.ipynb
├── plant_disease_cnn.keras
│
├── edpweek6_7/
│   └── week6_7_evaluation.ipynb
│
├── .gitattributes
└── README.md
Dataset

The project uses a subset of the PlantVillage dataset containing approximately 20,637 images across 15 plant disease classes.

Dataset Split
Training: 80%
Validation: 20%
Validation images: 4,127
Plant Disease Classes
Pepper Bell Bacterial Spot
Pepper Bell Healthy
Potato Early Blight
Potato Late Blight
Potato Healthy
Tomato Bacterial Spot
Tomato Early Blight
Tomato Late Blight
Tomato Leaf Mold
Tomato Septoria Leaf Spot
Tomato Spider Mites
Tomato Target Spot
Tomato Yellow Leaf Curl Virus
Tomato Mosaic Virus
Tomato Healthy
CNN Model

The project uses a Convolutional Neural Network for plant disease classification.

The model consists of:

Convolutional layers
Pooling layers
Flatten layer
Dense layer with 128 neurons
Output layer with 15 classes
Model Parameters

Total Parameters: 11,170,895

The trained model is stored as:

plant_disease_cnn.keras
Week 5 — Training Results

The CNN model was trained for 5 epochs.

Metric	Result
Training Accuracy	96.81%
Training Loss	0.0980
Validation Accuracy	88.39%
Validation Loss	0.4135
Final Validation Accuracy

88.39%

The CNN successfully learned to classify plant diseases and achieved good performance on unseen validation images.

Week 6–7 — Per-Class Accuracy
Plant Disease	Accuracy
Pepper Bell Bacterial Spot	76.14%
Pepper Bell Healthy	97.86%
Potato Early Blight	97.94%
Potato Late Blight	82.59%
Potato Healthy	46.43%
Tomato Bacterial Spot	94.94%
Tomato Early Blight	60.00%
Tomato Late Blight	85.60%
Tomato Leaf Mold	81.34%
Tomato Septoria Leaf Spot	84.87%
Tomato Spider Mites	95.00%
Tomato Target Spot	77.11%
Tomato Yellow Leaf Curl Virus	94.96%
Tomato Mosaic Virus	98.67%
Tomato Healthy	96.20%
Model Evaluation
Classification Report

A classification report was generated to evaluate the model using:

Precision
Recall
F1-score
Support

for each plant disease class.

Confusion Matrix

A confusion matrix was generated to visualize the correctly and incorrectly classified images for all 15 classes.

The confusion matrix helps identify classes that are easy for the model to recognize and classes that may require further improvement.

Technologies Used
Python
TensorFlow
Keras
NumPy
Matplotlib
Scikit-learn
Jupyter Notebook
Git
GitHub
Git LFS
Development Environment
Python 3.11
TensorFlow 2.15.0
Keras 2.15.0
How to Use
Clone the Repository
git clone https://github.com/sarvajeethdiguva/edpweek5and6.git
Week 5 — Training

Open:

week5_training.ipynb

The notebook contains:

Dataset loading
Image preprocessing
Dataset splitting
CNN architecture
Model training
Training and validation
Model saving
Week 6–7 — Evaluation

Open:

edpweek6_7/week6_7_evaluation.ipynb

The notebook contains:

Trained model loading
Validation evaluation
Per-class accuracy
Classification report
Confusion matrix
Trained Model

The trained CNN model is stored in:

plant_disease_cnn.keras

The model file is approximately 134 MB. Therefore, Git LFS (Git Large File Storage) is used to store the trained model in the GitHub repository.

Project Outcome

The CNN-based Plant Disease Detector achieved a validation accuracy of 88.39%.

The Week 6–7 evaluation provided detailed information about the performance of the model for each individual plant disease class.

The project demonstrates the use of deep learning and computer vision for automated plant disease classification.

Future Work

The project can be further improved by:

Increasing the training dataset
Applying data augmentation
Using transfer learning
Performing hyperparameter tuning
Improving accuracy for weaker classes
Deploying the model as a web application
Creating a mobile-friendly interface
Enabling real-time plant disease detection
Project Information

Project: Plant Disease Detector

Internship: EDP Internship Project

Weeks: 5, 6 & 7

Domain: Deep Learning / Computer Vision / Agriculture

Model: Convolutional Neural Network (CNN)

Dataset: PlantVillage

Validation Accuracy: 88.39%