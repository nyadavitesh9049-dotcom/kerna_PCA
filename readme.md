# Kernel PCA Based Network Traffic Classification

## 📌 Project Overview

This project applies Kernel-based dimensionality reduction techniques and Machine Learning algorithms to classify network traffic.

The project uses network traffic datasets and performs preprocessing, dimensionality reduction, model training, and performance evaluation.

## 📂 Dataset

The project uses network traffic data from:

- Tuesday Working Hours
- Wednesday Working Hours
- Thursday Working Hours - Morning Web Attacks

The datasets are combined using Pandas before preprocessing and model training.

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

- Dataset concatenation
- Removal of unnecessary spaces from column names
- Conversion of features to numeric values
- Handling infinite values
- Handling missing values using SimpleImputer
- Label encoding of target classes
- Train-test splitting

## 🧠 Dimensionality Reduction

The project uses the Nystroem kernel approximation technique for kernel-based feature transformation.

Different kernel functions can be tested, including:

- RBF
- Polynomial
- Linear
- Sigmoid
- Cosine

## 🤖 Machine Learning Algorithms

The following algorithms are evaluated:

1. AdaBoost
2. CatBoost
3. LightGBM
4. Naive Bayes
5. XGBoost

## 📊 Evaluation Metrics

The performance of the models is evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- CatBoost
- LightGBM
- VS Code
- Git
- GitHub

## 👥 Contributors

Ramashankar prajapati
nitesh yadav


## 📌 Project Purpose

This project was developed as an academic Machine Learning project to study dimensionality reduction and compare the performance of different classification algorithms on network traffic data.