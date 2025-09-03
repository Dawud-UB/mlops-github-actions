# CI/CD Workflow for MLOps with GitHub Actions

This repository demonstrates a **simple CI/CD pipeline for Machine Learning** using GitHub Actions.  
The example trains a **Logistic Regression model** on the Breast Cancer dataset from scikit-learn,  
runs unit tests, and delivers the trained model (`model.pkl`) together with training metrics (`metrics.txt`) as **GitHub Action artifacts**.

---

## 📌 Workflow Overview

- **Continuous Integration (CI)**  
  - On every push or pull request to `main`:  
    - Install dependencies  
    - Train the ML model  
    - Run unit tests  

- **Continuous Delivery (CD)**  
  - Deliver the trained model and metrics as artifacts  
  - These can be downloaded directly from the **Actions tab** on GitHub  

---

## 📂 Repository Structure

```
mlops-github-actions/
│
├── src/
│ ├── train.py # Training script (Breast Cancer dataset)
│ └── test_model.py # Unit tests for model
│
├── requirements.txt # Dependencies
└── .github/
└── workflows/
└── ci.yml # CI/CD workflow definition
```

---

## ⚙️ How It Works

### 1. Training
The `train.py` script:
- Loads Breast Cancer dataset  
- Trains Logistic Regression model  
- Saves the model to `model.pkl`  
- Evaluates training accuracy and writes result to `metrics.txt`

### 2. Testing
The `test_model.py` script:
- Ensures `model.pkl` is created  
- Checks the model can make predictions  

### 3. Workflow (ci.yml)
Workflow steps:
```yml
- Checkout repository
- Setup Python environment
- Install dependencies
- Train model
- Run tests
- Upload artifacts (model.pkl + metrics.txt)
