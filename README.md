# 🏦 Bank Customer Churn Prediction using ANN

An Artificial Neural Network (ANN)-based machine learning model that predicts whether a bank customer is likely to churn based on demographic and financial information. The project includes data preprocessing, model training, evaluation, visualization, and model serialization for future deployment.

---

## 🚀 Project Highlights

- ANN-based binary classification model
- Data preprocessing and feature scaling
- Customer churn prediction
- Early stopping to prevent overfitting
- Learning rate scheduling
- Performance evaluation using multiple metrics
- Model and scaler serialization
- Training visualization

---

## 📂 Dataset

**Dataset:** `Churn_Modelling.csv`

### Features

| Feature | Description |
|---------|-------------|
| CreditScore | Customer credit score |
| Geography | Customer country |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products |
| HasCrCard | Credit card holder |
| IsActiveMember | Active customer status |
| EstimatedSalary | Estimated annual salary |

**Target Variable**

| Value | Meaning |
|------|---------|
| 0 | Customer Stays |
| 1 | Customer Leaves |

---

## ⚙️ Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Label Encoding
   │
   ▼
Feature Scaling
   │
   ▼
Train-Test Split
   │
   ▼
ANN Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Prediction
   │
   ▼
Save Model
```

---

## 🧠 Model Architecture

| Layer | Activation | Neurons |
|------|------------|---------|
| Hidden Layer 1 | ReLU | 6 |
| Hidden Layer 2 | ReLU | 6 |
| Output Layer | Sigmoid | 1 |

### Training Configuration

| Parameter | Value |
|----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 100 |
| Batch Size | 32 |

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## 📈 Output

The project generates:

- Training Accuracy Curve
- Validation Accuracy Curve
- Training Loss Curve
- Validation Loss Curve
- Customer Churn Prediction

---

## 💾 Saved Files

```
churn_model.keras
scaler.pkl
```

---

## 📁 Project Structure

```
Bank-Customer-Churn-Prediction/
│── Churn_Modelling.csv
│── main.py
│── churn_model.keras
│── scaler.pkl
│── README.md
│── requirements.txt
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

---

## ▶️ Installation

```bash
git clone https://github.com/your-username/Bank-Customer-Churn-Prediction.git

cd Bank-Customer-Churn-Prediction

pip install -r requirements.txt

python main.py
```

---

## 🔮 Future Improvements

- Hyperparameter Optimization
- Explainable AI (SHAP/LIME)
- Streamlit Dashboard
- REST API Deployment
- Docker Support
- Cloud Deployment

---

## 👨‍💻 Author

**Jeff John**

B.Tech Computer Science (Artificial Intelligence)

---

## 📄 License

This project is developed for educational and learning purposes.
