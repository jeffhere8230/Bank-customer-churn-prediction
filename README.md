# Bank-customer-churn-prediction
ann model to predict whether the customer of the bank do exit or not!
# Bank Customer Churn Prediction

A Deep Learning project that predicts whether a bank customer is likely to leave the bank using an Artificial Neural Network (ANN).

## Features

* Customer churn prediction using ANN
* Streamlit-based interactive web application
* Data preprocessing and feature scaling
* Real-time prediction with churn probability
* Model evaluation using Accuracy, Confusion Matrix, and ROC-AUC

## Technologies Used

* Python
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib

## Dataset Features

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

## Model Architecture

* Dense Layer (6 neurons, ReLU)
* Dense Layer (6 neurons, ReLU)
* Output Layer (1 neuron, Sigmoid)

## Run the Project

Install dependencies:

```bash
pip install streamlit tensorflow pandas numpy scikit-learn matplotlib joblib
```

Run the application:

```bash
streamlit run app.py
```

## Output

The application predicts:

* Customer Stay
* Customer Exit
* Churn Probability (%)

## Author

ANN-based Bank Customer Churn Prediction System developed using TensorFlow and Streamlit.
