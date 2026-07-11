# 💳 Credit Card Fraud Detection Using XGBoost & Flask

## 📌 Project Overview

This project is a Machine Learning-based Credit Card Fraud Detection System that identifies whether a credit card transaction is **Legitimate** or **Fraudulent** in real time.

The model is trained using the **XGBoost Classifier** and handles class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**. A Flask web application provides an interactive interface where users can enter transaction details and receive instant fraud predictions along with the fraud probability, risk level, and recommendations.

---

## 🚀 Features

- Real-time fraud prediction
- XGBoost classification model
- SMOTE for handling imbalanced data
- Feature engineering for improved performance
- Fraud probability estimation
- Risk level classification (High / Medium / Low)
- Transaction recommendation system
- Interactive Flask web interface
- Responsive HTML & CSS design

---

## 🛠 Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Joblib
- HTML5
- CSS3

---

## 📊 Dataset Features

The model uses the following transaction attributes:

- Transaction Amount
- Transaction Hour
- Merchant Category
- Foreign Transaction
- Location Mismatch
- Device Trust Score
- Transaction Velocity (Last 24 Hours)
- Cardholder Age

### Engineered Features

- High Amount
- Night Transaction
- High Velocity
- Geo Risk
- Low Trust Device

---

## 🤖 Machine Learning Workflow

1. Data Preprocessing
2. Feature Engineering
3. Label Encoding
4. Feature Scaling (StandardScaler)
5. Train-Test Split
6. SMOTE Oversampling
7. XGBoost Model Training
8. Model Evaluation
9. Model Saving (.pkl)
10. Flask Web Deployment

---

## 📈 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | **99.70%** |
| F1 Score | **0.906** |
| AUC-ROC | **0.999** |
| RMSE | **0.050** |

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── app.py
├── main.py
├── credit_card_fraud.csv
├── fraud_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/capstone-project.git
```

Move into the project directory

```bash
cd capstone-project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## 💻 Application Workflow

1. Enter transaction details.
2. Click **Detect Fraud**.
3. The system predicts whether the transaction is legitimate or fraudulent.
4. Displays:
   - Fraud Probability
   - Risk Level
   - Recommendation
   - Transaction Details
   - Detection Time

---



## 🔮 Future Enhancements

- Autoencoder-based anomaly detection
- Isolation Forest implementation
- Email/SMS fraud alerts
- Payment gateway integration
- REST API deployment
- Docker containerization
- Cloud deployment (Render/AWS/Azure)

---

## 📚 Learning Outcomes

This project demonstrates practical implementation of:

- Data Preprocessing
- Feature Engineering
- Handling Imbalanced Data
- Supervised Machine Learning
- Model Serialization
- Flask Web Development
- Machine Learning Deployment

---

## 📄 License

This project is developed for educational and academic purposes.
