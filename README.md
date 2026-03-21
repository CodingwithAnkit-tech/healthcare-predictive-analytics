# healthcare-predictive-analytics

A complete Healthcare Predictive Analytics project using Machine Learning to predict disease risks such as diabetes. Includes data preprocessing, EDA, feature engineering, model training, evaluation, explainability (SHAP), and deployment.

# 🩺 Healthcare Predictive Analytics (Diabetes Risk Prediction)

This project predicts **Diabetes Risk** using Machine Learning on the  
**Pima Indians Diabetes Dataset**.  
It includes end-to-end data processing, model training, evaluation,  
and a **Streamlit Web App** for real-time prediction.

---

## 🚀 Features
- ✔ End-to-end ML Pipeline  
- ✔ Data Cleaning & Preprocessing (Handling Missing/Zero Values)  
- ✔ XGBoost Model with AUROC ~0.82  
- ✔ Real-time Prediction using Streamlit  
- ✔ Modular and Production-ready Code Structure  
- ✔ Easy to run & deploy  

---

## 📂 Project Folder Structure

```
healthcare-predictive-analytics/
│
├── data/
│   ├── raw/               # Original dataset (diabetes.csv)
│   └── processed/         # Cleaned dataset (processed.csv)
│
├── src/
│   ├── data_processing.py # Cleaning & preprocessing
│   ├── train.py           # Train ML model
│   ├── predict.py         # Predict new values
│   └── evaluate.py        # Model evaluation
│
├── models/
│   └── model.pkl          # Trained model saved here
│
├── app.py                 # Streamlit Web App
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```


## 🧠 Technical Stack


**Languages & Libraries:**
- Python  
- Pandas  
- NumPy  
- Scikit-Learn   
- XGBoost  
- Joblib  
- Streamlit  

---

## 🧹 Step 1 — Data Preprocessing

Run the script to clean the raw dataset:

```
python src/data_processing.py --input data/raw/diabetes.csv --out data/processed/processed.csv
```

This will:
- Clean dataset  
- Replace zero values with NaN  
- Fill missing values  
- Save cleaned file  


---

## 🤖 Step 2 — Train the Machine Learning Model

```
python src/train.py --input data/processed/processed.csv --model models/model.pkl
```

This will:
- Train XGBoost model  
- Print AUROC score  
- Save model in `models/model.pkl`  

---

## 📊 Step 3 — Evaluate the Model

```
python src/evaluate.py --input data/processed/processed.csv --model models/model.pkl
```

Outputs:
- ROC-AUC  
- Confusion Matrix  
- Classification Report  
- Precision-Recall Curve (PR-AUC)  

---

## 🌐 Step 4 — Run Streamlit Web App

```
streamlit run app.py
```

The app will open at:

👉 http://localhost:8501

Enter patient details to get **Diabetes Risk Score**.

---

## 🖼️ Streamlit App Preview

<img width="1920" height="1080" alt="Screenshot (106)" src="https://github.com/user-attachments/assets/c81d4fc5-c2ac-415b-a416-a38e5b55fd3e" />
<img width="1920" height="1080" alt="Screenshot (107)" src="https://github.com/user-attachments/assets/e9453596-bfcf-46df-abc4-db7568e0030d" />


---
## 🚀 Live Demo

👉 **Open the deployed app here:**  
🔗 https://ankit-diabetes-app.streamlit.app




## 👨‍💻 Author

**Developed by _Coding With Ankit_**  
Machine Learning | Data Science | Python Developer  

---

## ⭐ Support the Project

If you like this project, give it a **⭐ star** on GitHub!






















