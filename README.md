# 🍷 Wine Quality Predictor

A Machine Learning web application that predicts the quality of white wine using physicochemical properties.

Built using:
- Python
- Scikit-learn
- Streamlit
- Matplotlib

---

## 🚀 How It Works

The model uses a Random Forest Regressor trained on the UCI White Wine Dataset to predict wine quality (0–10).

The app:
- Takes 11 wine features as input
- Predicts quality score
- Classifies wine as Premium / Normal / Low
- Shows visual charts and smart insights

---

## 📂 Project Structure

wine-quality-predictor/
│
├── app.py
├── model.py
├── charts.py
├── requirements.txt
├── winequality-white.csv
└── README.md

---

## ⚙️ Run Locally

1. Install dependencies:

pip install -r requirements.txt

2. Run the app:

streamlit run app.py

3. Open in browser:

http://localhost:8501

---

## 📊 Dataset

UCI White Wine Quality Dataset  
Contains 4,898 samples with 11 physicochemical properties.

---

## 👨‍💻 Author

Karm Karia  
GitHub: https://github.com/Karm2003