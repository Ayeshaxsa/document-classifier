# 📩 Document Classifier — Spam Detection App

An NLP + Machine Learning project that classifies SMS messages as **Spam** or **Not Spam** using TF-IDF Vectorization and Logistic Regression.

---

# 🚀 Live Demo

🌐 Streamlit App: *(link will be added.)*

---

# 📌 Features

✅ Text preprocessing  
✅ Stopword removal  
✅ TF-IDF vectorization  
✅ Logistic Regression classifier  
✅ Spam prediction system  
✅ Interactive Streamlit web app  
✅ Model saving using Joblib  

---

# 🧠 Machine Learning Workflow

```text
Raw Message
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression Model
     ↓
Spam / Not Spam Prediction
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Dataset handling |
| Scikit-learn | Machine Learning |
| NLTK | Text preprocessing |
| Streamlit | Web application |
| Joblib | Save/load trained model |

---

# 📂 Project Structure

```text
document-classifier/
│
├── app.py
├── train.py
├── predict.py
├── spam.csv
│
├── spam_model.pkl
├── vectorizer.pkl
│
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Ayeshaxsa/document-classifier
cd document-classifier
```

---

## 2️⃣ Install Dependencies

```bash
pip install pandas scikit-learn nltk streamlit joblib
```

---

# 🏋️ Train the Model

```bash
python train.py
```

This generates:

- `spam_model.pkl`
- `vectorizer.pkl`

---

# 🌐 Run Streamlit App

```bash
streamlit run app.py
```

---

# 💡 Example Messages

### 🚨 Spam
```text
Congratulations! You won a free iPhone
```

### ✅ Not Spam
```text
Hey, are we meeting tomorrow?
```

---

# 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 95% |

---

# 📸 Application Preview

*(imgs will be added.)*

---

# 📚 What I Learned

- NLP preprocessing
- TF-IDF Vectorization
- Logistic Regression
- Text classification workflow
- Streamlit deployment
- Saving ML models

---

Made while learning Machine Learning + NLP 🚀