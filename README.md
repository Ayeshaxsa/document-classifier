# Document Classifier (Spam Detection)

## Overview
This project is a simple Machine Learning model that classifies text messages as Spam or Not Spam using NLP techniques.

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- NLTK

---

## How it works

1. Text is cleaned (lowercase, remove symbols, stopwords)
2. Text is converted into numbers using TF-IDF
3. Model (Naive Bayes) learns patterns
4. New messages are classified as spam or not spam

---

## How to run

### 1. Install dependencies

- pip install pandas scikit-learn nltk joblib

### 2. Train model

- python train.py
- This will generate:
    - spam_model.pkl
    - vectorizer.pkl

### 3. Test model

- python predict.py
- Then type a message like: Congratulations! You won a free prize

## Example Output

- Spam

## What I learned
- Text preprocessing
- TF-IDF vectorization
- Naive Bayes classification
- Model evaluation workflow
- Saving ML models