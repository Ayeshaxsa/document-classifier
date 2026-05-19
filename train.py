import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import re
from nltk.corpus import stopwords
import joblib

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # lower case rem spl char
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    # rem stop words
    words = text.split()
    words = [w for w in words if not w in stop_words]

    return " ".join(words)
    
data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({
    'ham' : 0,
    'spam': 1
})
data['message'] = data['message'].apply(clean_text)

X = data['message']
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)        # learn and convert-> discover words, assign idx, calculate importance
X_test_tfidf = vectorizer.transform(X_test)                       # only convert for testing.

print("X_train_tfidf: ", X_train_tfidf.shape) 
print("X_test_tfidf: ", X_test_tfidf.shape)


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Model Trained.")


joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")