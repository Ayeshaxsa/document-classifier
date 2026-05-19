import re
from nltk.corpus import stopwords
import joblib

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # lower case rem spl char
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    # rem stop words
    words = text.split()
    words = [w for w in words if not w in stop_words]

    return " ".join(words)
    
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

msg = input("Enter Message: ")
msg = clean_text(msg)
msg_vec = vectorizer.transform([msg])

pred = model.predict(msg_vec)

if pred[0] == 1:
    print("Spam")
else:
    print("Not Spam.")