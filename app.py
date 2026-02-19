import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = pickle.load(open('spam_model.pkl',"rb"))
vectorizer = pickle.load(open('vectorizer.pkl',"rb"))

st.title("📩 Spam Message Detector")
st.write("Enter a message to check whether it is Spam or Ham")

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]','',text)
    words = text.split()
    new_words = []
    for w in words:
        if w not in stop_words:
            new_words.append(w)
    words = new_words
    return " ".join(words)

message = st.text_area("Enter message")
if st.button("Detect"):
    if message.strip()=="":
        st.warning("Please enter a message")
    else:
        cleaned = clean(message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        if prediction == 1:
            st.error(f"🚨 SPAM detected ({prob[1]*100:.2f}% confidence)")
        else:
            st.success(f"✅ NOT SPAM ({prob[0]*100:.2f}% confidence)")
