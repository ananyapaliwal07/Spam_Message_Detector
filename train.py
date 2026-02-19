import pandas as pd
import numpy as np
import nltk
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

df = pd.read_csv("spam.csv",encoding="latin-1")
df = df[['v1','v2']]
df.columns = ['labels','message']
df['labels'] = df['labels'].map({'ham':0,'spam':1})

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
df['message'] = df['message'].apply(clean)
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['message'])
y = df['labels']
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
model = MultinomialNB()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("accuracy: ",accuracy_score(y_test,pred))
pickle.dump(model,open("spam_model.pkl","wb"))
pickle.dump(vectorizer,open("vectorizer.pkl","wb"))
