from flask import Flask, render_template, request
# import pickle
# from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# import joblib

# app = Flask(__name__)
# model = joblib.load('model.pkl')
# cv = joblib.load('vectorizer.pkl')


# @app.route("/")
# def Home():
#     return render_template("index.html")


# @app.route("/predict", methods=["POST"])
# def predict():

#     text_input = request.form['input']
#     df = cv.transform([text_input]).toarray()

#     # Pass the text input to your pkl model to get a prediction
#     y_pred = model.predict(df)
#     if y_pred[0] == 0:
#         prediction = 'Not hate speech'
#     else:
#         prediction = 'Hate speech'
# #         # Return the prediction as a string to the user
#     return f"The predicted label for is {prediction}."


# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.corpus import stopwords
import string
import json


app = Flask(__name__)

# download the NLTK stopwords
nltk.download('stopwords')
stopword = set(stopwords.words("english"))

# load the data and preprocess it
df = pd.read_csv("twitter_data.csv")
df["tweet"].fillna("", inplace=True)


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    return text


df["tweet"] = df["tweet"].apply(clean)

# train the model
x = np.array(df["tweet"])
y = np.array(df["class"])
cv = CountVectorizer()
x = cv.fit_transform(x)
clf = DecisionTreeClassifier()
clf.fit(x, y)

# define the Flask routes


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['input']
    test_data = data[0]
    test_data = cv.transform([test_data]).toarray()
    prediction = 0
    prediction = clf.predict(test_data)[0]
    print(prediction)
    if prediction == 0:
        result = 'Hate Speech Detected'
    elif prediction == 1:
        result = 'Offensive Language Detected'
    else:
        result = 'No hate or offensive speech'
    return render_template("index.html", prediction_result=result)


if __name__ == '__main__':
    app.run(debug=True)
