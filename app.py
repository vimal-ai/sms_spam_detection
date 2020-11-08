import  numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)
model = pickle.load(open('/home/vimal/DataScience_projects/sms spam detection/models/model.sav', 'rb'))
pre = pickle.load(open('/home/vimal/DataScience_projects/sms spam detection/models/sc.sav', 'rb'))

def preprocess(input):
    ps = PorterStemmer()
    
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', input)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

    x = pre.transform(corpus).toarray()
    y = model.predict(x)

    if y[0]:
        output = ' spam'
    else:
        output = ' not spam'

    return output

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #for rendering result on html page
    req = request.form

    input = req.get("input")
    #return input

    output = preprocess(input)

    # prediction = model.predict(df)
    # output = round(prediction[0], 2)
    return render_template('index.html', prediction_text = 'This massage is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)