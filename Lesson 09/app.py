import re
import pickle
import numpy as np

from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def load_variables():
    global model, tokenizer
    model = load_model('trained_model.h5')
    model._make_predict_function()  # https://github.com/keras-team/keras/issues/6462
    with open('trained_tokenizer.pkl',  'rb') as f:
        tokenizer = pickle.load(f)


def do_preprocessing(reviews):
    processed_reviews = []
    for review in reviews:
        review = review.lower()
        processed_reviews.append(re.sub('[^a-zA-z0-9\s]', '', review))
    processed_reviews = tokenizer.texts_to_sequences(np.array(processed_reviews))
    processed_reviews = pad_sequences(processed_reviews, maxlen=250)
    return processed_reviews


app = Flask(__name__)


@app.route('/')
def home_routine():
    return 'Hello'


@app.route('/prediction', methods=['POST'])
def get_prediction():
  # get incoming text
  # run the model
    if request.method == 'POST':
        data = request.get_json()
    data = do_preprocessing(data)
    predicted_sentiment_prob = model.predict(data)
    predicted_sentiment = np.argmax(predicted_sentiment_prob, axis=-1)
    return str(predicted_sentiment)


if __name__ == '__main__':
  # load model
  load_variables()
  app.run(debug=True)
