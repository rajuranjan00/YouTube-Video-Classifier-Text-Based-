from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import os
import pickle
import requests

app = Flask(__name__)

model = tf.keras.models.load_model('saved_models/model.h5')


with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

category_mapping = {
    0: 'Education',
    1: 'Entertainment',
}

def predict_output(video_link):
    try:
        video_title = extract_video_title(video_link)

        if video_title:
            processed_title = preprocess_title(video_title)

            prediction_values = model.predict(processed_title)
            predicted_category = category_mapping.get(prediction_values.argmax(), 'Unknown')
            return predicted_category, prediction_values
        else:
            return "Failed to extract video title", None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None


def extract_video_title(video_link):
    try:

        api_key = 'YOUTUBE_DATA_API_KEY'
        video_id = video_link.split('=')[-1]
        api_url = f'https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet'
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            video_title = data['items'][0]['snippet']['title']
            return video_title
        else:
            return None
    except Exception as e:
        print(f"An error occurred during title extraction: {e}")
        return None


def preprocess_title(title):

    title_sequence = tokenizer.texts_to_sequences([title])
    title_sequence = pad_sequences(title_sequence, maxlen=50)

    return title_sequence


video_link_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    video_title = None
    if request.method == 'POST':
        video_link = request.form['video_link']

        # Append the new video link to the history list
        video_link_history.append(video_link)

        predicted_category, prediction = predict_output(video_link)

        if predicted_category is None:
            return "Failed to predict the category"

        # Extract the title of the YouTube video
        video_title = extract_video_title(video_link)

        return render_template('index.html', prediction=predicted_category, raw_prediction=prediction, video_title=video_title, video_link_history=video_link_history)

    return render_template('index.html', prediction=None, raw_prediction=None, video_title=video_title, video_link_history=video_link_history)

if __name__ == '__main__':
    app.run(debug=True)