from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5', compile=False)
metadata = pd.read_csv('musicnet_metadata.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    composer = int(request.form['composer'])
    instrument = int(request.form['instrument'])

    input_data = np.array([[composer, instrument]])
    input_data = input_data.reshape((1, 1, 2))

    prediction = model.predict(input_data)
    return jsonify({'duration': float(prediction[0][0])})


if __name__ == '__main__':
    app.run(debug=True)
