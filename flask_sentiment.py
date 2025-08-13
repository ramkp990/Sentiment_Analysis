from flask import Flask, request, jsonify
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
import tensorflow as tf
import nltk
import logging
from threading import Lock
import random
import pickle

# Download stopwords
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lock for global model updates
global_model_lock = Lock()

# Counter for requests
request_counter = 0
counter_lock = Lock()

# Preprocess the text for sentiment analysis
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Load and recompile model
def load_and_recompile_model(path, global_weights=None):
    model = load_model(path, compile=False)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if global_weights is not None:
        model.set_weights(global_weights)
    return model

# Load the global model
global_model = load_and_recompile_model('global_model.h5')
global_weights = global_model.get_weights()

# Load local models with the same weights as the global model
local_models = [
    load_and_recompile_model('local_model_0_round_4.h5', global_weights),
    load_and_recompile_model('local_model_1_round_4.h5', global_weights),
    load_and_recompile_model('local_model_2_round_4.h5', global_weights),
    load_and_recompile_model('local_model_3_round_4.h5', global_weights)
]

# Load the tokenizer from the pickle file
with open('/home/raman97/sentiment/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Predict sentiment using the global model
def predict_sentiment(text, tokenizer, model):
    preprocessed_text = preprocess_text(text)
    logger.info(f"Preprocessed Text: {preprocessed_text}")
    
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    logger.info(f"Tokenized Sequence: {sequence}")
    
    padded_sequence = pad_sequences(sequence, maxlen=100)
    logits = model.predict(padded_sequence)
    logger.info(f"Logits: {logits}")

    probabilities = Activation('softmax')(tf.convert_to_tensor(logits)).numpy()[0]
    logger.info(f"Probabilities: {probabilities}")
    
    sentiment_score = probabilities[1]
    threshold = 0.5
    sentiment = 1 if sentiment_score > threshold else 0
    return sentiment

# Aggregate local models to update the global model
def aggregate_models(local_models):
    logger.info("Starting aggregation of local models to update the global model.")
    model_weights = [model.get_weights() for model in local_models]
    
    new_weights = []
    for weights in zip(*model_weights):
        new_weights.append(np.mean(weights, axis=0))
    
    global_model.set_weights(new_weights)
    global_model.save('/home/raman97/sentiment/global_model.h5')
    logger.info("Global model has been updated and saved.")
    update_local_models_with_global_weights(global_model, local_models)

# Update local models with global weights
def update_local_models_with_global_weights(global_model, local_models):
    global_weights = global_model.get_weights()
    logger.info("Sharing global model weights with local models.")
    for i in range(len(local_models)):
        local_models[i].set_weights(global_weights)
    logger.info("Local models have been updated with the new global weights.")

# Update local model with new data
def update_local_model(local_model, text, label):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    label = np.array([label])
    local_model.fit(padded_sequence, label, epochs=1, verbose=0)
    return local_model

@app.route('/predict', methods=['POST'])
def predict():
    global request_counter
    
    try:
        data = request.json
        logger.info(f"Request data: {data}")

        text = data['text']
        client_id = random.randint(0, len(local_models) - 1)
        logger.info(f"Randomly selected Client ID: {client_id}")
        logger.info(f"Text: {text}")

        sentiment = predict_sentiment(text, tokenizer, global_model)
        sentiment_labels = ["Negative", "Positive"]
        sentiment_label = sentiment_labels[sentiment]

        local_models[client_id] = update_local_model(local_models[client_id], text, sentiment)
        local_models[client_id].save(f'local_model_{client_id}_round_4.h5')

        with counter_lock:
            request_counter += 1
            if request_counter >= 100:
                with global_model_lock:
                    aggregate_models(local_models)
                request_counter = 0

        return jsonify({'sentiment': sentiment_label})
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
