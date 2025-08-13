import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm
import pickle

nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Process a data chunk
def process_chunk(chunk, tokenizer, max_sequence_length=100):
    chunk['body'] = chunk['body'].apply(preprocess_text)
    sequences = tokenizer.texts_to_sequences(chunk['body'].tolist())
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    labels = chunk['sentiment'].tolist()
    return padded_sequences, labels

# Initialize tokenizer
tokenizer = Tokenizer(num_words=5000)

# Fit tokenizer on entire dataset
chunksize = 1000
file_path = '/home/raman97/sentiment/labeled_data_transformers.json'
total_lines = sum(1 for _ in open(file_path))
num_chunks = total_lines // chunksize + 1

for chunk in tqdm(pd.read_json(file_path, lines=True, chunksize=chunksize), total=num_chunks, desc="Fitting tokenizer"):
    texts = chunk['body'].tolist()
    tokenizer.fit_on_texts(texts)

# Save tokenizer
with open('/home/raman97/sentiment/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Distribute data among clients
def preprocess_and_distribute_data(file_path, tokenizer, num_clients=4, max_sequence_length=100):
    client_data = [[] for _ in range(num_clients)]
    chunksize = 1000

    for chunk in tqdm(pd.read_json(file_path, lines=True, chunksize=chunksize), total=num_chunks, desc="Preprocessing data"):
        padded_sequences, labels = process_chunk(chunk, tokenizer, max_sequence_length)
        for i in range(len(padded_sequences)):
            client_index = i % num_clients
            client_data[client_index].append((padded_sequences[i], labels[i]))

    for i in range(num_clients):
        client_data[i] = (np.array([x[0] for x in client_data[i]]), np.array([x[1] for x in client_data[i]]))
    
    return client_data

client_data = preprocess_and_distribute_data(file_path, tokenizer)

for i, (texts, labels) in enumerate(client_data):
    print(f"Client {i} texts shape: {texts.shape}")
    print(f"Client {i} labels shape: {labels.shape}")
    print(f"Client {i} first text: {texts[0]}")
    print(f"Client {i} first label: {labels[0]}")
    print()

# Define the model architecture
def create_model(input_length):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=100, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))  # Two output neurons
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train local models for each client
def train_local_models(client_data, global_model, epochs=5, batch_size=32):
    input_length = client_data[0][0].shape[1]
    for i, (client_texts, client_labels) in tqdm(enumerate(client_data), total=len(client_data), desc="Training models"):
        global_model.fit(client_texts, client_labels, epochs=epochs, batch_size=batch_size, verbose=2)
    return global_model

# Aggregate weights from all models
def aggregate_models(models):
    new_weights = []
    for weights in zip(*[model.get_weights() for model in models]):
        new_weights.append(np.mean(weights, axis=0))
    return new_weights

# Federated learning process
def federated_learning(client_data, num_rounds=5, epochs_per_round=1, batch_size=32):
    input_length = client_data[0][0].shape[1]
    global_model = create_model(input_length)
    
    for round in range(num_rounds):
        print(f"Federated Learning Round {round+1}")
        local_models = []
        
        for texts, labels in client_data:
            model = create_model(input_length)
            model.set_weights(global_model.get_weights())
            model.fit(texts, labels, epochs=epochs_per_round, batch_size=batch_size, verbose=2)
            local_models.append(model)
        
        # Save local models and print accuracy
        for idx, model in enumerate(local_models):
            model.save(f'local_model_{idx}_round_{round}.h5')
            accuracy = model.evaluate(client_data[idx][0], client_data[idx][1], verbose=0)[1]
            print(f"Local Model {idx} Accuracy: {accuracy:.2f}")
        
        new_weights = aggregate_models(local_models)
        global_model.set_weights(new_weights)
    
    # Save the global model at the end of all rounds
    global_model.save('global_model.h5')
    
    return global_model

global_model = federated_learning(client_data)

# Evaluate the model
def evaluate_model(model, test_texts, test_labels):
    predictions = model.predict(test_texts).argmax(axis=1)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='macro')
    return accuracy, precision

# Load and preprocess test data
def load_and_preprocess_test_data(file_path, tokenizer, max_sequence_length=100):
    chunksize = 1000
    test_texts, test_labels = [], []

    for chunk in tqdm(pd.read_json(file_path, lines=True, chunksize=chunksize), total=num_chunks, desc="Loading test data"):
        chunk['body'] = chunk['body'].apply(preprocess_text)
        sequences = tokenizer.texts_to_sequences(chunk['body'].tolist())
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
        labels = chunk['sentiment'].tolist()
        test_texts.extend(padded_sequences)
        test_labels.extend(labels)
    
    return np.array(test_texts), np.array(test_labels)

padded_test_sequences, test_labels = load_and_preprocess_test_data(file_path, tokenizer)

accuracy, precision = evaluate_model(global_model, padded_test_sequences, test_labels)

print(f"Global Model Accuracy: {accuracy:.2f}")
print(f"Global Model Precision: {precision:.2f}")

# Display predictions
def display_predictions(model, test_texts, test_labels, tokenizer):
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    for i in range(5):
        text = test_texts[i]
        true_label = test_labels[i]
        prediction = model.predict(np.array([text])).argmax(axis=1)[0]

        decoded_text = ' '.join([reverse_word_index.get(word, '?') for word in text if word != 0])
        print(f"Text: {decoded_text}")
        print(f"True Label: {true_label}, Predicted Label: {prediction}\n")

display_predictions(global_model, padded_test_sequences, test_labels, tokenizer)
