# Sentiment_Analysis
# Federated Learning for Streaming Sentiment Analysis

An end-to-end sentiment analysis system combining federated learning for privacy-preserving model training, Apache Flink + Kafka for real-time stream processing, and a Flask REST API for live inference — with a dashboard for monitoring predictions.

---

## Overview

This project tackles two problems simultaneously:

1. **Privacy**: Training sentiment models on data distributed across clients without centralizing raw text — using federated learning with LSTM aggregation.
2. **Scale**: Scoring incoming text streams in real time using a Flink pipeline connected to a Kafka topic, with keyword extraction and sentiment prediction happening on each message.

---

## Architecture

```
Kafka Topic
    │
    ▼
Apache Flink Pipeline
    ├── Keyword Extraction (RAKE)
    ├── HTTP call → Flask Sentiment API
    └── Results → Dashboard (Flask UI)

Flask Sentiment API (/predict)
    └── Global LSTM model (aggregated from federated clients)

Federated Training (offline)
    ├── Client 0 → local_model_0
    ├── Client 1 → local_model_1
    ├── Client 2 → local_model_2
    ├── Client 3 → local_model_3
    └── FedAvg aggregation → global_model.h5
```

---

## Components

### Federated Training (`train_federated_sentiment_analysis_model.py`)
- LSTM model (Embedding → SpatialDropout → LSTM → Dense) trained across 4 simulated clients
- Data distributed in chunks; each client trains locally without sharing raw text
- Weights aggregated via FedAvg into a single global model
- Preprocessing: lowercasing, stopword removal, Porter stemming, sequence padding

### Streaming Pipeline (`flink_app.py`)
- Connects to a Kafka topic consuming JSON messages
- Extracts keywords from each message using RAKE
- Calls the Flask `/predict` endpoint for sentiment scoring
- Pushes enriched results to the live dashboard

### REST API (`flask_sentiment.py`)
- Serves the global LSTM model via `/predict`
- Thread-safe with a lock on global model updates
- Supports live model swapping without server restart

### Dashboard (`flask_ui.py`)
- Receives enriched prediction results from Flink
- Displays sentiment trends and keyword summaries in real time

---

## File Structure

```
train_federated_sentiment_analysis_model.py  # Federated LSTM training
flink_app.py                                 # Flink streaming pipeline
flask_sentiment.py                           # Sentiment prediction API
flask_ui.py                                  # Live monitoring dashboard
data_flow.py / data_flow1.py                 # Data ingestion and preprocessing
send_comments.py                             # Kafka producer for test messages
global_model.h5                              # Aggregated global model weights
```

---

## Tech Stack

- **TensorFlow / Keras** — LSTM model definition and training
- **Apache Flink (PyFlink)** — real-time stream processing
- **Apache Kafka** — message queue for incoming text streams
- **Flask** — REST API and monitoring dashboard
- **RAKE-NLTK** — unsupervised keyword extraction
- **NLTK** — text preprocessing (stopwords, stemming)

---

## Setup

```bash
pip install tensorflow flask pyflink kafka-python rake-nltk nltk pandas numpy tqdm

# 1. Train federated model
python train_federated_sentiment_analysis_model.py

# 2. Start the sentiment API
python flask_sentiment.py

# 3. Start the dashboard
python flask_ui.py

# 4. Start the Flink streaming pipeline (requires running Kafka + Flink cluster)
python flink_app.py

# 5. Send test messages
python send_comments.py
```

> **Note:** Flink and Kafka must be running locally or on a cluster. Update broker addresses in `flink_app.py` and `send_comments.py` as needed.
