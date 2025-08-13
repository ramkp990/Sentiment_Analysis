import json
import logging
import uuid
import requests
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from rake_nltk import Rake

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global dictionary to store the extracted data
extracted_data = {}

def extract_keywords(data):
    """
    Extract keywords from the provided JSON data using RAKE.
    """
    logging.info(f"Extracting keywords from data: {data}")
    r = Rake()  # Initialize RAKE
    body_text = data.get('body', '')
    r.extract_keywords_from_text(body_text)
    keywords = r.get_ranked_phrases()
    utc_created = data.get('utc_created', '')
    return keywords, utc_created

def get_sentiment(text):
    """
    Call the sentiment analysis service and return the sentiment.
    """
    response = requests.post('http://localhost:5000/predict', json={'text': text})
    sentiment = response.json()['sentiment']
    return sentiment

def send_data_to_dashboard(data_id, data):
    """
    Send the extracted data to the Flask dashboard.
    """
    url = 'http://localhost:5001/update_data'
    payload = {
        'id': data_id,
        'data': data
    }
    requests.post(url, json=payload)

def process_json_data(json_string):
    """
    Process each JSON string, extract keywords, predict sentiment,
    and store the results in the global dictionary.
    """
    data = json.loads(json_string)
    logging.info(f"Processing data: {data}")

    keywords, utc_created = extract_keywords(data)
    sentiment = get_sentiment(data.get('body', ''))

    data_id = data.get('id', str(uuid.uuid4()))  # Use UUID if 'id' is not present
    data_to_send = {
        'keywords': keywords,
        'utc_created': utc_created,
        'sentiment': sentiment
    }
    
    extracted_data[data_id] = data_to_send
    logging.info(f"Extracted data: {extracted_data[data_id]}")

    # Send the extracted data to the dashboard
    send_data_to_dashboard(data_id, data_to_send)

def main():
    env = StreamExecutionEnvironment.get_execution_environment()

    kafka_consumer_properties = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'flink-group'
    }

    kafka_producer_properties = {
        'bootstrap.servers': 'localhost:9092'
    }

    kafka_consumer = FlinkKafkaConsumer(
        topics='json-data',
        deserialization_schema=SimpleStringSchema(),
        properties=kafka_consumer_properties
    )

    kafka_producer = FlinkKafkaProducer(
        topic='extracted-comments',
        serialization_schema=SimpleStringSchema(),
        producer_config=kafka_producer_properties
    )

    stream = env.add_source(kafka_consumer)

    stream.map(process_json_data).name("Process JSON Data")

    # Uncomment if you need to send data back to Kafka
    # stream.add_sink(kafka_producer)

    logging.info("Starting Flink job")

    env.execute("Keyword and Sentiment Extraction Job")

if __name__ == '__main__':
    main()
