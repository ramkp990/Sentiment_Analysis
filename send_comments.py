import json
import random
from kafka import KafkaProducer
from datetime import datetime, timezone

# Kafka configuration
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'json-data'

# Path to the local JSON file
JSON_FILE_PATH = 'avengers_hulk.json'

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BROKER],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def convert_timestamp_to_date(timestamp):
    """
    Convert UTC timestamp to human-readable date format.
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d')

def send_random_comments_to_kafka(file_path, num_comments):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            for _ in range(num_comments):
                # Select a random comment
                random_comment = random.choice(lines)
                data = json.loads(random_comment.strip())  # Assuming each line is a JSON object
                body_data = {
                    "body": data.get("body"),
                    "utc_created": convert_timestamp_to_date(data.get("created_utc")) if data.get("created_utc") else None  # Extract only the 'body' and 'utc_created' fields
                }
                producer.send(KAFKA_TOPIC, value=body_data)
                producer.flush()  # Ensure the message is sent immediately
                print(f"Sent random comment to Kafka: {body_data}")

            print(f"{num_comments} random comments sent to Kafka successfully.")

    except Exception as e:
        print(f"Error sending comments to Kafka: {e}")
    finally:
        producer.close()

if __name__ == '__main__':
    send_random_comments_to_kafka(JSON_FILE_PATH, 100)