from confluent_kafka import Producer
import csv
import time
import os

# Configure Kafka connection
bootstrap_servers = 'localhost:9092'
topic = 'test'

# Configure submission speed (delay between submissions in seconds)
submission_speed = 0.1  # Adjust as needed

# Initialize Kafka producer
producer = Producer({'bootstrap.servers': bootstrap_servers})

# Function to read data from CSV files and submit to Kafka
def submit_data_to_kafka(data_folder):
    
    for root, dirs, files in os.walk(data_folder):
        print("here")
        for file in files:
            if file.endswith(".csv"):
                with open(os.path.join(root, file), 'r', newline='') as csvfile:
                    csvreader = csv.reader(csvfile)
                    # Skip header if present
                    next(csvreader, None)
                    for row in csvreader:
                        # Assuming each row contains a comment in the first column
                        comment = row[0]
                        print(row[0])
                        # Submit comment to Kafka
                        producer.produce(topic, value=comment)
                        # Adjust submission speed
                        time.sleep(submission_speed)

# Example usage
data_folder = r'C:\Users\HP\Documents\BigData'
submit_data_to_kafka(data_folder)

# Flush messages to ensure they are sent
producer.flush()

# Close Kafka producer
#producer.close()
