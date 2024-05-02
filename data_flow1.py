from confluent_kafka import Producer
import csv
import time
import os
import zstandard as zstd
'''
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
            if file.endswith(".file"):
                print("here")
                with open(file, 'rb') as file1:
                    # Create a Zstd decompressor
                    dctx = zstd.ZstdDecompressor()
                    print("here")
                    # Decompress the file
                    with dctx.stream_reader(file1) as reader:
                        # Read and decode the data
                        data = reader.read()
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
data_folder = r'D:\BigData'
submit_data_to_kafka(data_folder)

# Flush messages to ensure they are sent
producer.flush()

# Close Kafka producer
#producer.close()
'''
file_path = "D:\BigData\RS_2019_04\RS_2019_04"  # Replace "path_to_your_file" with the actual file path

# Open the file
with open(file_path, 'r') as file:
    # Read the first 10 lines of the file
    for i, line in enumerate(file):
        print(line.rstrip()+'\n')  # Print the line (strip trailing newline character)
        if i == 9:  # Stop after printing 10 lines
            break
