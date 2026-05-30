import os
from faker import Faker
import csv
import json
import random
fake = Faker()


def generate_mock_data(num_records, file_type):
    """
    Generates mock data for magazines and writes it to a CSV or JSON file.

    :param num_records: Number of records to generate
    :param file_type: The output file type, either 'csv' or 'json'
    """
    data = []

    # Generate mock data for the given number of records
    for _ in range(num_records):
        
        record = {
            "title": fake.sentence(nb_words=5),  # Generates a random title
            "author": fake.name(),               # Generates a random author name
            "publication_date": fake.date(),     # Generates a random publication date
            "category": fake.word(),             # Generates a random category
            # Generates random content
            "content": fake.paragraph(nb_sentences=random.randint(100,750)) # change this to handle different lengths
        }
        data.append(record)

    # Write data to the specified file type
    if file_type == 'csv':
        write_to_csv(data)
    elif file_type == 'json':
        write_to_json(data)
    else:
        print("Unsupported file type. Please choose 'csv' or 'json'.")


def write_to_csv(data):
    """
    Writes mock data to a CSV file.

    :param data: List of dictionaries containing mock data
    """
    file_name = 'mock_data.csv'
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"Mock data written to {file_name}")


def write_to_json(data):
    """
    Writes mock data to a JSON file.

    :param data: List of dictionaries containing mock data
    """
    file_name = 'mock_data.json'
    with open(file_name, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    print(f"Mock data written to {file_name}")

def get_file_type():
    """
    Prompt the user for a file type and validate the input. 
    Deletes existing mock data files if present.

    :return: The validated file type ('csv' or 'json')
    """
    while True:
        file_type = input("Enter the file type ('csv' or 'json'): ").strip().lower()
        if file_type in ['csv', 'json']:
            delete_existing_files()
            return file_type
        else:
            print("Unsupported file type. Please choose 'csv' or 'json'.")

def delete_existing_files():
    """
    Deletes existing mock_data.csv and mock_data.json files if they exist
    in the current directory.
    """
    csv_file = 'mock_data.csv'
    json_file = 'mock_data.json'
    
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Deleted existing file: {csv_file}")
    
    if os.path.exists(json_file):
        os.remove(json_file)
        print(f"Deleted existing file: {json_file}")

def get_num_records():
    """
    Prompt the user for the number of records to generate and validate the input.

    :return: The number of records as a positive integer
    """
    while True:
        try:
            num_records = int(input("Enter the number of records to generate: "))
            if num_records > 0:
                return num_records
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
            
            
if __name__ == "__main__":
    num_records = get_num_records()
    file_type = get_file_type()
    generate_mock_data(num_records, file_type)
