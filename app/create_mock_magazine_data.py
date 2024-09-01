from faker import Faker
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

fake = Faker()

def generate_mock_data_chunk(start_index, end_index):
    """
    Generates a chunk of mock data for magazines.

    :param start_index: Starting index for the chunk
    :param end_index: Ending index for the chunk
    :return: List of dictionaries containing mock data
    """
    data = []
    for _ in range(start_index, end_index):
        record = {
            "title": fake.sentence(nb_words=5),
            "author": fake.name(),
            "publication_date": fake.date(),
            "category": fake.word(),
            "content": fake.paragraph(nb_sentences=500)
        }
        data.append(record)
    return data


def write_to_csv(data_chunk, file_name):
    """
    Writes a chunk of mock data to a CSV file.

    :param data_chunk: List of dictionaries containing mock data
    :param file_name: The name of the CSV file to write to
    """
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data_chunk[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(data_chunk)


def write_to_json(data_chunk, file_name):
    """
    Writes a chunk of mock data to a JSON file.

    :param data_chunk: List of dictionaries containing mock data
    :param file_name: The name of the JSON file to write to
    """
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', encoding='utf-8') as file:
        if not file_exists:
            file.write('[\n')
        else:
            file.seek(file.tell() - 2, os.SEEK_SET)  # Move back 2 bytes to overwrite the last `]` or `,`
            file.write(',\n')
        
        for record in data_chunk:
            json.dump(record, file, indent=4)
            file.write(',\n')
        
        file.write(']\n')


def generate_and_write_data(num_records, file_type, num_workers=4):
    """
    Generates and writes mock data for magazines using parallel processing.

    :param num_records: Number of records to generate
    :param file_type: The output file type, either 'csv' or 'json'
    :param num_workers: Number of parallel processes to use
    """
    chunk_size = num_records // num_workers
    file_name = 'mock_data.' + file_type

    # Clear the file before starting
    with open(file_name, 'w') as f:
        pass

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, num_records, chunk_size):
            start_index = i
            end_index = min(i + chunk_size, num_records)
            futures.append(executor.submit(generate_mock_data_chunk, start_index, end_index))

        for future in as_completed(futures):
            data_chunk = future.result()
            if file_type == 'csv':
                write_to_csv(data_chunk, file_name)
            elif file_type == 'json':
                write_to_json(data_chunk, file_name)
            else:
                print("Unsupported file type. Please choose 'csv' or 'json'.")
                return

    print(f"Mock data written to {file_name}")


if __name__ == "__main__":
    # Example usage
    num_records = int(input("Enter the number of records to generate: "))
    file_type = input("Enter the file type ('csv' or 'json'): ").strip().lower()
    num_workers = int(input("Enter the number of parallel workers: "))

    generate_and_write_data(num_records, file_type, num_workers)
