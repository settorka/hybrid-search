from faker import Faker
import csv
import json

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
            "content": fake.paragraph(nb_sentences=500)
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


if __name__ == "__main__":
    # Example usage
    num_records = int(input("Enter the number of records to generate: "))
    file_type = input(
        "Enter the file type ('csv' or 'json'): ").strip().lower()

    generate_mock_data(num_records, file_type)
