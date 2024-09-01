import os
import json
import csv
from elasticsearch_orm import ElasticsearchORM
from create_magazine_indices import MAGAZINE_INFO_INDEX, MAGAZINE_CONTENT_INDEX
from sentence_transformers import SentenceTransformer

# Initialize ElasticsearchORM and SentenceTransformer
es_orm = ElasticsearchORM()
model = SentenceTransformer("all-MiniLM-L6-v2")


def read_mock_data(file_path):
    """
    Reads mock data from either a CSV or JSON file.
    :param file_path: Path to the mock data file
    :return: List of dictionaries containing the mock data
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return list(reader)
    elif file_extension == '.json':
        with open(file_path, mode='r', encoding='utf-8') as file:
            return json.load(file)
    else:
        raise ValueError("Unsupported file type. Please use CSV or JSON.")


def insert_mock_data(data):
    """
    Inserts mock data into Elasticsearch.
    :param data: List of dictionaries containing the mock data
    """
    magazine_info_docs = []
    magazine_content_docs = []

    for idx, record in enumerate(data, start=1):
        magazine_info = {
            "id": idx,
            "title": record["title"],
            "author": record["author"],
            "publication_date": record["publication_date"],
            "content": record["content"],
            "category": record["category"]
        }

        # Generate vector embedding for the content
        content_vector = model.encode(record["content"]).tolist()

        magazine_content = {
            "id": idx,
            "magazine_id": idx,
            "title": record["title"],
            "author": record["author"],
            "content": record["content"],
            "summary": record.get("summary", ""),  # Add summary if available
            "category": record["category"],
            "updated_at": record.get("updated_at", record["publication_date"]),
            "content_vector": content_vector
        }

        magazine_info_docs.append(magazine_info)
        magazine_content_docs.append(magazine_content)

        # Bulk index every 1000 documents
        if idx % 1000 == 0:
            es_orm.bulk_index(MAGAZINE_INFO_INDEX, magazine_info_docs)
            es_orm.bulk_index(MAGAZINE_CONTENT_INDEX, magazine_content_docs)
            magazine_info_docs = []
            magazine_content_docs = []
            print(f"Inserted {idx} documents")

    # Insert any remaining documents
    if magazine_info_docs:
        es_orm.bulk_index(MAGAZINE_INFO_INDEX, magazine_info_docs)
        es_orm.bulk_index(MAGAZINE_CONTENT_INDEX, magazine_content_docs)

    print(f"Inserted all {len(data)} documents")


if __name__ == "__main__":
    file_path = input(
        "Enter the path to the mock data file (CSV or JSON): ").strip()
    try:
        mock_data = read_mock_data(file_path)
        insert_mock_data(mock_data)
        print("Mock data insertion completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
