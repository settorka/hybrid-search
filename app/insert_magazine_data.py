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

def chunk_content(content: str, chunk_size: int = 1000, overlap: int = 200):
    """Split content into overlapping chunks."""
    words = content.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

async def index_content_chunks(es, magazine_id: int, content: str):
    """Index content chunks with their vector representations."""
    chunks = chunk_content(content)
    actions = []
    for i, chunk in enumerate(chunks):
        doc = {
            'magazine_id': magazine_id,
            'chunk_id': i,
            'content': chunk,
            'content_vector': model.encode(chunk).tolist()
        }
        actions.append({
            '_index': MAGAZINE_CONTENT_INDEX,
            '_source': doc
        })
    await es.bulk(actions)

async def index_sentences(es, magazine_id: int, content: str):
    """Index individual sentences with their vector representations."""
    sentences = content.split('.')
    actions = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if sentence:  # Avoid empty sentences
            doc = {
                'magazine_id': magazine_id,
                'sentence_id': i,
                'sentence': sentence,
                'sentence_vector': model.encode(sentence).tolist()
            }
            actions.append({
                '_index': MAGAZINE_CONTENT_INDEX,
                '_source': doc
            })
    await es.bulk(actions)


def insert_mock_data(data):
    """
    Inserts mock data into Elasticsearch with enhanced indexing.
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
            "category": record["category"],
        }

        # Generate vector embedding for the content
        content_vector = model.encode(record["content"]).tolist()
        
        # indexing mappings;  for  GPU/multinode setup
        
        # Generate chunks
        # chunks = chunk_content(record["content"])
        # chunk_data = []
        # for i, chunk in enumerate(chunks):
        #     chunk_data.append({
        #         "chunk_id": i,
        #         "chunk_content": chunk,
        #         "chunk_vector": model.encode(chunk).tolist()
        #     })

        # # Generate sentences
        # sentences = record["content"].split('.')
        # sentence_data = []
        # for i, sentence in enumerate(sentences):
        #     sentence = sentence.strip()
        #     if sentence:
        #         sentence_data.append({
        #             "sentence_id": i,
        #             "sentence": sentence,
        #             "sentence_vector": model.encode(sentence).tolist()
        #         })

        magazine_content = {
            "id": idx,
            "magazine_id": idx,
            "title": record["title"],
            "author": record["author"],
            "content": record["content"],
            "summary": record.get("summary", ""),
            "category": record["category"],
            "updated_at": record.get("updated_at", record["publication_date"]),
            "content_vector": content_vector,
            # "chunks": chunk_data,
            # "sentences": sentence_data
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

def get_mock_data_file():
    """
    Determine the file to read mock data from by checking for 
    mock_data.csv or mock_data.json in the current directory.
    """
    # Define the possible filenames
    csv_file = "mock_data.csv"
    json_file = "mock_data.json"
    
    # Check if the CSV file exists in the current directory
    if os.path.exists(csv_file):
        return csv_file
    # If CSV does not exist, check for the JSON file
    elif os.path.exists(json_file):
        return json_file
    else:
        # Raise an error if neither file is found
        raise FileNotFoundError("No mock data file found. Ensure 'mock_data.csv' or 'mock_data.json' is present in the current directory.")

if __name__ == "__main__":
    # Automatically get the mock data file
    file_path = get_mock_data_file()
    try:
        mock_data = read_mock_data(file_path)
        insert_mock_data(mock_data)
        print("Mock data insertion completed successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
