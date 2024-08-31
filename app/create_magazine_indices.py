from elasticsearch_orm import ElasticsearchORM

# Initialize ElasticsearchORM
es_orm = ElasticsearchORM()

# Define your data models here
MAGAZINE_INFO_INDEX = "magazine_info"
MAGAZINE_INFO_MAPPINGS = {
    "properties": {
        "id": {"type": "integer"},
        "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "publication_date": {"type": "date"},
        "category": {"type": "keyword"}
    }
}

MAGAZINE_CONTENT_INDEX = "magazine_content"
MAGAZINE_CONTENT_MAPPINGS = {
    "properties": {
        "id": {"type": "integer"},
        "magazine_id": {"type": "integer"},
        "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "content": {"type": "text"},
        "summary": {"type": "text"},
        "category": {"type": "keyword"},
        "updated_at": {"type": "date"},
        "content_vector": {
            "type": "dense_vector",
            "dims": 384  # Dimensionality of the vector, adjust based on your model
        }
    }
}


def create_magazine_info_index():
    """Create the magazine info index with the specified mappings."""
    es_orm.create_index(MAGAZINE_INFO_INDEX, MAGAZINE_INFO_MAPPINGS)


def create_magazine_content_index():
    """Create the magazine content index with the specified mappings."""
    es_orm.create_index(MAGAZINE_CONTENT_INDEX, MAGAZINE_CONTENT_MAPPINGS)


if __name__ == "__main__":
    create_magazine_info_index()
    print(f"Index '{MAGAZINE_INFO_INDEX}' created successfully.")
    create_magazine_content_index()
    print(f"Index '{MAGAZINE_CONTENT_INDEX}' created successfully.")
