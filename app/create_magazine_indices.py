from elasticsearch_orm import ElasticsearchORM
import os

# Ensures environment variables are set
os.environ.setdefault("ES_HOST", "elasticsearch")
os.environ.setdefault("ES_PORT", "9200")
os.environ.setdefault("ES_SCHEME", "http")


# Initializes the predefined ElasticsearchORM
es_orm = ElasticsearchORM()

# Defines the index  analyzers with stemming and lowercase optimization
INDEX_OPTIMIZATIONS = {
    "analyzer": {
        "standard_stem_analyzer": {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "porter_stem"]
        }
    }
}

# Data models
MAGAZINE_INFO_INDEX = "magazine_info"
MAGAZINE_INFO_MAPPINGS = {
    "settings": INDEX_OPTIMIZATIONS,
    "properties": {
        "id": {"type": "integer"},
        "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        "publication_date": {"type": "date"},
        "content": {"type": "text"},
        "category": {"type": "keyword"}
    }
}

MAGAZINE_CONTENT_INDEX = "magazine_content"
MAGAZINE_CONTENT_MAPPINGS = {
    "settings": INDEX_OPTIMIZATIONS,
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
            "dims": 384  # Dimensionality of the vector
        }
    }
}
#production - multinode cluster / GPU
# MAGAZINE_CONTENT_MAPPINGS = {
#     "settings": INDEX_OPTIMIZATIONS,
#     "properties": {
#         "id": {"type": "integer"},
#         "magazine_id": {"type": "integer"},
#         "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
#         "author": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
#         "content": {"type": "text"},
#         "summary": {"type": "text"},
#         "category": {"type": "keyword"},
#         "updated_at": {"type": "date"},
#         "content_vector": {
#             "type": "dense_vector",
#             "dims": 384  # Dimensionality of the vector
#         },
#
#         "chunks": {
#             "type": "nested",
#             "properties": {
#                 "chunk_id": {"type": "integer"},
#                 "chunk_content": {"type": "text"},
#                 "chunk_vector": {
#                     "type": "dense_vector",
#                     "dims": 384
#                 }
#             }
#         },
#         "sentences": {
#             "type": "nested",
#             "properties": {
#                 "sentence_id": {"type": "integer"},
#                 "sentence": {"type": "text"},
#                 "sentence_vector": {
#                     "type": "dense_vector",
#                     "dims": 384
#                 }
#             }
#         }
#     }
# }

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
