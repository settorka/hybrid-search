# Adapted from: 
# https://github.com/elastic/elasticsearch-labs/blob/main/example-apps/search-tutorial/v2/search-tutorial/search.py

import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import List, Dict, Any

class ElasticsearchORM:
    def __init__(self):
        ES_HOST = os.getenv("ES_HOST", "elasticsearch")
        ES_PORT = int(os.getenv("ES_PORT", 9200))
        ES_SCHEME = os.getenv("ES_SCHEME", "http")
        self.es = Elasticsearch([f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"])

    def create_index(self, index_name: str, mappings: Dict[str, Any]):
        """Create an index with the given mappings."""
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body={
                                   "mappings": mappings})

    def index_document(self, index_name: str, doc_id: str, document: Dict[str, Any]):
        """Index a single document."""
        self.es.index(index=index_name, id=doc_id, body=document)

    def bulk_index(self, index_name: str, documents: List[Dict[str, Any]]):
        """Bulk index multiple documents."""
        actions = [
            {
                "_index": index_name,
                "_id": doc["id"],
                "_source": doc
            }
            for doc in documents
        ]
        return bulk(self.es, actions)

    def search(self, index_name: str, query: Dict[str, Any], size: int = 10):
        """Perform a search query."""
        return self.es.search(index=index_name, body=query, size=size)

    def get_document(self, index_name: str, doc_id: str):
        """Retrieve a document by its ID."""
        return self.es.get(index=index_name, id=doc_id)

    def update_document(self, index_name: str, doc_id: str, update_body: Dict[str, Any]):
        """Update a document."""
        self.es.update(index=index_name, id=doc_id, body={"doc": update_body})

    def delete_document(self, index_name: str, doc_id: str):
        """Delete a document."""
        self.es.delete(index=index_name, id=doc_id)

    def delete_index(self, index_name: str):
        """Delete an entire index."""
        self.es.indices.delete(index=index_name, ignore=[400, 404])
