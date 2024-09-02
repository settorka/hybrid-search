# Hybrid Search API for Magazine Articles

## Introduction
This project implements an efficient hybrid search system designed for a large collection of magazine articles. It integrates traditional keyword-based search with advanced vector-based search methodologies to provide highly relevant results. The system is capable of managing and querying approximately 1 million records, ensuring high scalability and responsiveness.

## Project Goal
Develop an efficient hybrid search system for a large collection of magazine articles by integrating traditional keyword-based search with advanced vector-based search methodologies. This system must handle and query approximately 1 million records efficiently, ensuring high scalability and responsiveness.
Achieved
- ~750 ms on first unique query on average, ~ 10 ms on next, with further reductions in response time using lazy loading in Redis
- Handled 750,000 records efficiently on a single node cluster, budget, computational and time constraints.
- Implemented innovative hybrid search techniques to handle different types of queries relating to magazine articles


## High Level Overview
### Data Model
- Magazine Articles Characteristics
    - Content Length: Typically 500 to 1000 words.
    - Language: Engaging and conversational with varied vocabulary.
    - Structure: A mix of short and long sentences, with attention-grabbing headlines.

- Indices:
    - magazine_info: Contains metadata about each magazine article.
    - magazine_content: Contains the full content of each article and its vector representation.

- Fields:
    - magazine_info: id, title, author, publication_date, category
    - magazine_content: id, magazine_id (foreign key), content, vector_representation # sentence_vector and chunk_vector available for production purposes on more powerful hardware



### Search Methodology
- Keyword-Based Search
    - Technology: Elasticsearch
    - Features: Stemming, tokenization, and lowercasing.
    - Purpose: Achieves high precision in matching keywords within diverse content structures.

- Vector Search
    - Technology: SentenceTransformer for dense vector embeddings
    - Function: Converts textual information into dense vectors that capture semantic meanings and contextual relationships.
    - Method: Utilizes cosine similarity to measure proximity between query vectors and candidate vectors.
    - Purpose: Identifies related content based on vector similarity, capturing nuanced relationships within data points.
    - Advantage: Goes beyond surface-level keyword matches, allowing more precise navigation through vast datasets.
    - Limitation: While effective at finding similar content, it doesn't inherently understand user intent or provide deep semantic understanding unlike semantic search with LLM for RAG or ELSER for sparse encodings

- Hybrid Search
    - Integration: Combines results from both keyword and vector searches.
    - Features:
        - Dynamic Weight Adjustment: Adjusts search weights based on the length of the query to balance relevance and precision.
        - Boosting Strategies: Applies boosts for exact matches in titles and authors (keyword search) and for term matching within content (vector search).
        - Score Aggregation: Merges and aggregates scores from both search methods to produce a unified relevance ranking.

- Technology Stack
    - Backend: Python 3.9 with FastAPI
    - Database: Elasticsearch
    - Caching: Redis
    - Vector Embeddings: SentenceTransformer

- API Endpoint
    - Endpoint: /search
    - Function: Performs hybrid search by combining keyword and vector search results. Implemented using FastAPI for efficient and scalable API handling.

## Deliverables
- Source Code:
    - Location: main.py app folder
    - Dockerfile and docker-compose.yml

- ORM Scripts:
    - elasticsearch_orm.py: Elasticsearch ORM implementation
    - create_magazine_indices.py: Script for creating Elasticsearch indices
    - insert_magazine_data.py: Script for inserting magazine data into Elasticsearch

- Technical Documentation:
    - API-Docs.md: Detailed API documentation
    - Test_Cases.pdf: Comprehensive test scenarios and results
- Performance Report:
    - Performance-Report.md: Detailed analysis of performance considerations and optimizations



### Scalability Considerations
- Prepared for chunk and sentence-level indexing in high-performance setups
- Ready for integration with GPU-accelerated Elasticsearch configurations
- Designed to work with multi-node Elasticsearch clusters for horizontal scaling using Elasticsearch shard and replica commands in cluster setup