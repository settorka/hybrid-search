# Hybrid Search API

docker-compose up --build -d

docker-compose exec api python create_magazine_indices.py

docker-compose exec api python create_mock_magazine_data.py

docker-compose exec api python insert_magazine_data.py    