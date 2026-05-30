import json
from pathlib import Path

from config import Settings
from helpers.text import hash_embedding
from models import IndexedMagazine, Magazine, MagazineContent


class InMemoryMagazineRepository:
    """Searchable local repository for v1 development and tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.documents = self._load_documents()

    def all(self) -> list[IndexedMagazine]:
        """Return all indexed documents."""

        return self.documents

    def validate(self) -> bool:
        """Validate index and embedding invariants."""

        for document in self.documents:
            vector = document.content.vector_representation
            if len(vector) != self.settings.embedding_dimension:
                return False
            if document.content.embedding_model_version != self.settings.model_version:
                return False
        return True

    def _load_documents(self) -> list[IndexedMagazine]:
        """Load deterministic seed data."""

        rows = json.loads(Path(self.settings.seed_data_path).read_text(encoding="utf-8"))
        documents: list[IndexedMagazine] = []
        for row in rows:
            magazine = Magazine(
                id=row["id"],
                title=row["title"],
                author=row["author"],
                publication_date=row["publication_date"],
                category=row["category"],
            )
            content = MagazineContent(
                id=row["id"],
                magazine_id=row["id"],
                content=row["content"],
                vector_representation=hash_embedding(
                    row["content"],
                    self.settings.embedding_dimension,
                ),
                content_version=self.settings.content_version,
                embedding_model_version=self.settings.model_version,
            )
            documents.append(IndexedMagazine(magazine=magazine, content=content))
        return documents
