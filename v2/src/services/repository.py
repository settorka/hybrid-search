import json
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, ValidationError

from config import Settings
from helpers.text import cosine_similarity, hash_embedding, tokenize
from models import IndexedMagazine, Magazine, MagazineContent, SearchRequest
from services.admission import RequestContext
from services.repository_base import MagazineRepository


class SeedMagazine(BaseModel):
    """Validated seed magazine row."""

    id: int
    title: str
    author: str
    publication_date: str
    category: str
    content: str


class InMemoryMagazineRepository(MagazineRepository):
    """Searchable local repository for v2 development and tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = RLock()
        self.load_error: str | None = None
        self.quarantined_records: list[dict[str, object]] = []
        self.documents = self._load_documents()

    async def all(self) -> tuple[IndexedMagazine, ...]:
        """Return all indexed documents."""

        with self._lock:
            return self.documents

    async def validate(self) -> bool:
        """Validate index and embedding invariants."""

        with self._lock:
            if self.load_error or self.quarantined_records:
                return False
            for document in self.documents:
                vector = document.content.vector_representation
                if len(vector) != self.settings.embedding_dimension:
                    return False
                if document.content.embedding_model_version != self.settings.model_version:
                    return False
            return True

    async def keyword_search(
        self,
        request: SearchRequest,
        context: RequestContext,
    ) -> dict[int, float]:
        """Return bounded keyword candidates."""

        query_terms = tokenize(request.query)
        scores: dict[int, float] = {}
        for index, document in enumerate(await self.all()):
            if index % 100 == 0:
                self._ensure_budget(context)
            if request.category and document.magazine.category != request.category:
                continue
            title_terms = tokenize(document.magazine.title)
            author_terms = tokenize(document.magazine.author)
            content_terms = tokenize(document.content.content)
            score = 0.0
            score += 3.0 * sum(term in title_terms for term in query_terms)
            score += 2.0 * sum(term in author_terms for term in query_terms)
            score += 1.0 * sum(term in content_terms for term in query_terms)
            if score > 0:
                scores[document.magazine.id] = score
        return dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)[
                : self.settings.max_keyword_candidates
            ]
        )

    async def vector_search(
        self,
        request: SearchRequest,
        query_vector: list[float],
        context: RequestContext,
    ) -> dict[int, float]:
        """Return bounded vector candidates."""

        scores: dict[int, float] = {}
        for index, document in enumerate(await self.all()):
            if index % 100 == 0:
                self._ensure_budget(context)
            if request.category and document.magazine.category != request.category:
                continue
            vector = document.content.vector_representation
            if len(vector) != self.settings.embedding_dimension:
                continue
            score = cosine_similarity(query_vector, vector)
            if score > 0:
                scores[document.magazine.id] = score
        return dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)[
                : self.settings.max_vector_candidates
            ]
        )

    async def get_documents(self, magazine_ids: list[int]) -> dict[int, IndexedMagazine]:
        """Return documents by magazine id."""

        ids = set(magazine_ids)
        return {
            document.magazine.id: document
            for document in await self.all()
            if document.magazine.id in ids
        }

    async def close(self) -> None:
        """Release repository resources."""

        return None

    @staticmethod
    def _ensure_budget(context: RequestContext) -> None:
        """Raise timeout when request budget is exhausted."""

        if context.remaining_seconds() <= 0.001:
            raise TimeoutError("request deadline exceeded")

    def _load_documents(self) -> tuple[IndexedMagazine, ...]:
        """Load deterministic seed data."""

        try:
            raw_rows = json.loads(Path(self.settings.seed_data_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.load_error = str(exc)
            return ()
        if not isinstance(raw_rows, list):
            self.load_error = "seed data must be a list"
            return ()
        documents: list[IndexedMagazine] = []
        for raw_row in raw_rows:
            try:
                row = SeedMagazine.model_validate(raw_row)
            except ValidationError as exc:
                self.quarantined_records.append(
                    {"record": raw_row if isinstance(raw_row, dict) else {}, "error": str(exc)}
                )
                continue
            magazine = Magazine(
                id=row.id,
                title=row.title,
                author=row.author,
                publication_date=row.publication_date,
                category=row.category,
            )
            content = MagazineContent(
                id=row.id,
                magazine_id=row.id,
                content=row.content,
                vector_representation=hash_embedding(
                    row.content,
                    self.settings.embedding_dimension,
                ),
                content_version=self.settings.content_version,
                embedding_model_version=self.settings.model_version,
            )
            documents.append(IndexedMagazine(magazine=magazine, content=content))
        return tuple(documents)
