from config import Settings
from models import SearchRequest
from services.cache import VersionedCache
from services.repository import InMemoryMagazineRepository


def test_cache_key_includes_versions_and_filters() -> None:
    """Cache keys include query parameters and versions."""

    first_settings = Settings(index_version="index-a", model_version="model-a")
    second_settings = Settings(index_version="index-b", model_version="model-a")
    request = SearchRequest(query="AI systems", top_k=5, offset=0, category="technology")

    first_key = VersionedCache(first_settings).key_for(request)
    second_key = VersionedCache(second_settings).key_for(request)

    assert "category=technology" in first_key
    assert "schema=schema-v1" in first_key
    assert "index=index-a" in first_key
    assert "model=model-a" in first_key
    assert first_key != second_key


def test_repository_validates_active_model_version() -> None:
    """Repository validation catches model/version drift."""

    settings = Settings(model_version="expected-model")
    repository = InMemoryMagazineRepository(settings)
    repository.documents[0].content.embedding_model_version = "other-model"

    assert repository.validate() is False
