import asyncio

from conftest import make_settings

from models import SearchRequest
from services.cache import VersionedCache
from services.repository import InMemoryMagazineRepository


def test_cache_key_includes_versions_and_filters() -> None:
    """Cache keys include query parameters and versions."""

    first_settings = make_settings(index_version="index-a", model_version="model-a")
    second_settings = make_settings(index_version="index-b", model_version="model-a")
    request = SearchRequest(query="AI systems", top_k=5, offset=0, category="technology")

    first_key = VersionedCache(first_settings).key_for(request)
    second_key = VersionedCache(second_settings).key_for(request)

    assert "category=technology" in first_key
    assert "schema=schema-v2" in first_key
    assert "index=index-a" in first_key
    assert "model=model-a" in first_key
    assert first_key != second_key


def test_repository_validates_active_model_version() -> None:
    """Repository validation catches model/version drift."""

    async def run() -> bool:
        settings = make_settings(model_version="expected-model")
        repository = InMemoryMagazineRepository(settings)
        repository.documents[0].content.embedding_model_version = "other-model"
        return await repository.validate()

    assert asyncio.run(run()) is False


def test_cache_evicts_at_configured_bound() -> None:
    """Cache cardinality is bounded."""

    settings = make_settings(cache_max_entries=1)
    cache = VersionedCache(settings)
    first = SearchRequest(query="first")
    second = SearchRequest(query="second")

    first_key = cache.key_for(first)
    second_key = cache.key_for(second)
    assert asyncio.run(cache.set(first_key, response=None)) is False
    assert asyncio.run(cache.set(second_key, response=None)) is True

    assert list(cache.records.keys()) == [second_key]


def test_invalid_settings_are_rejected() -> None:
    """Invalid env-style settings fail construction."""

    try:
        make_settings(embedding_dimension=0)
    except ValueError as exc:
        assert "embedding_dimension" in str(exc)
    else:
        raise AssertionError("expected invalid settings to fail")


def test_observability_budget_cannot_exceed_monthly_budget() -> None:
    """Cost settings preserve the v2 budget envelope."""

    try:
        make_settings(monthly_budget_gbp=100, observability_budget_gbp=101)
    except ValueError as exc:
        assert "observability budget" in str(exc)
    else:
        raise AssertionError("expected invalid budget settings to fail")


def test_repository_quarantines_bad_seed_record(tmp_path) -> None:
    """Malformed seed records do not crash the repository."""

    seed_file = tmp_path / "bad_seed.json"
    seed_file.write_text('[{"id": 1, "title": "missing required fields"}]', encoding="utf-8")
    repository = InMemoryMagazineRepository(make_settings(seed_data_path=str(seed_file)))

    assert asyncio.run(repository.all()) == ()
    assert repository.quarantined_records
    assert asyncio.run(repository.validate()) is False
