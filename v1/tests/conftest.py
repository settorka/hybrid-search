import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import Settings


@pytest.fixture
def settings() -> Settings:
    """Return tight test settings."""

    return Settings()


@pytest.fixture
def client(settings: Settings) -> TestClient:
    """Return a test client."""

    return TestClient(create_app(settings))
