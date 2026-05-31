import uvicorn

from app import create_app
from config import get_settings

app = create_app()


def main() -> None:
    """Run the local v1 API."""

    settings = get_settings()
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
