# Hybrid Search

## Problem

Build an API with one endpoint that performs hybrid search over 1 million magazine records.

The search must combine:

- keyword search over magazine title, author, and content
- vector search over magazine content embeddings
- hybrid ranking that returns the most relevant results from both methods

The data model has two logical tables:

- `MagazineInfo`: `id`, `title`, `author`, `publication_date`, `category`, and related metadata
- `MagazineContent`: `id`, `magazine_id`, `content`, `vector_representation`, and related content fields

The original task required:

- Python 3.9 backend
- one hybrid search endpoint
- a database/search backend that supports vector search
- schema or ORM models
- setup and usage documentation
- query examples
- performance considerations for 1 million records
- delivery within a 7-day take-home window

## Refactor Direction

This repository treats the original task as a versioned system-design exercise.

- `v0`: take-home prototype; proves feasibility.
- `v1`: production-aware contract; bounds operational and economic risk.
- `v2`: production-grade cloud deployment; targets GCP with Terraform and an approximate 100 GBP monthly budget.
- `v3`: business-mature system; adds governance, auditability, compliance, and sustainable operations.

## Current Artifacts

- [v0](./v0): original FastAPI, Elasticsearch, Redis, and SentenceTransformer implementation.
- [v1](./v1): bounded production-aware implementation using FastAPI, `uv`, `.env`, local deterministic retrieval, metrics, OTel hooks, and tests.
- [v1 README](./v1/README.md): refactored functional, non-functional, scale, lifecycle, and failure-aware contract.
- [v1 look ahead](./v1/look%20ahead.md): version contract and v0 gap analysis.

## v1 Local Commands

```sh
cd v1
uv sync
uv run pytest
uv run ruff check src tests
uv run uvicorn main:app --app-dir src --host 127.0.0.1 --port 8001
```

## Non-Negotiable Rule

No version may claim a property it does not measure.
