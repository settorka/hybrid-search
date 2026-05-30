# v1 Local Deployment

Local-only Compose stack for bounded performance checks.

## Services

- `api`: FastAPI v1 service.
- `redis`: real cache dependency.
- `elasticsearch`: real keyword/vector search dependency.

No GCP, Terraform, Kubernetes, or Jaeger.

## Commands

```sh
cd v1/deployment
docker compose up -d --build
docker compose exec api uv run python scripts/ingest_faker.py --count 10000 --reset
curl http://127.0.0.1:8001/health/ready
curl -X POST http://127.0.0.1:8001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"technology search databases vectors","top_k":10}'
docker compose exec api uv run python scripts/smoke_performance.py --requests 25
docker compose down -v --remove-orphans
```

Use larger ingest counts only after the 10k smoke path is healthy.

## Defensive Server Bounds (Local Only)

These are required to make body-size limits and deadlines meaningful under slow clients.

Compose uses `uvicorn` directly. Keep these bounds when running the API:

- Set a reverse proxy in front if exposing beyond localhost.
- Enforce connection/read timeouts at the proxy/server layer (slowloris defense).
- Keep `HYBRID_SEARCH_MAX_BODY_SIZE_BYTES` low and enforce request deadlines.

## Reproducible 1M Run (Local Only)

This is the v1 performance claim path.

```sh
cd v1/deployment
docker compose up -d --build

# Ingest 1,000,000 magazines (multi-paragraph content). Disables refresh during ingest for throughput.
docker compose exec api uv run python scripts/ingest_faker.py \
  --count 1000000 \
  --batch-size 2000 \
  --paragraphs 8 \
  --seed 42 \
  --reset \
  --refresh-interval -1

curl http://127.0.0.1:8001/health/ready

# Benchmark with explicit throttle to avoid 429s (admission limits are part of v1 bounds).
docker compose exec api uv run python scripts/smoke_performance.py \
  --requests 50 \
  --unique \
  --timeout 10 \
  --top-k 10 \
  --sleep-ms 200

# Tear down and delete all persisted data.
docker compose down -v --remove-orphans
```
