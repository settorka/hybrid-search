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
docker compose down
```

Use larger ingest counts only after the 10k smoke path is healthy.
