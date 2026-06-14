# v2 Deployment

v2 is scoped to a Terraform-managed GCP deployment with a local Compose path for dependency smoke tests.

## Local Compose

Compose runs the API with Redis and Elasticsearch so request bounds, cache degradation, readiness, and 1M ingest can be measured before cloud rollout.

```sh
cd v2/deployment
docker compose up -d --build
docker compose exec api uv run python scripts/ingest_faker.py --count 10000 --reset
docker compose exec api uv run python scripts/index_rollout.py verify --index magazine_content_v2
curl http://127.0.0.1:8002/health/ready
curl http://127.0.0.1:8002/rollout/status
curl -X POST http://127.0.0.1:8002/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"technology search databases vectors","top_k":10}'
docker compose exec api uv run python scripts/smoke_performance.py --base-url http://127.0.0.1:8002 --requests 25
docker compose down -v --remove-orphans
```

## Terraform

The Terraform entrypoint in `cloud/gcp/` creates the minimum v2 GCP runtime envelope:

- Artifact Registry for the API image.
- Cloud Run service for the API.
- Memorystore Redis.
- VPC connector for Cloud Run to reach Redis.
- Budget alert policy for the `<= 100 GBP/month` constraint.
- Required service APIs.

It intentionally does not claim production acceptance by itself. Go-live still requires the measured gates in the v2 README: 1M indexed docs, p50/p95/p99 per scenario, rollback drill, 23:00 cutover drill, and bounded telemetry volume.

```sh
cd v2/deployment/cloud/gcp
terraform init
terraform plan \
  -var='project_id=YOUR_PROJECT' \
  -var='region=europe-west2' \
  -var='alert_email=you@example.com'
```

## Reproducible 1M Run

```sh
cd v2/deployment
docker compose up -d --build
docker compose exec api uv run python scripts/ingest_faker.py \
  --count 1000000 \
  --batch-size 2000 \
  --paragraphs 8 \
  --seed 42 \
  --reset \
  --refresh-interval -1
docker compose exec api uv run python scripts/index_rollout.py verify --index magazine_content_v2
curl http://127.0.0.1:8002/rollout/status
docker compose exec api uv run python scripts/smoke_performance.py \
  --base-url http://127.0.0.1:8002 \
  --requests 50 \
  --unique \
  --timeout 10 \
  --top-k 10 \
  --sleep-ms 200
docker compose down -v --remove-orphans
```
