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

The Terraform entrypoint in `cloud/gcp/` creates the EU single-region GCP runtime envelope:

- Artifact Registry for the API image.
- Cloud Run service for the API.
- Public Cloud Run invoker binding when `allow_unauthenticated=true`.
- Private VPC and subnet.
- Cloud Router/NAT for private Elasticsearch node startup egress.
- VPC connector for Cloud Run private egress.
- Memorystore Redis.
- Terraform-managed GCE Elasticsearch nodes.
- Private firewall rules for Elasticsearch HTTP and transport traffic.
- Budget alert policy for the `<= 100 GBP/month` constraint.
- Required service APIs.

The default deployment profile is calibrated for an EU live beta target:

- `50 req/s` sustained baseline.
- `100 req/s` short peak.
- `150-200 req/s` overload validation.
- Cloud Run max instances: `10`.
- Cloud Run concurrency: `16`.
- Redis memory: `2 GB`.
- Elasticsearch: `2` nodes by default, `e2-standard-4`, `150 GB` disk each.
- Cache TTL: `600s`.
- Cache max entries: `50,000`.
- Global process-local rate limit: `6,000/min`.

This is still not a full production claim. Go-live requires measured gates: indexed corpus, p50/p95/p99 per scenario, rollback drill, cutover drill, ES/Redis/Cloud Run resource metrics, and bounded telemetry volume.

```sh
cd v2/deployment/cloud/gcp
terraform init \
  -backend-config='bucket=YOUR_TERRAFORM_STATE_BUCKET' \
  -backend-config='prefix=hybrid-search/v2/europe-west2'
terraform plan \
  -var='project_id=YOUR_PROJECT' \
  -var='region=europe-west2' \
  -var='billing_account_id=YOUR_BILLING_ACCOUNT' \
  -var='alert_email=you@example.com' \
  -var='api_image=europe-west2-docker.pkg.dev/YOUR_PROJECT/hybrid-search-v2/api:TAG'
```

For GitHub Actions deployments, Terraform state is stored in GCS. Create the state bucket before the first workflow run and configure these repository values:

Variables:

- `GCP_PROJECT_ID`
- `ALERT_EMAIL`

Secrets:

- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`
- `GCP_BILLING_ACCOUNT_ID`
- `GCP_TERRAFORM_STATE_BUCKET`

The deploy workflow is manual only. It bootstraps Artifact Registry, builds and pushes the API image, runs `terraform plan`, and applies only when the `apply` input is set to `true`.

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
