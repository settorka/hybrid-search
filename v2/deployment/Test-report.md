# v2 Load / Soak Test Report

## Run: 2026-06-14 22:29:03 BST

### Load Result

- Target: `10,000 req/min` for `5 min`
- Sent: `50,000`
- Actual rate: `9,999 req/min`
- Statuses:
  - `200`: `214`
  - `429`: `49,786`
- Errors:
  - `rate_limited`: `49,700`
  - `overloaded`: `86`
- Client-observed latency across all responses:
  - p50: `3.604ms`
  - p95: `6.063ms`
  - p99: `12.561ms`
  - max: `603.024ms`

### Artifacts

- Load report: `/private/tmp/hybrid_search_v2_load_10000rpm_5m.json`
- API log: `/private/tmp/hybrid_search_v2_api_load_10000rpm_5m.log`
- Metrics snapshot: `/private/tmp/hybrid_search_v2_metrics_after_10000rpm_5m.prom`

### Deduction

The API handled overload correctly: it stayed ready after the run, emitted deterministic `429`s, and logs had no `ERROR`, `Traceback`, `Exception`, or `CRITICAL`.

This was not a throughput capacity test. Current v2 config caps admission at `60 req/min` per client and `120 req/min` global, so `10,000 req/min` mostly validates rejection behavior.

Important finding: successful requests were only `214` over 5 minutes, lower than the rough `60/min * 5 = 300` per-client expectation. That suggests a real admission accounting issue: global rate tokens are consumed before client-rate rejection, so rejected client requests can burn global capacity. Fix this before using these numbers as acceptance evidence.

### Post-Run Checks

- `ruff`: passed
- `pytest`: `27 passed`
- Stack health: ready
- Containers: API up, Redis healthy, Elasticsearch healthy

## Runbook

Use this runbook to test bounded overload behavior and collect reproducible artifacts.

Important: default v2 admission limits are intentionally low:

- `HYBRID_SEARCH_PER_CLIENT_RATE_PER_MINUTE=60`
- `HYBRID_SEARCH_GLOBAL_RATE_PER_MINUTE=120`

A `10,000 req/min` run with these defaults is an overload test. Most responses should be `429`; it is not a successful-throughput capacity claim.

## 1. Start Dependencies

```sh
cd v2/deployment
docker compose up -d --build
docker compose ps
```

## 2. Seed Elasticsearch

Use `10,000` records for a quick dependency-backed load run:

```sh
docker compose exec api uv run python scripts/ingest_faker.py \
  --count 10000 \
  --batch-size 1000 \
  --reset
```

For acceptance evidence, use the 1M ingest path in `deployment/README.md` instead.

## 3. Check Readiness

```sh
curl -sS http://127.0.0.1:8002/health/ready
curl -sS http://127.0.0.1:8002/rollout/status
curl -sS http://127.0.0.1:8002/metrics
```

Expected:

- `/health/ready` returns `status=ready`.
- `/rollout/status` returns `status=pass`.
- `/metrics` exposes request, cache, dependency, rollout, and state-machine metrics.

## 4. Run 10k/min For 5 Minutes

From the `v2/` directory:

```sh
cd ..
uv run python scripts/load_performance.py \
  --base-url http://127.0.0.1:8002 \
  --rate-per-minute 10000 \
  --duration-seconds 300 \
  --output /private/tmp/hybrid_search_v2_load_10000rpm_5m.json
```

The load client schedules `50,000` requests over `300` seconds and writes a JSON report with:

- sent request count
- actual request rate
- status counts
- error counts
- latency min/p50/p95/p99/max

## 5. Collect Logs And Metrics

```sh
curl -sS -o /private/tmp/hybrid_search_v2_metrics_after_10000rpm_5m.prom \
  http://127.0.0.1:8002/metrics

docker compose logs --no-color api \
  > /private/tmp/hybrid_search_v2_api_load_10000rpm_5m.log

wc -l \
  /private/tmp/hybrid_search_v2_load_10000rpm_5m.json \
  /private/tmp/hybrid_search_v2_metrics_after_10000rpm_5m.prom \
  /private/tmp/hybrid_search_v2_api_load_10000rpm_5m.log

rg "ERROR|Traceback|Exception|CRITICAL" \
  /private/tmp/hybrid_search_v2_api_load_10000rpm_5m.log
```

No matches from the final `rg` command is the expected result.

## 6. Interpret Results

For the default overload run:

- `hybrid_search_requests_total{status="rate_limited"}` should dominate.
- `hybrid_search_requests_total{status="ok"}` should stay near configured admission capacity.
- `hybrid_search_inflight_requests` should return to `0`.
- `/health/ready` should still return ready after the run.
- API logs should show bounded `429 Too Many Requests`, not stack traces.

If the goal is successful throughput capacity instead of overload behavior, raise these settings deliberately and rerun:

- `HYBRID_SEARCH_PER_CLIENT_RATE_PER_MINUTE`
- `HYBRID_SEARCH_GLOBAL_RATE_PER_MINUTE`
- `HYBRID_SEARCH_MAX_CONCURRENT_REQUESTS`
- Elasticsearch heap/replica/container sizing

Do not compare a raised-limit capacity run against the default overload run; they answer different questions.

## 7. Tear Down

```sh
docker compose down -v --remove-orphans
```
