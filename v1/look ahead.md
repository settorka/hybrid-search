# Hybrid Search Version Contract

## Status

- `v0`: take-home prototype.
- `v1`: production-aware bounded system.
- `v2`: production-grade cloud system.
- `v3`: business-mature search platform.

## v0 Finding

`v0` proves feasibility only. It is not a production-aware system.

## v0 Gaps

### Bounds

- No proven corpus limit.
- No QPS limit.
- No latency SLO.
- No CPU budget.
- No memory budget.
- No Redis budget.
- No Elasticsearch budget.
- No cold-query budget.
- No cached-query budget.
- No cost-per-query model.

### Search

- Uses brute-force vector scoring.
- Does not use indexed kNN.
- Does not bound vector search cost.
- Mixes BM25 and vector scores without calibration.
- Uses heuristic boosts without proof.
- Has no BM25-only baseline.
- Has no vector-only baseline.
- Has no hybrid uplift benchmark.
- Has no golden query set.
- Has no relevance regression test.

### Correctness

- Defines category filtering but does not apply it.
- Defines analyzer settings but does not install them.
- Cache key omits category.
- Cache key omits index version.
- Cache key omits schema version.
- Cache key omits model version.
- Cache invalidation is undefined.
- Reindex safety is undefined.
- Mixed-model safety is undefined.
- Mixed-schema safety is undefined.

### Health

- No readiness endpoint.
- No liveness endpoint.
- No dependency health contract.
- No index validation.
- No mapping validation.
- No vector dimension validation.
- No model readiness check.
- No service state model.
- No degraded state.
- No drain state.

### Control

- No request deadline.
- No Elasticsearch timeout policy.
- No Redis timeout policy.
- No embedding timeout policy.
- No concurrency limit.
- No queue limit.
- No backpressure.
- No load shedding.
- No circuit breaker.
- No bounded fallback.

### Safety

- No query length limit.
- No body size limit.
- No `top_k` policy beyond basic validation.
- No deep pagination policy.
- No cache cardinality control.
- No empty-query policy.
- No malformed-query policy.
- No missing-field policy.
- No missing-vector policy.
- No wrong-vector-dimension policy.

### Security

- No authentication.
- No API keys.
- No tenant isolation.
- No per-client rate limit.
- No global rate limit.
- No abuse detection.
- No cache-busting defense.
- No expensive-query defense.
- No log redaction policy.
- No privacy boundary.

### Observability

- No structured logs.
- No request IDs.
- No trace IDs.
- No metrics endpoint.
- No latency histograms.
- No cold/warm latency split.
- No cache hit ratio metric.
- No dependency latency metrics.
- No zero-result metric.
- No ranking diagnostics.
- No dashboard.
- No alert contract.

### Ingestion

- Mock data is not representative.
- Ingest is not resumable.
- No checkpointing.
- No failed-document path.
- No dead-letter path.
- No post-ingest verification.
- No update lifecycle.
- No delete lifecycle.
- No reindex runbook.
- No refresh policy.

### Testing

- No unit test contract.
- No integration test contract.
- No dependency-failure tests.
- No cache-staleness tests.
- No abuse tests.
- No boundary tests.
- No pass/fail thresholds for load tests.
- No correctness assertions in load tests.
- No cold-path load test.
- No reproducible performance report.

### Deployment

- Local Docker Compose only.
- No production topology.
- No resource requests.
- No resource limits.
- No worker sizing policy.
- No connection pool policy.
- Dependencies are unpinned.
- No lockfile.
- No `uv` workflow.
- Library versions are prototype-era.
- No vulnerability scanning.
- No upgrade policy.

### Operations

- No owner map.
- No runbook.
- No incident path.
- No rollback path.
- No backup plan.
- No restore plan.
- No data retention policy.
- No audit policy.
- No recovery target.
- No human escalation path.

## v1 Contract

### Objective

`v1` shall bound economic and operational risk.

`v1` shall remain simple unless risk requires complexity.

`v1` shall make failure visible, finite, and recoverable.

### Required Bounds

- Declare max corpus size.
- Declare max QPS.
- Declare p95 latency target.
- Declare p99 latency target.
- Declare max request duration.
- Declare CPU budget.
- Declare memory budget.
- Declare Redis budget.
- Declare Elasticsearch budget.
- Declare monthly cost target.
- Declare cold-query capacity.
- Declare cached-query capacity.

### Required Controls

- Enforce query length limit.
- Enforce body size limit.
- Enforce `top_k` limit.
- Enforce pagination limit.
- Enforce request timeout.
- Enforce dependency timeouts.
- Enforce global concurrency limit.
- Enforce per-client rate limit.
- Enforce cache cardinality control.
- Fail fast when limits are exceeded.

### Required Health

- Provide liveness endpoint.
- Provide readiness endpoint.
- Validate Redis connectivity.
- Validate Elasticsearch connectivity.
- Validate index existence.
- Validate mappings.
- Validate vector dimensions.
- Validate model readiness.
- Expose service state.
- Support degraded state.

### Required Observability

- Emit structured logs.
- Emit request IDs.
- Emit error categories.
- Emit latency histograms.
- Emit cold-query latency.
- Emit cached-query latency.
- Emit cache hit ratio.
- Emit Redis latency.
- Emit Elasticsearch latency.
- Emit embedding latency.
- Emit zero-result rate.
- Emit rate-limit count.
- Emit timeout count.

### Required Correctness

- Apply category filtering or remove it.
- Apply analyzer settings or remove the claim.
- Include query parameters in cache keys.
- Include index version in cache keys.
- Include schema version in cache keys.
- Include model version in cache keys.
- Define cache invalidation.
- Define reindex safety.
- Define missing-field behavior.
- Define missing-vector behavior.

### Required Evaluation

- Create a golden query set.
- Measure BM25-only results.
- Measure vector-only results.
- Measure hybrid results.
- Define relevance acceptance threshold.
- Define latency acceptance threshold.
- Define load-test pass/fail threshold.
- Define abuse-test pass/fail threshold.
- Define dependency-failure pass/fail threshold.

### Required Runtime

- Use `uv`.
- Use `pyproject.toml`.
- Use `uv.lock`.
- Pin maintained modern library versions.
- Define supported Python version.
- Remove prototype-era dependency ambiguity.
- Add vulnerability review command.
- Add upgrade review cadence.

### Required Runbooks

- Boot.
- Shutdown.
- Ingest.
- Reindex.
- Cache clear.
- Redis failure.
- Elasticsearch failure.
- Slow dependency.
- Bad deploy.
- Rollback.

### v1 Out Of Scope

- GCP production deployment.
- Terraform.
- Multi-region deployment.
- Full microservice split.
- Advanced autoscaling.
- GDPR compliance.
- SLA commitment.
- Zero-trust architecture.
- Profitability model.
- Fully calibrated ranking.
- On-call sustainability proof.
- Business governance model.

## v2 Contract

### Objective

`v2` shall be production-grade.

`v2` shall run in a realistic GCP deployment.

`v2` shall target a monthly budget of 100 GBP.

`v2` shall simulate v3 operational behavior where possible.

### Required

- Provide Terraform for GCP.
- Define cloud topology.
- Define budget alerts.
- Define cost dashboard.
- Define deployment pipeline.
- Define rollback pipeline.
- Add auth.
- Add stronger abuse controls.
- Add production dashboards.
- Add production alerts.
- Add backup and restore.
- Add indexed vector retrieval or justified equivalent.
- Add load, stress, soak, and failure tests.
- Add SLOs.
- Add error budgets.

### Conditional

- Split services only when it reduces risk.
- Add queues only when they bound failure.
- Add distributed search only when required by measured limits.
- Add workers only when synchronous work violates the envelope.

## v3 Contract

### Objective

`v3` shall be business mature.

`v3` shall include governance, auditability, compliance, and sustainable operation.

### Required

- Define ownership by domain.
- Define audit trail.
- Define data retention.
- Define deletion policy.
- Define GDPR posture.
- Define SLA posture.
- Define incident review process.
- Define on-call model.
- Define business-value metrics.
- Define ranking calibration process.
- Define cost profitability model.
- Define environmental measurement.
- Maintain planned architecture documents.
- Maintain threat model.
- Maintain lifecycle policy.

## Non-Negotiable Rule

No version may claim a property it does not measure.
