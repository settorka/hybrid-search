variable "project_id" {
  description = "GCP project id for the v2 environment."
  type        = string
}

variable "region" {
  description = "GCP region for v2 resources."
  type        = string
  default     = "europe-west2"
}

variable "billing_account_id" {
  description = "Billing account id used for the v2 budget alert."
  type        = string
}

variable "alert_email" {
  description = "Email address for budget alerts."
  type        = string
}

variable "api_image" {
  description = "Container image for the API. Build and push before terraform apply."
  type        = string
}

variable "subnet_cidr" {
  description = "Primary private subnet range for API egress, Redis, and Elasticsearch."
  type        = string
  default     = "10.10.0.0/24"
}

variable "vpc_connector_cidr" {
  description = "Dedicated /28 range for the Cloud Run VPC connector."
  type        = string
  default     = "10.8.0.0/28"
}

variable "vpc_connector_min_instances" {
  description = "Minimum Cloud Run VPC connector instances."
  type        = number
  default     = 2
}

variable "vpc_connector_max_instances" {
  description = "Maximum Cloud Run VPC connector instances."
  type        = number
  default     = 4
}

variable "cloud_run_ingress" {
  description = "Cloud Run ingress mode."
  type        = string
  default     = "INGRESS_TRAFFIC_ALL"
}

variable "allow_unauthenticated" {
  description = "Allow public unauthenticated Cloud Run invocations."
  type        = bool
  default     = true
}

variable "monthly_budget_gbp" {
  description = "Monthly budget ceiling in GBP."
  type        = number
  default     = 100
}

variable "observability_budget_gbp" {
  description = "Monthly observability budget ceiling in GBP."
  type        = number
  default     = 20
}

variable "api_min_instances" {
  description = "Cloud Run minimum instances."
  type        = number
  default     = 1
}

variable "api_max_instances" {
  description = "Cloud Run maximum instances for 50 req/s sustained and 100 req/s peak."
  type        = number
  default     = 10
}

variable "api_concurrency" {
  description = "Cloud Run request concurrency per instance."
  type        = number
  default     = 16
}

variable "api_cpu" {
  description = "Cloud Run API CPU limit."
  type        = string
  default     = "1"
}

variable "api_memory" {
  description = "Cloud Run API memory limit."
  type        = string
  default     = "1Gi"
}

variable "api_timeout_seconds" {
  description = "Cloud Run request timeout in seconds."
  type        = number
  default     = 2
}

variable "redis_tier" {
  description = "Memorystore Redis tier."
  type        = string
  default     = "BASIC"
}

variable "redis_memory_size_gb" {
  description = "Memorystore Redis memory size."
  type        = number
  default     = 2
}

variable "redis_version" {
  description = "Memorystore Redis version."
  type        = string
  default     = "REDIS_7_0"
}

variable "elasticsearch_node_count" {
  description = "Number of Terraform-managed GCE Elasticsearch nodes."
  type        = number
  default     = 2

  validation {
    condition     = var.elasticsearch_node_count >= 1 && var.elasticsearch_node_count <= 3
    error_message = "elasticsearch_node_count must be between 1 and 3 for this deployment profile."
  }
}

variable "elasticsearch_zone_suffixes" {
  description = "Zone suffixes used to spread Elasticsearch nodes inside the selected region."
  type        = list(string)
  default     = ["a", "b", "c"]
}

variable "elasticsearch_machine_type" {
  description = "GCE machine type for each Elasticsearch node."
  type        = string
  default     = "e2-standard-4"
}

variable "elasticsearch_disk_size_gb" {
  description = "Boot/data disk size for each Elasticsearch node."
  type        = number
  default     = 150
}

variable "elasticsearch_disk_type" {
  description = "Boot/data disk type for each Elasticsearch node."
  type        = string
  default     = "pd-balanced"
}

variable "elasticsearch_boot_image" {
  description = "Boot image for Elasticsearch GCE nodes."
  type        = string
  default     = "debian-cloud/debian-12"
}

variable "elasticsearch_image" {
  description = "Elasticsearch Docker image."
  type        = string
  default     = "docker.elastic.co/elasticsearch/elasticsearch:8.15.3"
}

variable "elasticsearch_cluster_name" {
  description = "Elasticsearch cluster name."
  type        = string
  default     = "hybrid-search-v2"
}

variable "elasticsearch_heap_size" {
  description = "Elasticsearch JVM heap size per node."
  type        = string
  default     = "8g"
}

variable "elasticsearch_container_memory" {
  description = "Docker memory limit for each Elasticsearch container."
  type        = string
  default     = "14g"
}

variable "schema_version" {
  description = "Active schema version exposed by the API."
  type        = string
  default     = "schema-v2"
}

variable "index_version" {
  description = "Initial active index version exposed by the API."
  type        = string
  default     = "index-v2"
}

variable "model_version" {
  description = "Active embedding model version exposed by the API."
  type        = string
  default     = "hash-embedding-v2"
}

variable "content_version" {
  description = "Active content version exposed by the API."
  type        = string
  default     = "content-v2"
}

variable "magazine_info_index" {
  description = "Magazine info index name."
  type        = string
  default     = "magazine_info_v2"
}

variable "magazine_content_index" {
  description = "Magazine content index name."
  type        = string
  default     = "magazine_content_v2"
}

variable "elasticsearch_num_candidates" {
  description = "Elasticsearch kNN num_candidates."
  type        = number
  default     = 100
}

variable "embedding_dimension" {
  description = "Embedding vector dimension."
  type        = number
  default     = 32
}

variable "max_query_length" {
  description = "Maximum query length."
  type        = number
  default     = 256
}

variable "max_body_size_bytes" {
  description = "Maximum request body size."
  type        = number
  default     = 4096
}

variable "max_top_k" {
  description = "Maximum returned result count."
  type        = number
  default     = 20
}

variable "max_offset" {
  description = "Maximum pagination offset."
  type        = number
  default     = 1000
}

variable "max_keyword_candidates" {
  description = "Maximum keyword candidates before fusion."
  type        = number
  default     = 100
}

variable "max_vector_candidates" {
  description = "Maximum vector candidates before fusion."
  type        = number
  default     = 100
}

variable "max_fusion_candidates" {
  description = "Maximum fusion candidate set size."
  type        = number
  default     = 200
}

variable "request_deadline_ms" {
  description = "Application request deadline in milliseconds."
  type        = number
  default     = 1200
}

variable "redis_timeout_ms" {
  description = "Redis timeout in milliseconds."
  type        = number
  default     = 100
}

variable "search_timeout_ms" {
  description = "Elasticsearch timeout in milliseconds."
  type        = number
  default     = 800
}

variable "embedding_timeout_ms" {
  description = "Embedding timeout in milliseconds."
  type        = number
  default     = 500
}

variable "max_concurrent_requests" {
  description = "Application-level concurrent request cap."
  type        = number
  default     = 16
}

variable "semaphore_acquire_timeout_ms" {
  description = "Application semaphore acquire timeout."
  type        = number
  default     = 25
}

variable "per_client_rate_per_minute" {
  description = "Per-client rate limit."
  type        = number
  default     = 6000
}

variable "global_rate_per_minute" {
  description = "Global process-local rate limit."
  type        = number
  default     = 6000
}

variable "rate_window_seconds" {
  description = "Rate-limit window in seconds."
  type        = number
  default     = 60
}

variable "rate_limiter_max_clients" {
  description = "Maximum tracked client IDs in the process-local rate limiter."
  type        = number
  default     = 100000
}

variable "rate_limiter_cleanup_interval_seconds" {
  description = "Rate-limiter cleanup interval."
  type        = number
  default     = 30
}

variable "max_client_id_length" {
  description = "Maximum accepted client ID length."
  type        = number
  default     = 128
}

variable "retry_after_seconds" {
  description = "Retry-After response value for bounded 429s."
  type        = number
  default     = 1
}

variable "trust_client_id_header" {
  description = "Trust x-client-id from the edge."
  type        = bool
  default     = false
}

variable "max_query_tokens" {
  description = "Maximum query token count."
  type        = number
  default     = 48
}

variable "cutover_hour" {
  description = "Daily index cutover hour."
  type        = number
  default     = 23
}

variable "cutover_minute" {
  description = "Daily index cutover minute."
  type        = number
  default     = 0
}

variable "cache_ttl_seconds" {
  description = "Redis cache TTL."
  type        = number
  default     = 600
}

variable "cache_max_entries" {
  description = "Application cache max entries."
  type        = number
  default     = 50000
}

variable "cache_required_for_readiness" {
  description = "Whether cache availability is required for readiness."
  type        = bool
  default     = true
}

variable "log_raw_queries" {
  description = "Whether raw user queries may be logged."
  type        = bool
  default     = false
}
