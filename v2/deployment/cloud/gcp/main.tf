locals {
  service_name            = "hybrid-search-v2-api"
  redis_name              = "hybrid-search-v2-redis"
  network_name            = "hybrid-search-v2"
  elasticsearch_name      = "hybrid-search-v2-es"
  elasticsearch_node_ips  = [for address in google_compute_address.elasticsearch : address.address]
  elasticsearch_node_urls = [for address in google_compute_address.elasticsearch : "http://${address.address}:9200"]
}

resource "google_project_service" "required" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "compute.googleapis.com",
    "monitoring.googleapis.com",
    "redis.googleapis.com",
    "run.googleapis.com",
    "vpcaccess.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

resource "google_compute_network" "main" {
  depends_on = [google_project_service.required]

  name                    = local.network_name
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "main" {
  name                     = "${local.network_name}-${var.region}"
  region                   = var.region
  network                  = google_compute_network.main.id
  ip_cidr_range            = var.subnet_cidr
  private_ip_google_access = true
}

resource "google_compute_router" "main" {
  name    = "${local.network_name}-${var.region}"
  network = google_compute_network.main.id
  region  = var.region
}

resource "google_compute_router_nat" "main" {
  name                               = "${local.network_name}-${var.region}"
  router                             = google_compute_router.main.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"

  subnetwork {
    name                    = google_compute_subnetwork.main.id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }
}

resource "google_artifact_registry_repository" "api" {
  depends_on = [google_project_service.required]

  location      = var.region
  repository_id = "hybrid-search-v2"
  description   = "Hybrid Search v2 API images"
  format        = "DOCKER"
}

resource "google_service_account" "api" {
  account_id   = "hybrid-search-v2-api"
  display_name = "Hybrid Search v2 API"
}

resource "google_service_account" "elasticsearch" {
  account_id   = "hybrid-search-v2-es"
  display_name = "Hybrid Search v2 Elasticsearch"
}

resource "google_vpc_access_connector" "api" {
  depends_on = [google_project_service.required]

  name          = "hybrid-search-v2"
  region        = var.region
  network       = google_compute_network.main.name
  ip_cidr_range = var.vpc_connector_cidr
  min_instances = var.vpc_connector_min_instances
  max_instances = var.vpc_connector_max_instances
}

resource "google_redis_instance" "cache" {
  depends_on = [google_project_service.required]

  name               = local.redis_name
  tier               = var.redis_tier
  memory_size_gb     = var.redis_memory_size_gb
  region             = var.region
  redis_version      = var.redis_version
  authorized_network = google_compute_network.main.id
}

resource "google_compute_address" "elasticsearch" {
  count = var.elasticsearch_node_count

  name         = "${local.elasticsearch_name}-${count.index}"
  address_type = "INTERNAL"
  subnetwork   = google_compute_subnetwork.main.id
  region       = var.region
}

resource "google_compute_firewall" "elasticsearch_internal" {
  name    = "${local.elasticsearch_name}-internal"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["9200", "9300"]
  }

  source_ranges = [
    var.subnet_cidr,
    var.vpc_connector_cidr,
  ]

  target_tags = ["hybrid-search-v2-es"]
}

resource "google_compute_instance" "elasticsearch" {
  count = var.elasticsearch_node_count

  name         = "${local.elasticsearch_name}-${count.index}"
  machine_type = var.elasticsearch_machine_type
  zone         = "${var.region}-${var.elasticsearch_zone_suffixes[count.index % length(var.elasticsearch_zone_suffixes)]}"
  tags         = ["hybrid-search-v2-es"]

  boot_disk {
    initialize_params {
      image = var.elasticsearch_boot_image
      size  = var.elasticsearch_disk_size_gb
      type  = var.elasticsearch_disk_type
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.main.id
    network_ip = google_compute_address.elasticsearch[count.index].address
  }

  metadata = {
    startup-script = templatefile("${path.module}/startup-elasticsearch.sh.tftpl", {
      cluster_name         = var.elasticsearch_cluster_name
      node_name            = "${local.elasticsearch_name}-${count.index}"
      node_ips             = local.elasticsearch_node_ips
      node_names           = [for index in range(var.elasticsearch_node_count) : "${local.elasticsearch_name}-${index}"]
      elasticsearch_image  = var.elasticsearch_image
      elasticsearch_heap   = var.elasticsearch_heap_size
      elasticsearch_memory = var.elasticsearch_container_memory
    })
  }

  service_account {
    email  = google_service_account.elasticsearch.email
    scopes = ["cloud-platform"]
  }

  allow_stopping_for_update = true

  depends_on = [
    google_compute_firewall.elasticsearch_internal,
    google_compute_router_nat.main,
    google_project_service.required,
  ]
}

resource "google_cloud_run_v2_service" "api" {
  depends_on = [
    google_project_service.required,
    google_redis_instance.cache,
    google_vpc_access_connector.api,
    google_compute_instance.elasticsearch,
  ]

  name     = local.service_name
  location = var.region
  ingress  = var.cloud_run_ingress

  template {
    service_account                  = google_service_account.api.email
    timeout                          = "${var.api_timeout_seconds}s"
    max_instance_request_concurrency = var.api_concurrency

    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = var.api_max_instances
    }

    vpc_access {
      connector = google_vpc_access_connector.api.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = var.api_image

      ports {
        container_port = 8002
      }

      resources {
        limits = {
          cpu    = var.api_cpu
          memory = var.api_memory
        }
      }

      env {
        name  = "HYBRID_SEARCH_APP_NAME"
        value = "hybrid-search-v2"
      }
      env {
        name  = "HYBRID_SEARCH_API_VERSION"
        value = "v2"
      }
      env {
        name  = "HYBRID_SEARCH_SCHEMA_VERSION"
        value = var.schema_version
      }
      env {
        name  = "HYBRID_SEARCH_INDEX_VERSION"
        value = var.index_version
      }
      env {
        name  = "HYBRID_SEARCH_MODEL_VERSION"
        value = var.model_version
      }
      env {
        name  = "HYBRID_SEARCH_CONTENT_VERSION"
        value = var.content_version
      }
      env {
        name  = "HYBRID_SEARCH_TRACER_NAME"
        value = "hybrid_search_v2"
      }
      env {
        name  = "HYBRID_SEARCH_HOST"
        value = "0.0.0.0"
      }
      env {
        name  = "HYBRID_SEARCH_PORT"
        value = "8002"
      }
      env {
        name  = "HYBRID_SEARCH_SEED_DATA_PATH"
        value = "data/seed_magazines.json"
      }
      env {
        name  = "HYBRID_SEARCH_SEARCH_BACKEND"
        value = "elasticsearch"
      }
      env {
        name  = "HYBRID_SEARCH_CACHE_BACKEND"
        value = "redis"
      }
      env {
        name  = "HYBRID_SEARCH_REDIS_URL"
        value = "redis://${google_redis_instance.cache.host}:${google_redis_instance.cache.port}/0"
      }
      env {
        name  = "HYBRID_SEARCH_ELASTICSEARCH_URL"
        value = local.elasticsearch_node_urls[0]
      }
      env {
        name  = "HYBRID_SEARCH_MAGAZINE_INFO_INDEX"
        value = var.magazine_info_index
      }
      env {
        name  = "HYBRID_SEARCH_MAGAZINE_CONTENT_INDEX"
        value = var.magazine_content_index
      }
      env {
        name  = "HYBRID_SEARCH_ELASTICSEARCH_NUM_CANDIDATES"
        value = tostring(var.elasticsearch_num_candidates)
      }
      env {
        name  = "HYBRID_SEARCH_EMBEDDING_DIMENSION"
        value = tostring(var.embedding_dimension)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_QUERY_LENGTH"
        value = tostring(var.max_query_length)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_BODY_SIZE_BYTES"
        value = tostring(var.max_body_size_bytes)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_TOP_K"
        value = tostring(var.max_top_k)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_OFFSET"
        value = tostring(var.max_offset)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_KEYWORD_CANDIDATES"
        value = tostring(var.max_keyword_candidates)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_VECTOR_CANDIDATES"
        value = tostring(var.max_vector_candidates)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_FUSION_CANDIDATES"
        value = tostring(var.max_fusion_candidates)
      }
      env {
        name  = "HYBRID_SEARCH_REQUEST_DEADLINE_MS"
        value = tostring(var.request_deadline_ms)
      }
      env {
        name  = "HYBRID_SEARCH_REDIS_TIMEOUT_MS"
        value = tostring(var.redis_timeout_ms)
      }
      env {
        name  = "HYBRID_SEARCH_SEARCH_TIMEOUT_MS"
        value = tostring(var.search_timeout_ms)
      }
      env {
        name  = "HYBRID_SEARCH_EMBEDDING_TIMEOUT_MS"
        value = tostring(var.embedding_timeout_ms)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_CONCURRENT_REQUESTS"
        value = tostring(var.max_concurrent_requests)
      }
      env {
        name  = "HYBRID_SEARCH_SEMAPHORE_ACQUIRE_TIMEOUT_MS"
        value = tostring(var.semaphore_acquire_timeout_ms)
      }
      env {
        name  = "HYBRID_SEARCH_PER_CLIENT_RATE_PER_MINUTE"
        value = tostring(var.per_client_rate_per_minute)
      }
      env {
        name  = "HYBRID_SEARCH_GLOBAL_RATE_PER_MINUTE"
        value = tostring(var.global_rate_per_minute)
      }
      env {
        name  = "HYBRID_SEARCH_RATE_WINDOW_SECONDS"
        value = tostring(var.rate_window_seconds)
      }
      env {
        name  = "HYBRID_SEARCH_RATE_LIMITER_MAX_CLIENTS"
        value = tostring(var.rate_limiter_max_clients)
      }
      env {
        name  = "HYBRID_SEARCH_RATE_LIMITER_CLEANUP_INTERVAL_SECONDS"
        value = tostring(var.rate_limiter_cleanup_interval_seconds)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_CLIENT_ID_LENGTH"
        value = tostring(var.max_client_id_length)
      }
      env {
        name  = "HYBRID_SEARCH_RETRY_AFTER_SECONDS"
        value = tostring(var.retry_after_seconds)
      }
      env {
        name  = "HYBRID_SEARCH_TRUST_CLIENT_ID_HEADER"
        value = tostring(var.trust_client_id_header)
      }
      env {
        name  = "HYBRID_SEARCH_MAX_QUERY_TOKENS"
        value = tostring(var.max_query_tokens)
      }
      env {
        name  = "HYBRID_SEARCH_CUTOVER_HOUR"
        value = tostring(var.cutover_hour)
      }
      env {
        name  = "HYBRID_SEARCH_CUTOVER_MINUTE"
        value = tostring(var.cutover_minute)
      }
      env {
        name  = "HYBRID_SEARCH_MONTHLY_BUDGET_GBP"
        value = tostring(var.monthly_budget_gbp)
      }
      env {
        name  = "HYBRID_SEARCH_OBSERVABILITY_BUDGET_GBP"
        value = tostring(var.observability_budget_gbp)
      }
      env {
        name  = "HYBRID_SEARCH_CACHE_TTL_SECONDS"
        value = tostring(var.cache_ttl_seconds)
      }
      env {
        name  = "HYBRID_SEARCH_CACHE_MAX_ENTRIES"
        value = tostring(var.cache_max_entries)
      }
      env {
        name  = "HYBRID_SEARCH_CACHE_REQUIRED_FOR_READINESS"
        value = tostring(var.cache_required_for_readiness)
      }
      env {
        name  = "HYBRID_SEARCH_LOG_RAW_QUERIES"
        value = tostring(var.log_raw_queries)
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  count = var.allow_unauthenticated ? 1 : 0

  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_monitoring_notification_channel" "budget_email" {
  depends_on = [google_project_service.required]

  display_name = "Hybrid Search v2 budget email"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

resource "google_billing_budget" "monthly" {
  depends_on = [google_project_service.required]

  billing_account = var.billing_account_id
  display_name    = "hybrid-search-v2-monthly"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "GBP"
      units         = tostring(var.monthly_budget_gbp)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }

  threshold_rules {
    threshold_percent = 0.8
  }

  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    monitoring_notification_channels = [google_monitoring_notification_channel.budget_email.id]
  }
}
