locals {
  service_name = "hybrid-search-v2-api"
  redis_name   = "hybrid-search-v2-redis"
}

resource "google_project_service" "required" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "monitoring.googleapis.com",
    "redis.googleapis.com",
    "run.googleapis.com",
    "vpcaccess.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
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

resource "google_vpc_access_connector" "api" {
  depends_on = [google_project_service.required]

  name          = "hybrid-search-v2"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  min_instances = 2
  max_instances = 3
}

resource "google_redis_instance" "cache" {
  depends_on = [google_project_service.required]

  name           = local.redis_name
  tier           = "BASIC"
  memory_size_gb = 1
  region         = var.region
  redis_version  = "REDIS_7_0"
}

resource "google_cloud_run_v2_service" "api" {
  depends_on = [
    google_project_service.required,
    google_redis_instance.cache,
    google_vpc_access_connector.api,
  ]

  name     = local.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.api.email
    timeout         = "2s"

    scaling {
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
        value = "schema-v2"
      }
      env {
        name  = "HYBRID_SEARCH_INDEX_VERSION"
        value = "index-v2"
      }
      env {
        name  = "HYBRID_SEARCH_MODEL_VERSION"
        value = "hash-embedding-v2"
      }
      env {
        name  = "HYBRID_SEARCH_CONTENT_VERSION"
        value = "content-v2"
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
        value = var.elasticsearch_url
      }
      env {
        name  = "HYBRID_SEARCH_MAGAZINE_INFO_INDEX"
        value = "magazine_info_v2"
      }
      env {
        name  = "HYBRID_SEARCH_MAGAZINE_CONTENT_INDEX"
        value = "magazine_content_v2"
      }
      env {
        name  = "HYBRID_SEARCH_ELASTICSEARCH_NUM_CANDIDATES"
        value = "100"
      }
      env {
        name  = "HYBRID_SEARCH_EMBEDDING_DIMENSION"
        value = "32"
      }
      env {
        name  = "HYBRID_SEARCH_MAX_QUERY_LENGTH"
        value = "256"
      }
      env {
        name  = "HYBRID_SEARCH_MAX_BODY_SIZE_BYTES"
        value = "4096"
      }
      env {
        name  = "HYBRID_SEARCH_MAX_TOP_K"
        value = "20"
      }
      env {
        name  = "HYBRID_SEARCH_MAX_OFFSET"
        value = "1000"
      }
      env {
        name  = "HYBRID_SEARCH_MAX_KEYWORD_CANDIDATES"
        value = "100"
      }
      env {
        name  = "HYBRID_SEARCH_MAX_VECTOR_CANDIDATES"
        value = "100"
      }
      env {
        name  = "HYBRID_SEARCH_MAX_FUSION_CANDIDATES"
        value = "200"
      }
      env {
        name  = "HYBRID_SEARCH_REQUEST_DEADLINE_MS"
        value = "1200"
      }
      env {
        name  = "HYBRID_SEARCH_REDIS_TIMEOUT_MS"
        value = "100"
      }
      env {
        name  = "HYBRID_SEARCH_SEARCH_TIMEOUT_MS"
        value = "800"
      }
      env {
        name  = "HYBRID_SEARCH_EMBEDDING_TIMEOUT_MS"
        value = "500"
      }
      env {
        name  = "HYBRID_SEARCH_MAX_CONCURRENT_REQUESTS"
        value = "16"
      }
      env {
        name  = "HYBRID_SEARCH_PER_CLIENT_RATE_PER_MINUTE"
        value = "60"
      }
      env {
        name  = "HYBRID_SEARCH_GLOBAL_RATE_PER_MINUTE"
        value = "120"
      }
      env {
        name  = "HYBRID_SEARCH_CUTOVER_HOUR"
        value = "23"
      }
      env {
        name  = "HYBRID_SEARCH_CUTOVER_MINUTE"
        value = "0"
      }
      env {
        name  = "HYBRID_SEARCH_MONTHLY_BUDGET_GBP"
        value = tostring(var.monthly_budget_gbp)
      }
      env {
        name  = "HYBRID_SEARCH_OBSERVABILITY_BUDGET_GBP"
        value = "20"
      }
      env {
        name  = "HYBRID_SEARCH_CACHE_TTL_SECONDS"
        value = "300"
      }
      env {
        name  = "HYBRID_SEARCH_CACHE_MAX_ENTRIES"
        value = "1024"
      }
      env {
        name  = "HYBRID_SEARCH_CACHE_REQUIRED_FOR_READINESS"
        value = "true"
      }
      env {
        name  = "HYBRID_SEARCH_LOG_RAW_QUERIES"
        value = "false"
      }
    }
  }
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
