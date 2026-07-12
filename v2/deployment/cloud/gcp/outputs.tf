output "artifact_registry_repository" {
  description = "Docker repository for the API image."
  value       = google_artifact_registry_repository.api.name
}

output "api_url" {
  description = "Cloud Run API URL."
  value       = google_cloud_run_v2_service.api.uri
}

output "redis_host" {
  description = "Private Redis host."
  value       = google_redis_instance.cache.host
}

output "elasticsearch_private_urls" {
  description = "Private Elasticsearch node URLs."
  value       = local.elasticsearch_node_urls
}

output "elasticsearch_primary_url" {
  description = "Private Elasticsearch URL used by Cloud Run."
  value       = local.elasticsearch_node_urls[0]
}

output "budget_name" {
  description = "GCP budget resource name."
  value       = google_billing_budget.monthly.name
}
