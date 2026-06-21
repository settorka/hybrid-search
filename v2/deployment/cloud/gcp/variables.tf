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

variable "elasticsearch_url" {
  description = "Elasticsearch endpoint reachable from Cloud Run."
  type        = string
}

variable "monthly_budget_gbp" {
  description = "Monthly budget ceiling in GBP."
  type        = number
  default     = 100
}

variable "api_max_instances" {
  description = "Cloud Run max instances."
  type        = number
  default     = 2
}

variable "api_cpu" {
  description = "Cloud Run API CPU limit."
  type        = string
  default     = "1"
}

variable "api_memory" {
  description = "Cloud Run API memory limit."
  type        = string
  default     = "512Mi"
}
