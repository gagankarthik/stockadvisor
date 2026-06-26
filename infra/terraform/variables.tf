variable "region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region to deploy into."
}

variable "project" {
  type        = string
  default     = "marketdesk"
  description = "Name prefix for all resources."
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = <<-EOT
    Container image tag. Pass a unique value per deploy (e.g. the git SHA:
    `-var image_tag=$(git rev-parse --short HEAD)`) so the Lambdas pick up new
    code — reusing a tag won't trigger an update.
  EOT
}

variable "finnhub_key" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Finnhub API key."
}

variable "alphavantage_key" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Alpha Vantage API key."
}

variable "openai_api_key" {
  type        = string
  default     = ""
  sensitive   = true
  description = "OpenAI API key."
}

variable "cors_origin" {
  type        = string
  default     = "*"
  description = "Allowed CORS origin for the API (set to your UI domain in prod)."
}

variable "api_memory_mb" {
  type        = number
  default     = 1536
  description = "Memory for the API Lambda."
}

variable "train_memory_mb" {
  type        = number
  default     = 3008
  description = "Memory for the training Lambda."
}

variable "refresh_schedule" {
  type        = string
  default     = "cron(30 21 ? * MON-FRI *)"
  description = "EventBridge schedule for the refresh job (UTC; ~after US close)."
}
