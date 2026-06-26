output "api_url" {
  description = "Public API base URL — set as NEXT_PUBLIC_API_BASE_URL in the UI."
  value       = aws_lambda_function_url.api.function_url
}

output "artifact_bucket" {
  description = "S3 bucket holding the model + snapshot + plans."
  value       = aws_s3_bucket.artifacts.bucket
}

output "ecr_api_repo" {
  value = aws_ecr_repository.api.repository_url
}
