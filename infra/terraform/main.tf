data "aws_caller_identity" "current" {}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  ecr_host    = "${local.account_id}.dkr.ecr.${var.region}.amazonaws.com"
  repo_root   = "${path.module}/../.."
  api_image   = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
  common_env = {
    MARKETDESK_ARTIFACT_URI = "s3://${aws_s3_bucket.artifacts.bucket}/prod"
    FINNHUB_KEY             = var.finnhub_key
    ALPHAVANTAGE_KEY        = var.alphavantage_key
    OPENAI_API_KEY          = var.openai_api_key
  }
  tags = {
    Project   = var.project
    ManagedBy = "terraform"
  }
}

# --------------------------------------------------------------------------- #
# Container registry + build/push                                             #
# --------------------------------------------------------------------------- #
resource "aws_ecr_repository" "api" {
  name                 = "${var.project}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

# Build the single image (serves both the API and the scheduled refresh) and
# push to ECR. Requires Docker + AWS CLI on the machine running terraform
# (locally use Git Bash / WSL; in CI use a Linux runner).
resource "null_resource" "images" {
  triggers = {
    tag          = var.image_tag
    dockerfile   = filemd5("${local.repo_root}/Dockerfile")
    requirements = filemd5("${local.repo_root}/requirements.txt")
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-c"]
    working_dir = local.repo_root
    command     = <<-EOT
      set -euo pipefail
      aws ecr get-login-password --region ${var.region} \
        | docker login --username AWS --password-stdin ${local.ecr_host}
      docker build -t ${local.api_image} -f Dockerfile .
      docker push  ${local.api_image}
    EOT
  }

  depends_on = [aws_ecr_repository.api]
}

# --------------------------------------------------------------------------- #
# Artifact store (model + snapshot + plans)                                   #
# --------------------------------------------------------------------------- #
resource "aws_s3_bucket" "artifacts" {
  bucket        = "${var.project}-artifacts-${local.account_id}"
  force_destroy = true
  tags          = local.tags
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

# --------------------------------------------------------------------------- #
# IAM (shared execution role)                                                 #
# --------------------------------------------------------------------------- #
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.project}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "lambda" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:*:*:*"]
  }
  statement {
    sid       = "ArtifactBucket"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
    resources = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${var.project}-lambda"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

# --------------------------------------------------------------------------- #
# Single Lambda: FastAPI via Mangum (public Function URL) + the scheduled      #
# refresh. The handler dispatches by event type; sized for the heavy training  #
# run so the one function covers both jobs.                                    #
# --------------------------------------------------------------------------- #
resource "aws_lambda_function" "api" {
  function_name = "${var.project}-api"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = local.api_image
  memory_size   = var.train_memory_mb
  timeout       = 900
  architectures = ["x86_64"]

  environment {
    variables = merge(local.common_env, {
      MARKETDESK_CORS_ORIGINS = jsonencode([var.cors_origin])
    })
  }

  depends_on = [null_resource.images, aws_iam_role_policy.lambda]
  tags       = local.tags
}

# CORS is handled inside the app (FastAPI CORSMiddleware), so the Function URL
# deliberately has no cors block — setting it in both places makes Lambda emit
# duplicate Access-Control-Allow-Origin headers, which browsers reject.
resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
}

# Daily EventBridge schedule -> the same function (dispatched to the refresh job).
resource "aws_cloudwatch_event_rule" "refresh" {
  name                = "${var.project}-refresh"
  schedule_expression = var.refresh_schedule
  tags                = local.tags
}

resource "aws_cloudwatch_event_target" "refresh" {
  rule = aws_cloudwatch_event_rule.refresh.name
  arn  = aws_lambda_function.api.arn
}

resource "aws_lambda_permission" "events" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.refresh.arn
}
