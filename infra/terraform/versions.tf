terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }

  # Recommended for teams: store state in S3 + DynamoDB lock.
  # Uncomment and fill in, then `terraform init -reconfigure`.
  # backend "s3" {
  #   bucket         = "your-tf-state-bucket"
  #   key            = "marketdesk/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.region
}
