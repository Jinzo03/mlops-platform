terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# Unique S3 Bucket for MLflow Artifacts (Models, Metrics, Datasets)
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket        = "mlflow-artifacts-jinzo03-mlops"
  force_destroy = true
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "mlflow_artifacts_block" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.mlflow_artifacts.id
  description = "S3 bucket for MLflow artifacts"
}