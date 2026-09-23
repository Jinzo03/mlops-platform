# SQS Queue for Asynchronous Embedding Requests
resource "aws_sqs_queue" "embedding_queue" {
  name                       = "embedding-request-queue"
  delay_seconds              = 0
  max_message_size           = 262144 # 256 KB
  message_retention_seconds  = 86400  # 1 day
  receive_wait_time_seconds  = 10     # Long polling
  visibility_timeout_seconds = 60     # Processing deadline before retry

  tags = {
    Environment = "production"
    Service     = "mlops-embedding"
  }
}

output "sqs_queue_url" {
  description = "URL of the provisioned SQS queue"
  value       = aws_sqs_queue.embedding_queue.url
}

output "sqs_queue_arn" {
  description = "ARN of the provisioned SQS queue"
  value       = aws_sqs_queue.embedding_queue.arn
}