#!/usr/bin/env bash
set -euo pipefail


docker-compose up -d


export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
export S3_ENDPOINT_URL="http://localhost:4566"
export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"


aws --endpoint-url=$S3_ENDPOINT_URL s3 rb s3://nyc-duration --force || true
aws --endpoint-url=$S3_ENDPOINT_URL s3 mb s3://nyc-duration


pytest integration_test.py -q

docker-compose down
