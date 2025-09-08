#!/usr/bin/env bash

# Simple timeout function for macOS compatibility
timeout() {
    local seconds=$1; shift
    local cmd="$*"
    
    {
        eval "$cmd" &
        local pid=$!
        sleep "$seconds" & local sleep_pid=$!
        wait "$pid" 2>/dev/null && kill "$sleep_pid" 2>/dev/null
    } || {
        kill "$pid" 2>/dev/null 2>&1
        echo "Command timed out after ${seconds}s" >&2
        return 124
    }
}
set -euo pipefail

# Usage:
#   PROJECT_ID=the-tendril-466607-n8 REGION=us-central1 SERVICE=samo-unified-api \
#   ./scripts/deployment/deploy_secure_unified.sh

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-samo-unified-api}"
IMAGE_REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/samo-dl/${SERVICE}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required" >&2
  exit 1
fi

echo "Step: Starting Docker build (may take 5-10 minutes)..."
timeout 15m docker build -f Dockerfile.unified -t "${IMAGE_REPO}:latest" .
echo "Step: Docker build completed."

echo "Step: Starting Docker push (may take 2-5 minutes)..."
timeout 10m docker push "${IMAGE_REPO}:latest"
echo "Step: Docker push completed."

echo "Step: Starting Cloud Run deployment (may take 3-7 minutes)..."
timeout 10m gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --image "${IMAGE_REPO}:latest" \
  --allow-unauthenticated \
  --port 8080 \
  --memory=4Gi \
  --cpu=2 \
  --timeout=600 \
  --min-instances=0 \
  --max-instances=5 \
  --concurrency=50 \
  --set-env-vars="RATE_LIMIT_REQUESTS_PER_MINUTE=100,RATE_LIMIT_BURST_SIZE=20,RATE_LIMIT_MAX_CONCURRENT=10,RATE_LIMIT_RAPID_FIRE_THRESHOLD=20,RATE_LIMIT_SUSTAINED_THRESHOLD=150" \
  --set-env-vars="LOG_LEVEL=INFO,ENVIRONMENT=production" \
  --set-env-vars="EMOTION_MODEL_ID=0xmnrv/samo,TEXT_SUMMARIZER_MODEL=t5-small,VOICE_TRANSCRIBER_MODEL=base"
echo "Step: Cloud Run deployment completed."

echo "Deployment completed. Service URL:"
gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --platform managed --format='value(status.url)'