#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PROJECT_ID=the-tendril-466607-n8 REGION=us-central1 SERVICE=samo-unified-api \
#   ./scripts/deployment/deploy_unified_cloud_run.sh

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-samo-unified-api}"
IMAGE_REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/samo-dl/${SERVICE}"
TAG=$(date +%Y%m%d%H%M%S)

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required" >&2
  exit 1
fi

echo "Building image ${IMAGE_REPO}:${TAG}..."
gcloud builds submit --project "${PROJECT_ID}" --tag "${IMAGE_REPO}:${TAG}" .

echo "Deploying to Cloud Run service ${SERVICE} in ${REGION}..."
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --image "${IMAGE_REPO}:${TAG}" \
  --allow-unauthenticated \
  --port 8080

echo "Deployment triggered. Service URL:"
gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --platform managed --format='value(status.url)'

