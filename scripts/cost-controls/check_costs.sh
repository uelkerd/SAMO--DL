#!/bin/bash

# Simple Cost Monitoring Script
# This script checks current resource usage and costs

PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"

echo "=== GCP Cost Monitoring Report ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Date: $(date)"
echo

echo "=== Cloud Run Services ==="
gcloud run services list --region=$REGION --format="table(name,status.url,status.conditions[0].status)" 2>/dev/null || echo "No Cloud Run services found"

echo -e "\n=== Vertex AI Endpoints ==="
gcloud ai endpoints list --region=$REGION --format="table(name,displayName,deployedModels[0].model)" 2>/dev/null || echo "No Vertex AI endpoints found"

echo -e "\n=== Compute Instances ==="
gcloud compute instances list --format="table(name,zone,machineType,status)" 2>/dev/null || echo "No compute instances found"

echo -e "\n=== Cost Optimization Tips ==="
echo "1. Scale down unused services: gcloud run services update [SERVICE] --max-instances=0"
echo "2. Stop compute instances: gcloud compute instances stop [INSTANCE] --zone=[ZONE]"
echo "3. Emergency reduction: ./scripts/cost-controls/emergency_cost_reduction.sh"
echo "4. Check billing: https://console.cloud.google.com/billing"
