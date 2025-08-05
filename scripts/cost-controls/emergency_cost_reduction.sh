#!/bin/bash

# Emergency Cost Reduction Script
# This script immediately scales down all resources to minimize costs

set -e

PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"

echo "EMERGENCY COST REDUCTION - Scaling down all resources"

# Scale down Cloud Run services
gcloud run services list --format="value(name)" | \
while read service; do
    echo "Scaling down Cloud Run service: $service"
    gcloud run services update $service \
        --region=$REGION \
        --min-instances=0 \
        --max-instances=0
done

# Scale down Vertex AI endpoints
gcloud ai endpoints list --region=$REGION --format="value(name)" | \
while read endpoint; do
    echo "Scaling down Vertex AI endpoint: $endpoint"
    gcloud ai endpoints update $endpoint \
        --region=$REGION \
        --min-replica-count=0 \
        --max-replica-count=0
done

# Stop all compute instances
gcloud compute instances list --format="value(name,zone)" | \
while read name zone; do
    echo "Stopping compute instance: $name in zone $zone"
    gcloud compute instances stop $name --zone=$zone
done

echo "Emergency cost reduction completed"
echo "All resources have been scaled down to minimum levels"
