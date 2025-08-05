# GCP Cost Controls & Budget Management Guide

## Overview

This guide provides comprehensive cost control strategies for the SAMO Deep Learning project on Google Cloud Platform. It includes budget alerts, spending limits, resource quotas, and monitoring to prevent unexpected charges.

## Immediate Cost Control Actions

### 1. Set Up Budget Alerts

```bash
# Create a budget for the project
gcloud billing budgets create \
    --billing-account=0156F5-8F20E3-96A680 \
    --display-name="SAMO-DL Project Budget" \
    --budget-amount=100USD \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.8 \
    --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0 \
    --notifications-rule=pubsub-topic=projects/the-tendril-466607-n8/topics/budget-alerts \
    --notifications-rule=email=your-email@domain.com
```

### 2. Enable Billing Export

```bash
# Create a BigQuery dataset for billing export
bq mk --dataset the-tendril-466607-n8:billing_export

# Enable billing export to BigQuery
gcloud billing accounts update 0156F5-8F20E3-96A680 \
    --billing-account=0156F5-8F20E3-96A680 \
    --enable-bigquery-export \
    --bigquery-dataset=projects/the-tendril-466607-n8/datasets/billing_export
```

### 3. Set Resource Quotas

```bash
# Set compute engine quotas
gcloud compute regions describe us-central1 \
    --format="value(quotas[].limit,quotas[].metric,quotas[].usage)" \
    --project=the-tendril-466607-n8

# Request quota increase if needed
gcloud compute regions update us-central1 \
    --quotas=CPUS=4,CPUS_ALL_REGIONS=8 \
    --project=the-tendril-466607-n8
```

## Budget Configuration

### 1. Create Budget Configuration File

```yaml
# budgets/samo-dl-budget.yaml
displayName: "SAMO-DL Project Budget"
amount:
  specifiedAmount:
    currencyCode: "USD"
    units: "100"
budgetFilter:
  projects:
    - "projects/the-tendril-466607-n8"
thresholdRules:
  - thresholdPercent: 0.5
    spendBasis: CURRENT_SPEND
  - thresholdPercent: 0.8
    spendBasis: CURRENT_SPEND
  - thresholdPercent: 0.9
    spendBasis: CURRENT_SPEND
  - thresholdPercent: 1.0
    spendBasis: CURRENT_SPEND
notificationsRule:
  pubsubTopic: "projects/the-tendril-466607-n8/topics/budget-alerts"
  schemaVersion: "1.0"
  monitoringNotificationChannels:
    - "projects/the-tendril-466607-n8/notificationChannels/123456789"
```

### 2. Apply Budget Configuration

```bash
# Apply budget configuration
gcloud billing budgets create --billing-account=0156F5-8F20E3-96A680 \
    --budget-file=budgets/samo-dl-budget.yaml
```

## Resource-Specific Cost Controls

### 1. Cloud Run Cost Controls

```bash
# Set Cloud Run service limits
gcloud run services update emotion-detection-api \
    --region=us-central1 \
    --max-instances=5 \
    --min-instances=0 \
    --cpu-throttling \
    --execution-environment=gen2

# Set memory and CPU limits
gcloud run services update emotion-detection-api \
    --region=us-central1 \
    --memory=1Gi \
    --cpu=1 \
    --timeout=300
```

### 2. Vertex AI Cost Controls

```bash
# Set Vertex AI endpoint limits
gcloud ai endpoints update 1904603728447537152 \
    --region=us-central1 \
    --min-replica-count=0 \
    --max-replica-count=2 \
    --machine-type=e2-standard-2

# Enable autoscaling with conservative settings
gcloud ai endpoints update 1904603728447537152 \
    --region=us-central1 \
    --autoscaling-metric-specs=cpu-usage=70,request-counts-per-minute=100
```

### 3. Storage Cost Controls

```bash
# Set lifecycle policies for Cloud Storage
gsutil lifecycle set lifecycle-policy.json gs://samo-dl-models

# Lifecycle policy content
cat > lifecycle-policy.json << EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 30,
        "matchesStorageClass": ["NEARLINE", "COLDLINE"]
      }
    }
  ]
}
EOF
```

## Monitoring and Alerting

### 1. Create Monitoring Dashboard

```bash
# Create cost monitoring dashboard
gcloud monitoring dashboards create \
    --project=the-tendril-466607-n8 \
    --config-from-file=dashboards/cost-monitoring-dashboard.json
```

### 2. Dashboard Configuration

```json
{
  "displayName": "GCP Cost Monitoring Dashboard",
  "gridLayout": {
    "columns": "2",
    "widgets": [
      {
        "title": "Daily Cost Trend",
        "xyChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\"billing.googleapis.com/account/amount\"",
                  "aggregation": {
                    "alignmentPeriod": "86400s",
                    "perSeriesAligner": "ALIGN_SUM"
                  }
                }
              }
            }
          ]
        }
      },
      {
        "title": "Service Cost Breakdown",
        "pieChart": {
          "dataSets": [
            {
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "metric.type=\"billing.googleapis.com/account/amount\"",
                  "aggregation": {
                    "alignmentPeriod": "86400s",
                    "perSeriesAligner": "ALIGN_SUM",
                    "crossSeriesReducer": "REDUCE_SUM",
                    "groupByFields": ["resource.labels.service"]
                  }
                }
              }
            }
          ]
        }
      }
    ]
  }
}
```

### 3. Set Up Cost Alerts

```bash
# Create alerting policy for cost overruns
gcloud alpha monitoring policies create \
    --project=the-tendril-466607-n8 \
    --policy-from-file=alerting/cost-alert-policy.json
```

### 4. Alert Policy Configuration

```json
{
  "displayName": "Cost Overrun Alert",
  "conditions": [
    {
      "displayName": "Daily cost exceeds threshold",
      "conditionThreshold": {
        "filter": "metric.type=\"billing.googleapis.com/account/amount\"",
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 5.0,
        "duration": "300s",
        "aggregations": [
          {
            "alignmentPeriod": "86400s",
            "perSeriesAligner": "ALIGN_SUM"
          }
        ]
      }
    }
  ],
  "alertStrategy": {
    "notificationRateLimit": {
      "period": "300s"
    }
  },
  "notificationChannels": [
    "projects/the-tendril-466607-n8/notificationChannels/123456789"
  ]
}
```

## Automated Cost Optimization

### 1. Create Cost Optimization Script

```python
# scripts/cost-optimization/optimize_costs.py
import subprocess
import json
import logging
from datetime import datetime, timedelta

def get_current_costs():
    """Get current day's costs"""
    try:
        result = subprocess.run([
            'gcloud', 'billing', 'accounts', 'list',
            '--format=json'
        ], capture_output=True, text=True)
        
        billing_accounts = json.loads(result.stdout)
        return billing_accounts
    except Exception as e:
        logging.error(f"Error getting costs: {e}")
        return None

def optimize_cloud_run_costs():
    """Optimize Cloud Run costs"""
    try:
        # Scale down during low usage hours
        current_hour = datetime.now().hour
        
        if current_hour < 6 or current_hour > 22:  # Night hours
            subprocess.run([
                'gcloud', 'run', 'services', 'update', 'emotion-detection-api',
                '--region=us-central1',
                '--min-instances=0',
                '--max-instances=1'
            ])
        else:  # Day hours
            subprocess.run([
                'gcloud', 'run', 'services', 'update', 'emotion-detection-api',
                '--region=us-central1',
                '--min-instances=0',
                '--max-instances=3'
            ])
            
        logging.info("Cloud Run costs optimized")
    except Exception as e:
        logging.error(f"Error optimizing Cloud Run: {e}")

def cleanup_unused_resources():
    """Clean up unused resources"""
    try:
        # List and delete unused disks
        result = subprocess.run([
            'gcloud', 'compute', 'disks', 'list',
            '--filter=users=null',
            '--format=json'
        ], capture_output=True, text=True)
        
        unused_disks = json.loads(result.stdout)
        for disk in unused_disks:
            subprocess.run([
                'gcloud', 'compute', 'disks', 'delete', disk['name'],
                '--zone=' + disk['zone'].split('/')[-1],
                '--quiet'
            ])
            
        logging.info("Unused resources cleaned up")
    except Exception as e:
        logging.error(f"Error cleaning up resources: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    optimize_cloud_run_costs()
    cleanup_unused_resources()
```

### 2. Set Up Automated Cost Optimization

```bash
# Create Cloud Scheduler job for cost optimization
gcloud scheduler jobs create http cost-optimization-job \
    --schedule="0 */6 * * *" \
    --uri="https://us-central1-the-tendril-466607-n8.cloudfunctions.net/optimize-costs" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"action": "optimize"}'
```

## Cost Control Scripts

### 1. Budget Monitoring Script

```bash
#!/bin/bash
# scripts/cost-controls/monitor_budget.sh

PROJECT_ID="the-tendril-466607-n8"
BUDGET_AMOUNT=100
ALERT_THRESHOLD=80

# Get current spending
CURRENT_SPEND=$(gcloud billing accounts list \
    --filter="name:0156F5-8F20E3-96A680" \
    --format="value(displayName)" | head -1)

# Calculate percentage
PERCENTAGE=$((CURRENT_SPEND * 100 / BUDGET_AMOUNT))

echo "Current spend: $CURRENT_SPEND USD"
echo "Budget: $BUDGET_AMOUNT USD"
echo "Percentage used: $PERCENTAGE%"

if [ $PERCENTAGE -gt $ALERT_THRESHOLD ]; then
    echo "WARNING: Budget threshold exceeded!"
    # Send alert
    gcloud pubsub topics publish budget-alerts \
        --message="Budget alert: $PERCENTAGE% of budget used"
fi
```

### 2. Resource Cleanup Script

```bash
#!/bin/bash
# scripts/cost-controls/cleanup_resources.sh

PROJECT_ID="the-tendril-466607-n8"
REGION="us-central1"

echo "Cleaning up unused resources..."

# Stop unused Cloud Run services
gcloud run services list --region=$REGION --format="value(name)" | \
while read service; do
    # Check if service has been used in last 24 hours
    LAST_REQUEST=$(gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$service" \
        --limit=1 --format="value(timestamp)" --freshness=24h)
    
    if [ -z "$LAST_REQUEST" ]; then
        echo "Scaling down unused service: $service"
        gcloud run services update $service \
            --region=$REGION \
            --min-instances=0 \
            --max-instances=0
    fi
done

# Delete old logs (older than 30 days)
gcloud logging sinks list --format="value(name)" | \
while read sink; do
    echo "Cleaning old logs for sink: $sink"
    gcloud logging read "timestamp<=\"$(date -d '30 days ago' -u +%Y-%m-%dT%H:%M:%SZ)\"" \
        --limit=1000 --format="value(timestamp)" | \
    xargs -I {} gcloud logging entries delete {}
done

echo "Resource cleanup completed"
```

## Cost Control Policies

### 1. Development Environment Limits

```yaml
# policies/development-limits.yaml
development_limits:
  cloud_run:
    max_instances: 2
    memory: 1Gi
    cpu: 1
    timeout: 300s
  
  vertex_ai:
    max_replicas: 1
    machine_type: e2-standard-2
    min_replicas: 0
  
  storage:
    max_size_gb: 50
    lifecycle_days: 30
  
  compute:
    max_instances: 2
    machine_type: e2-standard-2
```

### 2. Production Environment Limits

```yaml
# policies/production-limits.yaml
production_limits:
  cloud_run:
    max_instances: 10
    memory: 2Gi
    cpu: 2
    timeout: 300s
  
  vertex_ai:
    max_replicas: 5
    machine_type: e2-standard-4
    min_replicas: 1
  
  storage:
    max_size_gb: 200
    lifecycle_days: 90
  
  compute:
    max_instances: 5
    machine_type: e2-standard-4
```

## Implementation Steps

### 1. Immediate Actions (Run Now)

```bash
# 1. Set up budget alerts
./scripts/cost-controls/setup_budget_alerts.sh

# 2. Enable billing export
./scripts/cost-controls/enable_billing_export.sh

# 3. Set resource quotas
./scripts/cost-controls/set_resource_quotas.sh

# 4. Apply cost controls to existing resources
./scripts/cost-controls/apply_cost_controls.sh
```

### 2. Monitoring Setup

```bash
# 1. Create monitoring dashboard
gcloud monitoring dashboards create \
    --project=the-tendril-466607-n8 \
    --config-from-file=dashboards/cost-monitoring-dashboard.json

# 2. Set up alerting policies
gcloud alpha monitoring policies create \
    --project=the-tendril-466607-n8 \
    --policy-from-file=alerting/cost-alert-policy.json

# 3. Create notification channels
gcloud alpha monitoring channels create \
    --display-name="Cost Alerts" \
    --type=email \
    --channel-labels=email_address=your-email@domain.com
```

### 3. Automated Optimization

```bash
# 1. Deploy cost optimization Cloud Function
gcloud functions deploy optimize-costs \
    --runtime=python39 \
    --trigger=http \
    --source=scripts/cost-optimization \
    --entry-point=optimize_costs

# 2. Set up scheduled optimization
gcloud scheduler jobs create http cost-optimization-job \
    --schedule="0 */6 * * *" \
    --uri="https://us-central1-the-tendril-466607-n8.cloudfunctions.net/optimize-costs"
```

## Cost Control Checklist

### ✅ Budget Management
- [ ] Budget alerts configured
- [ ] Billing export enabled
- [ ] Cost monitoring dashboard created
- [ ] Alert policies set up

### ✅ Resource Limits
- [ ] Cloud Run instance limits set
- [ ] Vertex AI replica limits configured
- [ ] Storage lifecycle policies applied
- [ ] Compute engine quotas set

### ✅ Monitoring & Alerting
- [ ] Cost monitoring dashboard active
- [ ] Budget threshold alerts configured
- [ ] Resource usage alerts set up
- [ ] Notification channels configured

### ✅ Automated Optimization
- [ ] Cost optimization script deployed
- [ ] Scheduled cleanup jobs configured
- [ ] Resource scaling policies applied
- [ ] Unused resource cleanup automated

## Cost Control Best Practices

### 1. Regular Monitoring
- Check costs daily during development
- Review weekly cost reports
- Monitor resource usage patterns
- Set up automated alerts

### 2. Resource Optimization
- Use appropriate machine types
- Scale down during low usage
- Clean up unused resources
- Implement lifecycle policies

### 3. Development Practices
- Use development environments with lower limits
- Test with smaller resources
- Clean up test resources promptly
- Monitor costs during development

### 4. Production Practices
- Set appropriate production limits
- Monitor performance vs cost
- Implement auto-scaling policies
- Regular cost optimization reviews

## Emergency Cost Control

### 1. Immediate Cost Reduction

```bash
# Emergency script to reduce costs immediately
./scripts/cost-controls/emergency_cost_reduction.sh
```

### 2. Emergency Script Content

```bash
#!/bin/bash
# scripts/cost-controls/emergency_cost_reduction.sh

echo "EMERGENCY COST REDUCTION - Scaling down all resources"

# Scale down Cloud Run services
gcloud run services list --format="value(name)" | \
while read service; do
    gcloud run services update $service \
        --region=us-central1 \
        --min-instances=0 \
        --max-instances=0
done

# Scale down Vertex AI endpoints
gcloud ai endpoints list --region=us-central1 --format="value(name)" | \
while read endpoint; do
    gcloud ai endpoints update $endpoint \
        --region=us-central1 \
        --min-replica-count=0 \
        --max-replica-count=0
done

# Stop all compute instances
gcloud compute instances list --format="value(name,zone)" | \
while read name zone; do
    gcloud compute instances stop $name --zone=$zone
done

echo "Emergency cost reduction completed"
```

## Summary

This comprehensive cost control setup will help you:

1. **Prevent unexpected charges** with budget alerts and spending limits
2. **Monitor costs in real-time** with dashboards and alerts
3. **Optimize resource usage** with automated scaling and cleanup
4. **Control development costs** with appropriate limits and policies
5. **Respond to emergencies** with immediate cost reduction capabilities

The system is designed to be proactive rather than reactive, helping you maintain control over your GCP spending while ensuring your SAMO Deep Learning project can continue operating effectively. 