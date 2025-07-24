# SAMO Deep Learning - Monitoring & Observability Playbook

## 📋 Overview

This playbook defines the monitoring strategy for SAMO Deep Learning systems, including metrics to track, dashboards, alerting rules, and incident response procedures. Following these guidelines ensures consistent monitoring across all environments and enables rapid detection and resolution of issues.

## 🎯 Key Metrics

### System Health Metrics

| Metric | Description | Warning Threshold | Critical Threshold | Action |
|--------|-------------|-------------------|-------------------|--------|
| CPU Usage | Percentage of CPU utilized | >70% for 5m | >85% for 5m | Scale up or optimize code |
| Memory Usage | Percentage of memory utilized | >75% for 5m | >90% for 2m | Check for memory leaks, scale up |
| Disk Usage | Percentage of disk space utilized | >75% | >90% | Clean up logs, add storage |
| Network I/O | Bytes in/out per second | >80% capacity | >95% capacity | Investigate traffic patterns |

### Application Metrics

| Metric | Description | Warning Threshold | Critical Threshold | Action |
|--------|-------------|-------------------|-------------------|--------|
| Request Rate | Requests per second | >1000 rps | >1500 rps | Check for unusual traffic patterns |
| Response Time | Average response time in ms | >200ms | >500ms | Optimize code, scale up |
| Error Rate | Percentage of requests resulting in errors | >1% | >5% | Investigate error causes |
| Queue Length | Number of requests waiting to be processed | >100 | >500 | Scale up workers |

### Model Performance Metrics

| Metric | Description | Warning Threshold | Critical Threshold | Action |
|--------|-------------|-------------------|-------------------|--------|
| Inference Latency | Time to generate predictions | >100ms | >300ms | Optimize model, consider quantization |
| Prediction Confidence | Average confidence score | <0.7 | <0.5 | Retrain model, adjust thresholds |
| F1 Score (Online) | F1 score from feedback | <0.75 | <0.65 | Investigate data drift, retrain |
| Data Drift | Distribution shift from training data | >10% | >20% | Collect new training data, retrain |

### Business Metrics

| Metric | Description | Warning Threshold | Critical Threshold | Action |
|--------|-------------|-------------------|-------------------|--------|
| API Usage | Total API calls per day | <1000 | <500 | Investigate user engagement |
| Unique Users | Number of unique users per day | <100 | <50 | Check for service issues |
| Successful Analyses | Percentage of analyses marked helpful | <80% | <60% | Investigate quality issues |
| Feature Usage | Usage distribution across features | >20% change | >40% change | Investigate user behavior shift |

## 📊 Dashboards

### System Dashboard

![System Dashboard](https://example.com/system-dashboard.png)

**Key Panels:**
- CPU, Memory, Disk, Network usage over time
- Container health status
- System logs frequency by severity
- Host metrics across the cluster

**Access:** [Grafana System Dashboard](https://grafana.samo.ai/dashboards/system)

### Application Dashboard

![Application Dashboard](https://example.com/application-dashboard.png)

**Key Panels:**
- Request rate and latency by endpoint
- Error rate and status code distribution
- Database query performance
- Cache hit/miss ratio
- Endpoint usage heatmap

**Access:** [Grafana Application Dashboard](https://grafana.samo.ai/dashboards/application)

### Model Performance Dashboard

![Model Dashboard](https://example.com/model-dashboard.png)

**Key Panels:**
- Inference latency distribution
- Prediction confidence histogram
- F1 score trend over time
- Feature importance
- Data drift indicators
- Model version comparison

**Access:** [Grafana Model Dashboard](https://grafana.samo.ai/dashboards/model)

### Business Metrics Dashboard

![Business Dashboard](https://example.com/business-dashboard.png)

**Key Panels:**
- Daily active users
- API usage by endpoint
- User satisfaction metrics
- Feature adoption rates
- Conversion funnel

**Access:** [Grafana Business Dashboard](https://grafana.samo.ai/dashboards/business)

## 🚨 Alerting Rules

### System Alerts

```yaml
# prometheus/alerts/system.yaml
groups:
- name: system
  rules:
  - alert: HighCPUUsage
    expr: avg(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (instance) > 0.85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage on {{ $labels.instance }}"
      description: "CPU usage is above 85% for 5 minutes."
      runbook_url: "https://wiki.samo.ai/runbooks/high-cpu-usage"

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"
      description: "Memory usage is above 90% for 2 minutes."
      runbook_url: "https://wiki.samo.ai/runbooks/high-memory-usage"
```

### Application Alerts

```yaml
# prometheus/alerts/application.yaml
groups:
- name: application
  rules:
  - alert: HighErrorRate
    expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate"
      description: "Error rate is above 5% for 2 minutes."
      runbook_url: "https://wiki.samo.ai/runbooks/high-error-rate"

  - alert: SlowResponses
    expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow API responses"
      description: "95th percentile of response time is above 500ms for 5 minutes."
      runbook_url: "https://wiki.samo.ai/runbooks/slow-responses"
```

### Model Performance Alerts

```yaml
# prometheus/alerts/model.yaml
groups:
- name: model
  rules:
  - alert: LowF1Score
    expr: samo_model_f1_score < 0.65
    for: 30m
    labels:
      severity: critical
    annotations:
      summary: "Low F1 score detected"
      description: "Model F1 score is below 0.65 for 30 minutes."
      runbook_url: "https://wiki.samo.ai/runbooks/low-f1-score"

  - alert: HighInferenceLatency
    expr: histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le, model)) > 0.3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High inference latency for {{ $labels.model }}"
      description: "95th percentile of inference time is above 300ms for 5 minutes."
      runbook_url: "https://wiki.samo.ai/runbooks/high-inference-latency"
```

## 📝 Logging Strategy

### Log Levels

| Level | When to Use | Examples |
|-------|-------------|----------|
| ERROR | Application failures requiring immediate attention | Database connection failures, API errors |
| WARNING | Potential issues that don't stop functionality | Slow database queries, retry attempts |
| INFO | Normal but significant events | API requests, model loading |
| DEBUG | Detailed information for debugging | Request payloads, model inputs/outputs |

### Structured Logging Format

```json
{
  "timestamp": "2025-07-23T14:30:12.123Z",
  "level": "INFO",
  "service": "emotion-detection-api",
  "trace_id": "abc123def456",
  "message": "Processed emotion detection request",
  "request_id": "req-789",
  "user_id": "user-456",
  "processing_time_ms": 120,
  "model_version": "v1.2.3",
  "additional_context": {
    "text_length": 156,
    "emotions_detected": ["joy", "surprise"]
  }
}
```

### Log Collection and Storage

- **Collection**: Fluentd agents on each node
- **Processing**: Fluentd filters for parsing and enrichment
- **Storage**: Elasticsearch (7 days hot storage, 30 days warm, 90 days cold)
- **Visualization**: Kibana dashboards

## 🔍 Tracing

### Distributed Tracing

All services implement OpenTelemetry for distributed tracing:

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Set up the tracer
resource = Resource(attributes={SERVICE_NAME: "emotion-detection-api"})
provider = TracerProvider(resource=resource)
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent",
    agent_port=6831,
)
processor = BatchSpanProcessor(jaeger_exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# Example usage in API endpoint
@app.route("/emotions/analyze", methods=["POST"])
def analyze_emotions():
    with tracer.start_as_current_span("analyze_emotions") as span:
        span.set_attribute("user.id", request.headers.get("X-User-ID"))
        
        # Process text
        with tracer.start_as_current_span("preprocess_text") as preprocess_span:
            text = request.json.get("text")
            preprocess_span.set_attribute("text.length", len(text))
            processed_text = preprocess_text(text)
        
        # Run model inference
        with tracer.start_as_current_span("model_inference") as inference_span:
            emotions = model.predict(processed_text)
            inference_span.set_attribute("emotions.count", len(emotions))
        
        return jsonify({"emotions": emotions})
```

### Key Spans to Trace

1. **API Request Processing**
   - Request parsing
   - Authentication/Authorization
   - Input validation

2. **Model Operations**
   - Text preprocessing
   - Model loading
   - Inference
   - Postprocessing

3. **Database Operations**
   - Query execution
   - Transaction boundaries

4. **External Service Calls**
   - Third-party API requests
   - Cache operations

## 🚑 Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P0 | Complete service outage | Immediate (24/7) | API completely down, data loss |
| P1 | Severe degradation | <30 minutes | >50% error rate, critical feature broken |
| P2 | Partial degradation | <2 hours | Slow responses, non-critical feature broken |
| P3 | Minor issues | Next business day | UI glitches, isolated errors |

### Incident Response Process

1. **Detection**
   - Alert triggered or issue reported
   - On-call engineer acknowledges

2. **Assessment**
   - Determine severity
   - Create incident channel in Slack
   - Notify stakeholders based on severity

3. **Mitigation**
   - Implement immediate fixes (rollback, scale up, etc.)
   - Update status page

4. **Resolution**
   - Implement permanent fix
   - Verify monitoring
   - Close incident

5. **Post-Mortem**
   - Document root cause
   - Identify preventive measures
   - Update runbooks

### Incident Communication Template

```
Incident: [Brief description]
Severity: [P0-P3]
Status: [Investigating/Mitigating/Resolved]

Impact:
- [Service/feature affected]
- [Users affected]
- [Business impact]

Timeline:
- [Time] - [Event]
- [Time] - [Event]

Current Actions:
- [What's being done now]

Next Update:
- [When to expect the next update]
```

## 📈 Performance Testing

### Load Testing

Regular load tests are conducted using Locust:

```python
# locustfile.py
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(3)
    def analyze_emotions(self):
        self.client.post("/emotions/analyze", 
                         json={"text": "I feel so happy about this achievement!"})
    
    @task(1)
    def summarize_text(self):
        self.client.post("/summarize", 
                         json={"text": "Long text to summarize...", "max_length": 100})
```

Run with:
```bash
locust -f locustfile.py --host=https://api.samo.ai --users 100 --spawn-rate 10
```

### Stress Testing

Monthly stress tests to determine breaking points:

```bash
# Run with increasing load until failure
for i in {100..1000..100}
do
  echo "Testing with $i users..."
  locust -f locustfile.py --host=https://staging.api.samo.ai --users $i --spawn-rate 20 --run-time 10m --headless --csv=results_$i
done
```

## 🔄 Continuous Monitoring Improvement

### Monitoring Review Cycle

1. **Weekly**: Review dashboards and alerts, adjust thresholds
2. **Monthly**: Analyze alert patterns, reduce false positives
3. **Quarterly**: Comprehensive monitoring review, add new metrics

### Monitoring as Code

All monitoring configurations are stored in version control:

```
monitoring/
├── dashboards/
│   ├── system.json
│   ├── application.json
│   ├── model.json
│   └── business.json
├── alerts/
│   ├── system.yaml
│   ├── application.yaml
│   └── model.yaml
├── logging/
│   ├── fluentd-config.yaml
│   └── log-retention-policy.yaml
└── tracing/
    └── opentelemetry-config.yaml
```

Deploy with:
```bash
# Apply Prometheus configuration
kubectl apply -f monitoring/alerts/

# Update Grafana dashboards
grafana-cli dashboard import monitoring/dashboards/system.json
```

## 📚 Runbooks

### High CPU Usage Runbook

1. **Check system load**
   ```bash
   kubectl top pods -n samo-dl
   ```

2. **Identify resource-intensive processes**
   ```bash
   kubectl exec -it <pod-name> -n samo-dl -- top -o cpu
   ```

3. **Check for traffic spikes**
   - Review request rate in Grafana dashboard
   - Check for unusual traffic patterns

4. **Mitigation options**
   - Scale horizontally: `kubectl scale deployment samo-dl-api -n samo-dl --replicas=5`
   - Implement rate limiting
   - Optimize code hotspots

### Low F1 Score Runbook

1. **Verify the issue**
   - Check F1 score metrics in Grafana
   - Compare with historical baseline

2. **Analyze recent data**
   ```python
   # scripts/analyze_model_performance.py
   from src.models.emotion_detection.dataset_loader import GoEmotionsDataLoader
   from src.models.emotion_detection.bert_classifier import BERTEmotionClassifier
   
   # Load recent data and evaluate
   loader = GoEmotionsDataLoader()
   recent_data = loader.load_recent_data()
   model = BERTEmotionClassifier.load_from_checkpoint("models/checkpoints/best_model.pt")
   metrics = model.evaluate(recent_data)
   print(f"F1 Score: {metrics['f1_score']}")
   print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
   ```

3. **Check for data drift**
   - Compare distribution of recent data vs. training data
   - Look for new patterns or categories

4. **Mitigation options**
   - Adjust prediction threshold
   - Retrain with recent data
   - Roll back to previous model version

## 🔒 Security Monitoring

### Security Metrics

| Metric | Description | Warning Threshold | Critical Threshold |
|--------|-------------|-------------------|-------------------|
| Failed Auth Attempts | Number of failed authentication attempts | >10 in 5m | >50 in 5m |
| Unusual Access Patterns | Requests from new locations/devices | N/A | N/A |
| Rate Limiting Triggers | Number of rate limit hits | >100 in 1h | >500 in 1h |

### Security Alerts

```yaml
# prometheus/alerts/security.yaml
groups:
- name: security
  rules:
  - alert: HighFailedAuthRate
    expr: sum(rate(auth_failures_total[5m])) > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High rate of authentication failures"
      description: "More than 10 authentication failures per minute for 5 minutes."
      runbook_url: "https://wiki.samo.ai/runbooks/auth-failures"

  - alert: AnomalousAPIUsage
    expr: sum(rate(http_requests_total{status="429"}[5m])) > 50
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Anomalous API usage detected"
      description: "High rate of rate-limited requests for 5 minutes."
      runbook_url: "https://wiki.samo.ai/runbooks/anomalous-usage"
``` 