# 🚀 HuggingFace Deployment Checklist

Based on practical deployment recommendations for DistilBERT emotion models.

## Pre-Upload Checklist

### 📁 Required Files
- [ ] **Model file**: `model.safetensors` (preferred) or `pytorch_model.bin`
- [ ] **Config**: `config.json` with proper `id2label`/`label2id` mappings
- [ ] **Tokenizer files**:
  - [ ] `tokenizer.json`
  - [ ] `tokenizer_config.json`
  - [ ] Vocabulary files (if needed)
- [ ] **README.md** with proper metadata

### 🏷️ Model Card Metadata (Critical for Serverless API)
```yaml
pipeline_tag: text-classification
library_name: transformers
labels: ["anxious", "calm", "content", "excited", "frustrated", "grateful", "happy", "hopeful", "overwhelmed", "proud", "sad", "tired"]
```

### 🔧 Git LFS Setup
```bash
# Track large files (>100MB)
git lfs track "*.bin"
git lfs track "*.safetensors"
git lfs track "*.onnx"
git lfs track "*.pkl"
git lfs track "*.pth"
```

## Privacy & Security Decision

### 📊 Public Repository (Recommended Start)
**Choose if:**
- [ ] Content is general emotion analysis
- [ ] No sensitive/health data involved
- [ ] Want completely free hosting
- [ ] Easy integration and sharing

**Benefits:**
- ✅ Free unlimited storage/bandwidth
- ✅ No token required for public access
- ✅ Better community discovery

### 🔒 Private Repository
**Choose if:**
- [ ] Journal content includes mental health data
- [ ] Therapy/counseling applications
- [ ] PII (personally identifiable information)
- [ ] Compliance requirements (HIPAA, etc.)

**Requirements:**
- ✅ HF token required for all access
- ✅ Free tier with quotas, paid plans available
- ✅ More secure for sensitive applications

## Deployment Strategy Selection

### 🆓 Start Here: Serverless API (Free)
**Default choice unless you have specific needs**

**Working Defaults:**
- **Traffic**: 1-5 RPS initially
- **Latency**: Plan for p95 < 800ms (includes cold starts)
- **Budget**: $0 to start

**When to upgrade:**
- [ ] Hit rate limits consistently
- [ ] Cold start delays impact user experience
- [ ] Need > 5 RPS sustained traffic
- [ ] Require <200ms consistent latency

### 🚀 Production: Inference Endpoints (Paid)
**Upgrade when you need predictable performance**

**Benefits:**
- ✅ No cold starts
- ✅ Consistent latency
- ✅ VPC options for security
- ✅ Custom containers if needed

**Costs:**
- 💰 CPU: ~$0.06-0.24/hour
- 💰 GPU: ~$0.60-1.20/hour

### 🏠 Enterprise: Self-Hosted
**For maximum control or compliance**

**Choose when:**
- [ ] Strict data residency requirements
- [ ] Custom inference optimizations needed
- [ ] High volume makes endpoints expensive
- [ ] Complete control over infrastructure

## Common Pitfalls Checklist

### ❌ Upload Issues
- [ ] **Missing tokenizer files** → Serverless API can't load model
- [ ] **No pipeline_tag** → Auto-detection fails
- [ ] **Large weights without LFS** → Push failures
- [ ] **Wrong label mappings** → Client-side mapping breaks

### ❌ Runtime Issues
- [ ] **Token not set** → Authentication failures
- [ ] **Wrong endpoint URL** → 404 errors
- [ ] **Expecting wrong output format** → Parsing failures
- [ ] **No error handling** → Poor user experience

## Pre-Production Testing

### 🧪 Serverless API Test
```python
import requests
import os

url = "https://api-inference.huggingface.co/models/your-username/samo-dl-emotion-model"
headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

# Test cases
test_cases = [
    "I felt calm after writing it all down.",
    "I am frustrated but hopeful.",
    "Today was overwhelming but I'm proud of getting through it.",
    ""  # Edge case: empty input
]

for text in test_cases:
    payload = {"inputs": text}
    response = requests.post(url, headers=headers, json=payload)
    print(f"Input: {text}")
    print(f"Status: {response.status_code}")
    print(f"Output: {response.json()}")
    print("-" * 50)
```

### 📊 Expected Output Validation
```json
[
  {
    "label": "calm",
    "score": 0.8234
  },
  {
    "label": "hopeful",
    "score": 0.1123
  }
]
```

**Validate:**
- [ ] Output is list of objects with `label` and `score`
- [ ] Labels match your trained emotion set
- [ ] Scores are probabilities (0-1)
- [ ] Highest score corresponds to expected emotion

## Security Setup

### 🔑 Token Management
```bash
# Development
export HF_TOKEN='hf_your_token_here'

# Production (never commit tokens!)
# Use environment variables or secret management
```

### 🛡️ For Sensitive Data
- [ ] Use private repository
- [ ] Consider Inference Endpoints over Serverless
- [ ] Implement client-side encryption if needed
- [ ] Log minimal information for debugging
- [ ] Regular security audits

## Performance Monitoring

### 📈 Key Metrics to Track
- [ ] **Response time** (p50, p95, p99)
- [ ] **Error rate** (4xx, 5xx responses)
- [ ] **Cold start frequency** (Serverless only)
- [ ] **Token usage** (if rate-limited)
- [ ] **Prediction accuracy** (spot-check results)

### 🔍 Health Checks
```python
def health_check():
    """Validate API is working correctly"""
    test_input = "I am feeling happy today"
    # Call your API
    # Validate response structure
    # Check latency
    # Return status
```

## Cost Optimization

### 💰 Serverless API (Free Tier)
- ✅ Start here for development
- ✅ Good for < 5 RPS sustained
- ⚠️ Watch rate limits and cold starts

### 💰 Inference Endpoints (Paid)
- 🎯 CPU instances for most text classification
- 🎯 GPU only if you need <100ms latency
- 🎯 Scale down/up based on traffic patterns
- 🎯 Monitor costs weekly

### 💰 Budget Planning
| Usage Level | Recommended | Monthly Cost |
|-------------|-------------|--------------|
| **Development** | Serverless | $0 |
| **Small prod** | CPU Endpoint | ~$50-150 |
| **Large prod** | GPU Endpoint | ~$200-500 |
| **Enterprise** | Self-hosted | Your infra |

## Launch Checklist

### 🚀 Pre-Launch (Final Steps)
- [ ] Model uploaded and validated
- [ ] Test with actual journal entries
- [ ] Error handling implemented
- [ ] Monitoring set up
- [ ] Security tokens configured
- [ ] Documentation updated
- [ ] Rollback plan ready

### 🚀 Launch Day
- [ ] Start with Serverless API (lowest risk)
- [ ] Monitor response times and errors
- [ ] Have team ready for quick issues
- [ ] Gradual traffic ramp-up

### 🚀 Post-Launch (First Week)
- [ ] Daily monitoring of metrics
- [ ] User feedback collection
- [ ] Performance analysis
- [ ] Cost tracking
- [ ] Plan Inference Endpoint upgrade if needed

## Troubleshooting Quick Reference

| Issue | Likely Cause | Quick Fix |
|-------|--------------|-----------|
| Model not found | Wrong repo name | Check `MODEL_NAME` env var |
| 401 Unauthorized | Missing/wrong token | Verify `HF_TOKEN` |
| 503 Service Unavailable | Cold start | Wait 30s, retry |
| Wrong labels in output | Missing id2label mapping | Check config.json |
| Slow responses | Using Serverless | Upgrade to Endpoint |
| Rate limit errors | Too many requests | Implement backoff or upgrade |

---

## 📋 Final Validation

Before going live, ensure:
- [ ] ✅ All files validated and uploaded
- [ ] ✅ Privacy settings match data sensitivity
- [ ] ✅ Test API calls return expected format
- [ ] ✅ Error handling works properly
- [ ] ✅ Monitoring is active
- [ ] ✅ Team knows how to troubleshoot issues
- [ ] ✅ Rollback plan documented

**Ready for production!** 🎉
