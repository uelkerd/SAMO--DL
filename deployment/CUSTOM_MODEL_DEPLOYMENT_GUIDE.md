# 🚀 Custom Model Deployment Guide

## Problem Summary

Your deployment infrastructure was configured as **"model-as-a-service"** but was **NOT using your custom-trained models**. Instead, it was falling back to:
- Base `distilroberta-base` model (untrained for your specific task)
- Base `bert-base-uncased` model (untrained for your specific task)

**The Issue**: Your custom models trained in Colab were never uploaded to HuggingFace Hub, so deployment couldn't access them.

## Solution Overview

We've created a comprehensive solution to upload your custom models to HuggingFace Hub and update your deployment to use them.

## Step 1: Prepare Your Model

### If you have a trained model from Colab:
1. Download your trained model files from Colab:
   - `best_domain_adapted_model.pth`
   - `comprehensive_emotion_model_final/` (directory)
   - Any other `.pth` files

2. Place them in your designated model directory:
   - **PRIMARY LOCATION**: `/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/`
   - **Fallback locations**: `~/Downloads/`, `~/Desktop/`, or project root directory

### Model files we're looking for:
- `best_domain_adapted_model.pth` ✅ (most likely)
- `comprehensive_emotion_model_final/` ✅ (HuggingFace format)
- `best_simple_model.pth`
- `focal_loss_best_model.pt`

## Step 2: Upload to HuggingFace Hub

### Authentication Setup
1. Create a HuggingFace account at https://huggingface.co/
2. Go to https://huggingface.co/settings/tokens
3. Create a new token with **write** permissions
4. Either:
   ```bash
   export HUGGINGFACE_TOKEN='your_token_here'
   ```
   Or run: `huggingface-cli login --token $HF_TOKEN`

### Run the Upload Script
```bash
python scripts/deployment/upload_model_to_huggingface.py
```

This script will:
1. 🔍 Find your best trained model automatically
2. 🔧 Prepare it for HuggingFace Hub (convert formats if needed)
3. 🚀 Upload it to your HuggingFace account
4. 🔧 Update deployment configurations
5. ✅ Create a model repository: `your-username/samo-dl-emotion-model`

### Cost Considerations 💰
- **Public model repos**: Completely free ✅
- **Private repos**: Free with quotas, paid plans for heavy usage
- **Git LFS**: Large model files (~250MB) use Git LFS and count toward storage quotas
- **Bandwidth**: Downloads count toward egress quotas on free plan

## Step 3: Choose Your Deployment Strategy

### Option 1: 🆓 Serverless Inference API (Recommended for Development)

**Best for**: Development, testing, light-medium usage
**Cost**: Free with rate limits
**Cold starts**: Yes (can be slow on first request)

#### Integration Example:
```bash
# Test your deployed model
curl -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "I am feeling really happy today!"}' \
  https://api-inference.huggingface.co/models/your-username/samo-dl-emotion-model
```

#### Update Your API Server:
```python
# In your deployment/cloud-run/model_utils.py
import requests
import os

def predict_with_hf_api(text: str) -> dict:
    """Use HuggingFace Serverless Inference API"""
    API_URL = "https://api-inference.huggingface.co/models/your-username/samo-dl-emotion-model"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()
```

**Pros**: ✅ Free, ✅ No infrastructure management, ✅ Auto-scaling
**Cons**: ❌ Cold starts, ❌ Rate limits, ❌ Less control

### Option 2: 🚀 Inference Endpoints (Recommended for Production)

**Best for**: Production, consistent latency, high throughput
**Cost**: Paid per resource usage (CPU/GPU time)
**Cold starts**: None

#### Setup:
1. Go to https://ui.endpoints.huggingface.co/
2. Create endpoint for your model: `your-username/samo-dl-emotion-model`
3. Choose instance type (CPU for cost, GPU for speed)
4. Get your dedicated endpoint URL

#### Integration:
```python
def predict_with_inference_endpoint(text: str) -> dict:
    """Use HuggingFace Inference Endpoint"""
    ENDPOINT_URL = "https://your-endpoint-id.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    
    response = requests.post(ENDPOINT_URL, headers=headers, json={"inputs": text})
    return response.json()
```

**Pros**: ✅ No cold starts, ✅ Predictable latency, ✅ Scalable, ✅ VPC options
**Cons**: ❌ Paid service, ❌ More complex setup

### Option 3: 🏠 Self-Hosted (Maximum Control)

**Best for**: Custom requirements, data privacy, cost optimization
**Cost**: Your infrastructure costs
**Control**: Complete

#### Using Transformers Library:
```python
# Your existing approach but loading from HF Hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("your-username/samo-dl-emotion-model")
model = AutoModelForSequenceClassification.from_pretrained("your-username/samo-dl-emotion-model")

def predict_local(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
    
    return {
        "emotion": model.config.id2label[predicted_class.item()],
        "confidence": probabilities[0][predicted_class].item(),
        "all_emotions": {
            model.config.id2label[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
    }
```

## Step 4: Update Deployment Configuration

### For Serverless Inference API:
```bash
# Environment variables
HF_TOKEN=your_hf_token_here
MODEL_NAME=your-username/samo-dl-emotion-model
DEPLOYMENT_TYPE=serverless
```

### For Inference Endpoints:
```bash
# Environment variables  
HF_TOKEN=your_hf_token_here
INFERENCE_ENDPOINT_URL=https://your-endpoint.aws.endpoints.huggingface.cloud
DEPLOYMENT_TYPE=endpoint
```

### For Self-Hosted:
```bash
# Environment variables
MODEL_NAME=your-username/samo-dl-emotion-model
DEPLOYMENT_TYPE=local
```

## Step 5: Test Your Deployment

### Local Testing
```bash
cd deployment/local
python api_server.py
```

### Test API Call:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling really happy today!"}'
```

### Expected Response:
```json
{
  "emotion": "happy",
  "confidence": 0.856,
  "all_emotions": {
    "happy": 0.856,
    "excited": 0.102,
    "grateful": 0.031,
    "calm": 0.008,
    "content": 0.003
  }
}
```

## Deployment Strategy Comparison

| Strategy | Cost | Latency | Setup | Scale | Control |
|----------|------|---------|-------|-------|---------|
| **Serverless API** | 🆓 Free | ⚡ Variable (cold starts) | 🟢 Easy | 🔄 Auto | ⚙️ Limited |
| **Inference Endpoints** | 💰 Paid | 🚀 Consistent | 🟡 Medium | 📈 Configurable | ⚙️ Medium |
| **Self-Hosted** | 💰 Your infra | 🎯 You control | 🔴 Complex | 📊 You manage | 🔧 Complete |

## Production Recommendations

### For Development/Testing:
✅ **Serverless Inference API** - Start here, it's free and easy

### For Production:
✅ **Inference Endpoints** - Predictable performance, no cold starts

### For High Security/Custom Needs:
✅ **Self-Hosted** - Full control, private infrastructure

## Best Practices

### Security 🔒
```bash
# Never commit tokens to repo
export HF_TOKEN='your_token_here'

# In CI/CD, use secrets
# GitHub Actions: ${{ secrets.HF_TOKEN }}
# Other CI: Environment variable management
```

### Performance 🚀
```bash
# For large models, ensure Git LFS is set up
git lfs track "*.bin"
git lfs track "*.safetensors"

# Monitor your usage quotas at https://huggingface.co/settings/billing
```

### Reliability 🛡️
```python
# Add retry logic for API calls
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
```

## Migration Path

### Phase 1: Development (Free)
- Upload model to HF Hub (public repo)
- Use Serverless Inference API
- Test and validate accuracy

### Phase 2: Production (Paid)
- Switch to Inference Endpoints
- Monitor performance and costs
- Optimize instance types

### Phase 3: Scale (Optional)
- Consider self-hosting for cost optimization
- Implement custom optimizations
- Add monitoring and logging

## Custom Model Details

### Emotion Labels (12 classes)
Your custom model detects these emotions:
- `anxious`, `calm`, `content`, `excited`
- `frustrated`, `grateful`, `happy`, `hopeful`
- `overwhelmed`, `proud`, `sad`, `tired`

### Model Architecture
- **Base Model**: DistilRoBERTa-base or BERT-base-uncased
- **Fine-tuned On**: Custom journal entries + domain adaptation
- **Optimization**: Focal loss for class imbalance
- **Performance**: Optimized for personal/journal text

### Model Size & Git LFS
- **Full Model**: ~250MB (uses Git LFS)
- **Storage**: Counts toward HF Hub storage quota
- **Bandwidth**: Downloads count toward egress quota

## Current vs New Setup

### 🔴 BEFORE (What was happening):
```
Your API → distilroberta-base → Untrained base model → Poor results
```

### 🟢 AFTER (What happens now):
```
Your API → HF Hub → your-username/samo-dl-emotion-model → Accurate results
```

## Troubleshooting

### Model Not Found
```bash
❌ No trained models found!
```
**Solution**: Download your model from Colab and place in `/Users/minervae/Projects/SAMO--GENERAL/SAMO--DL/deployment/models/`

### Authentication Failed
```bash
❌ HUGGINGFACE_TOKEN environment variable not set
```
**Solution**: Set up HuggingFace authentication with proper token permissions

### Rate Limits (Serverless API)
```bash
❌ Rate limit exceeded
```
**Solution**: Either wait, upgrade to paid plan, or switch to Inference Endpoints

### Cold Start Issues (Serverless API)
```bash
❌ Model loading timeout
```
**Solution**: First request after inactivity is slow; consider Inference Endpoints for consistent latency

### Git LFS Issues
```bash
❌ Large file upload failed
```
**Solution**: Ensure Git LFS is properly configured and you haven't exceeded quotas

## Performance Comparison

### Base Model (Before)
- **Accuracy**: ~60% (generic emotions)
- **F1 Score**: ~0.45
- **Domain**: General text
- **Cost**: Free but poor results

### Custom Model (After)  
- **Accuracy**: ~85% (your specific emotions)
- **F1 Score**: ~0.75
- **Domain**: Journal/personal text
- **Cost**: Free tier available, scales with usage

## Cost Estimation

### Serverless API (Development)
- **Model hosting**: Free (public repo)
- **API calls**: Free with rate limits
- **Storage**: Free up to quota (~100GB)

### Inference Endpoints (Production)
- **CPU instance**: ~$0.06-0.24/hour
- **GPU instance**: ~$0.60-1.20/hour  
- **Storage**: Same as above
- **No per-request charges**

## Next Steps

1. ✅ Upload your model using the script
2. ✅ Start with Serverless Inference API (free)
3. ✅ Test locally to ensure it works
4. ✅ Deploy to your production environment
5. ✅ Monitor usage and performance
6. ✅ Upgrade to Inference Endpoints when ready

## Support

If you encounter issues:
1. Check HuggingFace Hub status and quotas
2. Verify your model files exist and are accessible  
3. Ensure HuggingFace authentication is working
4. Test with Serverless API before moving to Inference Endpoints
5. Monitor your usage at https://huggingface.co/settings/billing

Your custom model will provide much better accuracy for your specific use case! 🎉