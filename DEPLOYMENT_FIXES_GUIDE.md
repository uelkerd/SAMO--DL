# SAMO API - Deployment Fixes Guide

**🎯 Objective:** Fix the three critical issues identified in comprehensive testing:
1. Voice model loading issue
2. Correct API request formats
3. Production-friendly rate limiting

---

## 🔧 Fix #1: Voice Model Loading Issue

### Problem
Voice processing model shows as `"loaded": false, "status": "unavailable"` in health checks.

### Root Cause
Missing dependencies and system packages in Cloud Run deployment.

### Solution

**Step 1: Update Requirements**
Replace `deployment/cloud-run/requirements.txt` with the new `requirements-full.txt`:

```bash
# Copy the complete requirements file
cp deployment/cloud-run/requirements-full.txt deployment/cloud-run/requirements.txt
```

**Step 2: Update Dockerfile**
Use the new `Dockerfile-full` that includes system dependencies:

```bash
# Use the full Dockerfile with audio processing support
cp deployment/cloud-run/Dockerfile-full deployment/cloud-run/Dockerfile
```

**Key additions:**
- `ffmpeg` and audio processing libraries
- `openai-whisper` Python package
- `pydub` for audio format conversion
- Robust error handling and fallbacks

**Step 3: Deploy Updated Container**
```bash
# Build and deploy with new dependencies
gcloud run deploy samo-unified-api \
  --source deployment/cloud-run/ \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s
```

---

## 📝 Fix #2: API Request Format Documentation

### Problem
Endpoints expected different request formats than documented, causing 422 validation errors.

### Fixed Formats

#### ✅ Emotion Detection (Working)
- **Format:** JSON
- **Endpoint:** `/analyze/journal`
- **Content-Type:** `application/json`

#### 🔧 Voice Transcription (Fixed)
- **Format:** multipart/form-data (NOT JSON)
- **Endpoint:** `/transcribe/voice`
- **Content-Type:** Automatically set by client

```bash
# CORRECT
curl -X POST "https://your-api.com/transcribe/voice" \
  -H "Authorization: Bearer TOKEN" \
  -F "audio_file=@audio.wav" \
  -F "language=en"

# WRONG (causes 422 error)
curl -X POST "https://your-api.com/transcribe/voice" \
  -H "Content-Type: application/json" \
  -d '{"audio_file": "...", "language": "en"}'
```

#### 🔧 Text Summarization (Fixed)
- **Format:** form data (NOT JSON)
- **Endpoint:** `/summarize/text`
- **Content-Type:** `application/x-www-form-urlencoded`

```bash
# CORRECT
curl -X POST "https://your-api.com/summarize/text" \
  -H "Authorization: Bearer TOKEN" \
  -d "text=Text to summarize..." \
  -d "model=t5-small"

# WRONG (causes 422 error)
curl -X POST "https://your-api.com/summarize/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Text to summarize..."}'
```

### Updated Documentation Files
- `docs/api/CORRECTED_API_FORMATS.md` - Comprehensive format guide
- `docs/api/API_DOCUMENTATION.md` - Updated main documentation

---

## ⚡ Fix #3: Rate Limiting Configuration

### Problem
Rate limiting was too aggressive, blocking legitimate testing and usage.

### Previous Limits (Too Restrictive)
- 60 requests/minute
- 10 burst size
- 5 concurrent requests
- 5-minute blocks

### New Production-Friendly Limits
- **300 requests/minute** (5x increase)
- **50 burst size** (5x increase)
- **20 concurrent requests** (4x increase)
- **2-minute blocks** (2.5x reduction)

### Implementation
Rate limiting is now automatically configured with production-friendly settings:

```python
# In unified_ai_api.py
add_rate_limiting(
    app,
    requests_per_minute=300,       # Increased
    burst_size=50,                 # Increased
    max_concurrent_requests=20,    # Increased
    rapid_fire_threshold=30,       # More lenient
    sustained_rate_threshold=600,  # More lenient
)
```

### Environment-Based Scaling
The system automatically detects Cloud Run environment and adjusts limits:

```python
# On Cloud Run (detected automatically):
requests_per_minute=500
burst_size=100

# High performance mode (PERFORMANCE_MODE=high):
requests_per_minute=1000
burst_size=200
```

---

## 🚀 Complete Deployment Process

### Option A: Quick Fix (Recommended)

If you want to quickly fix the existing deployment:

```bash
# 1. Update the unified API with fixes (already done in codebase)
# 2. Deploy the updated code
gcloud run deploy samo-unified-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s \
  --set-env-vars PERFORMANCE_MODE=high
```

### Option B: Full Re-deployment

For a complete clean deployment with all fixes:

```bash
# 1. Build with new Dockerfile
cd deployment/cloud-run/
docker build -f Dockerfile-full -t samo-api-full .

# 2. Push to Google Container Registry
docker tag samo-api-full gcr.io/YOUR-PROJECT/samo-api-full
docker push gcr.io/YOUR-PROJECT/samo-api-full

# 3. Deploy to Cloud Run
gcloud run deploy samo-unified-api \
  --image gcr.io/YOUR-PROJECT/samo-api-full \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s \
  --set-env-vars PERFORMANCE_MODE=high,ENABLE_REGISTRATION=true
```

---

## ✅ Verification Steps

After deployment, verify all fixes:

### 1. Check Voice Model Loading
```bash
curl https://your-api.com/health
# Look for: "voice_processing": {"loaded": true, "status": "available"}
```

### 2. Test Request Formats
```bash
# Test voice transcription (should work, not 422)
curl -X POST "https://your-api.com/transcribe/voice" \
  -H "Authorization: Bearer TOKEN" \
  -F "audio_file=@test.wav" \
  -F "language=en"

# Test text summarization (should work, not 422)
curl -X POST "https://your-api.com/summarize/text" \
  -H "Authorization: Bearer TOKEN" \
  -d "text=Test text to summarize"
```

### 3. Verify Rate Limiting
```bash
# Should allow more requests before rate limiting
for i in {1..50}; do
  curl https://your-api.com/health
done
```

---

## 🎉 Expected Results

After applying all fixes:

### Voice Processing ✅
- Model loaded: `true`
- Status: `"available"`
- Endpoints working: `/transcribe/voice`, `/analyze/voice-journal`

### Request Formats ✅
- No more 422 validation errors
- Correct formats documented and working
- All endpoints accepting expected formats

### Rate Limiting ✅
- More permissive limits for real usage
- Better error messages
- Automatic scaling based on environment

### Overall API Status ✅
- All 3 core features functional
- Production-ready configuration
- Comprehensive documentation updated

---

## 📊 Testing Commands

Run the comprehensive test suite to verify everything works:

```bash
# Run the fixed test suite
python scripts/testing/comprehensive_api_tester.py

# Or run the corrected format tests
python scripts/testing/fixed_api_tester.py
```

Expected results:
- **Authentication:** ✅ Working
- **Emotion Detection:** ✅ Working
- **Voice Transcription:** ✅ Working (with correct format)
- **Text Summarization:** ✅ Working (with correct format)
- **Rate Limiting:** ✅ Production-friendly

---

## 🔄 Rollback Plan

If issues occur, you can rollback:

```bash
# Rollback to previous Cloud Run revision
gcloud run services replace-traffic samo-unified-api --to-revisions=PREVIOUS_REVISION=100
```

---

**🎯 Result:** All three core features working perfectly with production-ready configuration!

**Next Steps:** Monitor the deployment and adjust rate limits based on actual usage patterns.