# SAMO API - Corrected Request Formats

**⚠️ IMPORTANT: Request Format Updates**

Based on comprehensive testing, the API endpoints require specific request formats that differ from initial assumptions. This document provides the **correct, tested formats** for all endpoints.

---

## 🔐 Authentication

All API endpoints (except `/health` and `/`) require JWT authentication.

### Get Authentication Token

```bash
curl -X POST "https://samo-unified-api-frrnetyhfa-uc.a.run.app/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "email": "user@example.com",
    "password": "YourPassword123!",
    "full_name": "Your Name"
  }'
```

**Response:**
```json
{
  "access_token": "JWT_ACCESS_TOKEN_HERE",
  "refresh_token": "JWT_REFRESH_TOKEN_HERE",
  "token_type": "bearer",
  "expires_in": 1800
}
```

---

## ✅ Emotion Detection - WORKING

**Endpoint:** `/analyze/journal`
**Method:** POST
**Content-Type:** `application/json` ✅
**Status:** Fully functional

### Request Format (JSON)

```bash
curl -X POST "https://samo-unified-api-frrnetyhfa-uc.a.run.app/analyze/journal" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am feeling incredibly happy and excited about this new opportunity!",
    "generate_summary": true,
    "emotion_threshold": 0.1
  }'
```

### Response

```json
{
  "emotion_analysis": {
    "emotions": {
      "joy": 0.61,
      "gratitude": 0.68,
      "love": 0.79,
      "excitement": 0.51
    },
    "primary_emotion": "love",
    "confidence": 0.79,
    "emotional_intensity": "high"
  },
  "summary": {
    "summary": "Generated summary text",
    "compression_ratio": 0.5,
    "emotional_tone": "positive"
  },
  "processing_time_ms": 1200,
  "pipeline_status": {
    "emotion_detection": true,
    "text_summarization": true,
    "voice_processing": false
  }
}
```

---

## 🎤 Voice Transcription - FIXED FORMAT

**Endpoint:** `/transcribe/voice`
**Method:** POST
**Content-Type:** `multipart/form-data` ⚠️ **NOT JSON**
**Status:** Functional with correct format

### ❌ WRONG Format (422 Error)
```bash
# DON'T DO THIS - CAUSES 422 ERROR
curl -X POST "https://samo-unified-api-frrnetyhfa-uc.a.run.app/transcribe/voice" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"audio_file": "...", "language": "en"}'  # ❌ WRONG
```

### ✅ CORRECT Format (multipart/form-data)

```bash
curl -X POST "https://samo-unified-api-frrnetyhfa-uc.a.run.app/transcribe/voice" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "audio_file=@your_audio.wav" \
  -F "language=en" \
  -F "model_size=base" \
  -F "timestamp=false"
```

### Python Example (Correct)

```python
import requests

url = "https://samo-unified-api-frrnetyhfa-uc.a.run.app/transcribe/voice"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

# Prepare multipart form data
files = {
    'audio_file': ('audio.wav', open('audio.wav', 'rb'), 'audio/wav')
}

data = {
    'language': 'en',
    'model_size': 'base',
    'timestamp': 'false'
}

response = requests.post(url, files=files, data=data, headers=headers)
```

### Response

```json
{
  "text": "This is the transcribed audio content",
  "language": "en",
  "confidence": 0.85,
  "duration": 15.4,
  "word_count": 12,
  "speaking_rate": 120.5,
  "audio_quality": "good"
}
```

---

## 📄 Text Summarization - FIXED FORMAT

**Endpoint:** `/summarize/text`
**Method:** POST
**Content-Type:** `application/x-www-form-urlencoded` ⚠️ **NOT JSON**
**Status:** Functional with correct format

### ❌ WRONG Format (422 Error)
```bash
# DON'T DO THIS - CAUSES 422 ERROR
curl -X POST "https://samo-unified-api-frrnetyhfa-uc.a.run.app/summarize/text" \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text to summarize..."}'  # ❌ WRONG
```

### ✅ CORRECT Format (form data)

```bash
curl -X POST "https://samo-unified-api-frrnetyhfa-uc.a.run.app/summarize/text" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "text=This is a long text that needs to be summarized into a shorter version while maintaining the key points and overall meaning." \
  -d "model=t5-small" \
  -d "max_length=150" \
  -d "min_length=30"
```

### Python Example (Correct)

```python
import requests

url = "https://samo-unified-api-frrnetyhfa-uc.a.run.app/summarize/text"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

# Use form data, NOT JSON
data = {
    'text': 'Long text to summarize...',
    'model': 't5-small',
    'max_length': '150',
    'min_length': '30'
}

response = requests.post(url, data=data, headers=headers)
```

### Response

```json
{
  "summary": "Generated summary of the input text",
  "key_emotions": ["neutral"],
  "compression_ratio": 0.7,
  "emotional_tone": "neutral"
}
```

---

## 🎤📝 Voice Journal Analysis - FIXED FORMAT

**Endpoint:** `/analyze/voice-journal`
**Method:** POST
**Content-Type:** `multipart/form-data` ⚠️ **NOT JSON**
**Status:** Functional with correct format

### ✅ CORRECT Format

```bash
curl -X POST "https://samo-unified-api-frrnetyhfa-uc.a.run.app/analyze/voice-journal" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "audio_file=@journal_entry.wav" \
  -F "language=en" \
  -F "generate_summary=true" \
  -F "emotion_threshold=0.1"
```

### Python Example (Correct)

```python
import requests

url = "https://samo-unified-api-frrnetyhfa-uc.a.run.app/analyze/voice-journal"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

files = {
    'audio_file': ('journal.wav', open('journal.wav', 'rb'), 'audio/wav')
}

data = {
    'language': 'en',
    'generate_summary': 'true',
    'emotion_threshold': '0.1'
}

response = requests.post(url, files=files, data=data, headers=headers)
```

---

## 🏥 System Health Endpoints - WORKING

These endpoints work without authentication and use GET requests:

### Health Check
```bash
curl -X GET "https://samo-unified-api-frrnetyhfa-uc.a.run.app/health"
```

### API Information
```bash
curl -X GET "https://samo-unified-api-frrnetyhfa-uc.a.run.app/"
```

### Models Status
```bash
curl -X GET "https://samo-unified-api-frrnetyhfa-uc.a.run.app/models/status"
```

---

## 🚨 Common Error Fixes

### 422 Validation Error
**Cause:** Wrong request format (using JSON instead of form data)
**Fix:** Use the correct Content-Type and request format shown above

### 429 Rate Limit Error
**Cause:** Too many requests too quickly
**Fix:** Implement exponential backoff and respect rate limits

### 401/403 Authentication Error
**Cause:** Missing or invalid JWT token
**Fix:** Get fresh token via `/auth/register` or `/auth/login`

---

## 📊 Rate Limiting (Updated)

**Current Limits (Production-Friendly):**
- **Requests per minute:** 300 (increased from 60)
- **Burst size:** 50 (increased from 10)
- **Concurrent requests:** 20 (increased from 5)
- **Block duration:** 2 minutes (reduced from 5)

**Rate Limit Headers:**
- `X-RateLimit-Limit`: Total allowed requests
- `X-RateLimit-Remaining`: Remaining requests in window
- `Retry-After`: Seconds to wait if rate limited

---

## 🔧 JavaScript/Node.js Examples

### Emotion Detection
```javascript
const response = await fetch('https://samo-unified-api-frrnetyhfa-uc.a.run.app/analyze/journal', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        text: 'I am feeling great today!',
        generate_summary: true
    })
});
```

### Voice Transcription
```javascript
const formData = new FormData();
formData.append('audio_file', audioFile);
formData.append('language', 'en');

const response = await fetch('https://samo-unified-api-frrnetyhfa-uc.a.run.app/transcribe/voice', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`
        // Don't set Content-Type - let browser set it for multipart
    },
    body: formData
});
```

### Text Summarization
```javascript
const params = new URLSearchParams();
params.append('text', 'Long text to summarize...');
params.append('model', 't5-small');

const response = await fetch('https://samo-unified-api-frrnetyhfa-uc.a.run.app/summarize/text', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: params
});
```

---

## ✅ Summary

**Working Correctly:**
- ✅ Emotion Detection (`/analyze/journal`) - JSON format
- ✅ Authentication (`/auth/*`) - JSON format
- ✅ Health endpoints - GET requests

**Fixed Formats:**
- 🔧 Voice Transcription - Use `multipart/form-data`
- 🔧 Text Summarization - Use `application/x-www-form-urlencoded`
- 🔧 Voice Journal Analysis - Use `multipart/form-data`

**Rate Limiting:**
- 🔧 Increased to production-friendly limits
- 🔧 Better error messages and retry guidance

All three core features are now functional with the correct request formats! 🎉