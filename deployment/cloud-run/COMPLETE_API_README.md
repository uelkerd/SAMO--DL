# 🚀 SAMO Complete AI API Documentation

## Overview

The SAMO Complete AI API provides a comprehensive deep learning pipeline for voice journal analysis, featuring:

- **🎭 Emotion Detection** - Multi-label emotion classification
- **📝 Text Summarization** - T5-based text compression and summarization
- **🎵 Voice Transcription** - OpenAI Whisper-powered speech-to-text
- **🔄 Complete Pipeline** - End-to-end voice journal analysis

## API Endpoints

### Base URL
```
https://emotion-detection-api-frrnetyhfa-uc.a.run.app
```

### Authentication
All endpoints require an API key header:
```
X-API-Key: your-api-key-here
```

---

## 🎭 Emotion Detection (Existing)

### POST `/predict`
Analyze text for emotions.

**Request:**
```json
{
  "text": "Today I received a promotion and I'm really excited!",
  "threshold": 0.1
}
```

**Response:**
```json
{
  "primary_emotion": "joy",
  "confidence": 0.89,
  "emotions": {
    "joy": 0.75,
    "gratitude": 0.65,
    "excitement": 0.45
  },
  "emotional_intensity": "high"
}
```

---

## 📝 Text Summarization (NEW)

### POST `/summarize`
Generate concise summaries using T5 model.

**Request:**
```json
{
  "text": "Your long text here...",
  "max_length": 150,
  "min_length": 30
}
```

**Response:**
```json
{
  "summary": "Condensed version of your text...",
  "original_length": 45,
  "summary_length": 12,
  "compression_ratio": 0.73,
  "processing_time": 0.85
}
```

---

## 🎵 Voice Transcription (NEW)

### POST `/transcribe`
Convert audio files to text using Whisper.

**Supported formats:** MP3, WAV, M4A, AAC, OGG, FLAC
**Max file size:** 45MB

**Request:**
```bash
curl -X POST "https://emotion-detection-api-frrnetyhfa-uc.a.run.app/transcribe" \
  -H "X-API-Key: your-api-key" \
  -F "audio=@your_audio_file.wav" \
  -F "language=en"
```

**Response:**
```json
{
  "text": "Transcribed text from your audio...",
  "language": "en",
  "confidence": 0.95,
  "duration": 15.4,
  "word_count": 23,
  "speaking_rate": 89.6,
  "processing_time": 2.1
}
```

---

## 🔄 Complete Analysis Pipeline (NEW)

### POST `/analyze/complete`
Full pipeline: transcription (if audio) → emotion analysis → summarization.

**Request (Text only):**
```json
{
  "text": "Your journal entry text...",
  "generate_summary": true,
  "emotion_threshold": 0.1
}
```

**Request (Audio + Analysis):**
```bash
curl -X POST "https://emotion-detection-api-frrnetyhfa-uc.a.run.app/analyze/complete" \
  -H "X-API-Key: your-api-key" \
  -F "audio=@journal_entry.wav" \
  -F "generate_summary=true" \
  -F "emotion_threshold=0.1"
```

**Response:**
```json
{
  "transcription": {
    "text": "Transcribed journal entry...",
    "language": "en",
    "confidence": 0.92,
    "duration": 24.5
  },
  "emotion_analysis": {
    "primary_emotion": "gratitude",
    "confidence": 0.87,
    "emotions": {...},
    "emotional_intensity": "moderate"
  },
  "summary": {
    "summary": "Key insights from journal entry...",
    "compression_ratio": 0.68,
    "emotional_tone": "positive"
  },
  "processing_time": 3.2,
  "pipeline_status": {
    "emotion_detection": true,
    "text_summarization": true,
    "voice_processing": true
  }
}
```

---

## 🏥 Health & Monitoring

### GET `/health`
Check API status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "models": {
    "emotion_detection": {
      "loaded": true,
      "status": "available"
    },
    "text_summarization": {
      "loaded": true,
      "status": "available"
    },
    "voice_processing": {
      "loaded": true,
      "status": "available"
    }
  }
}
```

---

## 📊 Rate Limits

- **Per User:** 1,000 requests per minute
- **Burst:** 100 concurrent requests
- **Global:** 50 concurrent requests max

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Set your API key
export API_KEY="your-api-key-here"

# Run tests
python deployment/cloud-run/test_complete_api.py
```

For voice transcription testing, create a `test_audio.wav` file in the same directory.

---

## 🔧 Deployment

The API is deployed on Google Cloud Run with:

- **Automatic scaling** (0-1000 instances)
- **CPU-only PyTorch** for cost optimization
- **Pre-downloaded models** for fast startup
- **Security hardening** and rate limiting
- **Comprehensive monitoring** and logging

---

## 🎯 Use Cases

### Voice Journal Analysis
1. User records voice journal entry
2. API transcribes speech to text
3. API analyzes emotions in the text
4. API generates summary for quick review
5. User gets complete emotional insights

### Text Journal Enhancement
1. User writes text journal entry
2. API analyzes emotional content
3. API generates concise summary
4. User gets emotional insights + key takeaways

### Real-time Emotional Support
1. User shares current emotional state
2. API provides immediate emotional analysis
3. API offers supportive summary
4. User receives empathetic, actionable insights

---

*Built with ❤️ using T5, Whisper, and emotion detection models*
