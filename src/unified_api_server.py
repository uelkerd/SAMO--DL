#!/usr/bin/env python3
"""
SAMO-DL Unified API Server
A production-ready Flask API server for the SAMO Deep Learning platform.
Integrates emotion detection, voice transcription, and text summarization.
"""

import os
import yaml
import logging
from flask import Flask, request, jsonify, Blueprint
from flask_restx import Api, Resource, fields
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import librosa
import ffmpeg
import io
import structlog

# Load environment variables
load_dotenv()

# Configure logging
log = structlog.get_logger()
logging.basicConfig(level=logging.INFO)

# Load configuration
with open('configs/samo_api_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Flask app
app = Flask(__name__)
api = Api(app, title='SAMO-DL API', version=config['api']['version'])

# API Namespaces
ns_emotion = api.namespace('emotion', description='Emotion Detection Endpoints')
ns_transcribe = api.namespace('transcribe', description='Voice Transcription Endpoints')
ns_summarize = api.namespace('summarize', description='Text Summarization Endpoints')
ns_health = api.namespace('health', description='Health Check Endpoints')

# Models
emotion_pipeline = None
whisper_pipeline = None
t5_pipeline = None

# Request model for predictions
class PredictionRequest(BaseModel):
    text: str

class BatchPredictionRequest(BaseModel):
    texts: list[str]

def load_models():
    global emotion_pipeline, whisper_pipeline, t5_pipeline
    
    # Load Emotion Detection Model
    if config['models']['emotion']['provider'] == 'hf':
        emotion_pipeline = pipeline(
            "text-classification",
            model=config['models']['emotion']['model_name'],
            local_files_only=config['models']['emotion']['local_only']
        )
        log.info("Emotion detection model loaded successfully")
    
    # Load Whisper Model (simplified for API)
    if config['models']['whisper']['provider'] == 'openai':
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=config['models']['whisper']['model_name'],
            local_files_only=config['models']['whisper']['local_only']
        )
        log.info("Whisper transcription model loaded successfully")
    
    # Load T5 Model
    if config['models']['t5']['provider'] == 'hf':
        t5_pipeline = pipeline(
            "summarization",
            model=config['models']['t5']['model_name'],
            local_files_only=config['models']['t5']['local_only']
        )
        log.info("T5 summarization model loaded successfully")

# Health Check
@ns_health.route('/health')
class HealthCheck(Resource):
    def get(self):
        return {
            "status": "healthy",
            "models_loaded": {
                "emotion": emotion_pipeline is not None,
                "whisper": whisper_pipeline is not None,
                "t5": t5_pipeline is not None
            },
            "api_version": config['api']['version']
        }

# Emotion Detection Endpoint
@ns_emotion.route('/analyze')
class EmotionAnalysis(Resource):
    def post(self):
        try:
            data = request.get_json()
            request_model = PredictionRequest(**data)
            text = request_model.text
            
            # Use emotion pipeline
            result = emotion_pipeline(text)
            
            return {
                "text": text,
                "emotions": result,
                "confidence": max([r['score'] for r in result]),
                "timestamp": "2025-09-10T10:00:00Z",  # Replace with actual timestamp
                "request_id": request.headers.get('X-Request-ID', 'unknown')
            }
        except ValidationError as e:
            return {"error": str(e)}, 400

@ns_emotion.route('/analyze/batch')
class BatchEmotionAnalysis(Resource):
    def post(self):
        try:
            data = request.get_json()
            request_model = BatchPredictionRequest(**data)
            texts = request_model.texts
            
            results = emotion_pipeline(texts)
            
            return {
                "results": [
                    {
                        "text": texts[i],
                        "emotions": [results[i]],
                        "confidence": max([r['score'] for r in results[i]]),
                        "timestamp": "2025-09-10T10:00:00Z",
                        "request_id": request.headers.get('X-Request-ID', 'unknown')
                    } for i in range(len(texts))
                ]
            }
        except ValidationError as e:
            return {"error": str(e)}, 400

# Voice Transcription Endpoint
@ns_transcribe.route('/voice')
class VoiceTranscription(Resource):
    def post(self):
        if 'audio_file' not in request.files:
            return {"error": "No audio file provided"}, 400
        
        audio_file = request.files['audio_file']
        audio_bytes = audio_file.read()
        
        # Use whisper pipeline
        result = whisper_pipeline(audio_bytes)
        
        return {
            "transcription": result['text'],
            "language": result.get('language', 'en'),
            "duration": len(audio_bytes) / 16000,  # Approximate duration
            "timestamp": "2025-09-10T10:00:00Z",
            "request_id": request.headers.get('X-Request-ID', 'unknown')
        }

# Text Summarization Endpoint
@ns_summarize.route('/text')
class TextSummarization(Resource):
    def post(self):
        try:
            data = request.get_json()
            request_model = PredictionRequest(**data)
            text = request_model.text
            
            # Use T5 pipeline
            result = t5_pipeline(text, max_length=config['models']['t5']['max_output_length'])
            
            return {
                "original_text": text,
                "summary": result[0]['summary_text'],
                "summary_length": len(result[0]['summary_text'].split()),
                "timestamp": "2025-09-10T10:00:00Z",
                "request_id": request.headers.get('X-Request-ID', 'unknown')
            }
        except ValidationError as e:
            return {"error": str(e)}, 400

if __name__ == '__main__':
    load_models()
    app.run(host=config['api']['host'], port=config['api']['port'], debug=config['api']['debug'])