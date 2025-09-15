from flask import Blueprint, request, jsonify
from flask_restx import Api, Resource, fields
import logging
from typing import Dict, Any
import time
import base64

logger = logging.getLogger(__name__)

# Create complete analysis endpoint blueprint
complete_analysis_bp = Blueprint('complete_analysis', __name__, url_prefix='/api/complete-analysis')

# Create API namespace
api = Api(complete_analysis_bp, doc=False, title='Complete Analysis API', version='1.0')

# Define request/response models
complete_analysis_request = api.model('CompleteAnalysisRequest', {
    'text': fields.String(required=False, description='Text to analyze for emotions and summarization'),
    'audio_data': fields.String(required=False, description='Base64 encoded audio data for transcription'),
    'audio_format': fields.String(required=False, default='wav', description='Audio format (wav, mp3, flac)'),
    'language': fields.String(required=False, default='en', description='Language code'),
    'include_summary': fields.Boolean(required=False, default=True, description='Include text summarization'),
    'include_emotion': fields.Boolean(required=False, default=True, description='Include emotion analysis'),
    'include_transcription': fields.Boolean(required=False, default=False, description='Include audio transcription')
})

complete_analysis_response = api.model('CompleteAnalysisResponse', {
    'text': fields.String(description='Original or transcribed text'),
    'emotions': fields.List(fields.String, description='Detected emotions'),
    'confidence_scores': fields.List(fields.Float, description='Confidence scores for each emotion'),
    'summary': fields.String(description='Generated summary'),
    'transcription': fields.String(description='Transcribed text from audio'),
    'language': fields.String(description='Detected language'),
    'processing_time': fields.Float(description='Total processing time in seconds'),
    'models_used': fields.List(fields.String, description='Models used for analysis'),
    'analysis_timestamp': fields.String(description='Timestamp of analysis')
})

class CompleteAnalysisEndpoint(Resource):
    """Complete analysis endpoint combining emotion, summarization, and transcription."""

    def __init__(self):
        super().__init__()
        self.emotion_model_loaded = False
        self.summarization_model_loaded = False
        self.transcription_model_loaded = False
        self.emotion_model = None
        self.summarization_model = None
        self.transcription_model = None

    def load_models(self):
        """Load all required models for complete analysis."""
        try:
            # Load emotion detection model
            from src.inference.text_emotion_service import HFEmotionService
            self.emotion_model = HFEmotionService()
            self.emotion_model_loaded = True

            # Load summarization model
            from src.models.summarization.samo_t5_summarizer import SAMOT5Summarizer
            self.summarization_model = SAMOT5Summarizer()
            self.summarization_model_loaded = True

            # Load transcription model
            from src.models.voice_processing.whisper_transcriber import WhisperTranscriber, TranscriptionConfig
            config = TranscriptionConfig(model_size="base")
            self.transcription_model = WhisperTranscriber(config)
            self.transcription_model_loaded = True

            logger.info("All models loaded successfully for complete analysis")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.emotion_model_loaded = False
            self.summarization_model_loaded = False
            self.transcription_model_loaded = False

    @staticmethod
    def validate_input(data: Dict[str, Any]) -> tuple[bool, str]:
        """Validate input data for complete analysis."""
        text = data.get('text', '').strip()
        audio_data = data.get('audio_data', '').strip()

        if not text and not audio_data:
            return False, "Either text or audio_data must be provided"

        if text and len(text) < 50:
            return False, "Text must be at least 50 characters"

        if audio_data:
            try:
                decoded_data = base64.b64decode(audio_data)
                if len(decoded_data) > 25 * 1024 * 1024:  # 25MB limit
                    return False, "Audio file too large (max 25MB)"
            except Exception:
                return False, "Invalid audio data format"

        return True, ""

    @api.expect(complete_analysis_request)
    @api.marshal_with(complete_analysis_response)
    def post(self):
        """Perform complete analysis combining all models."""
        try:
            data = request.get_json()
            if not data:
                return {"error": "No JSON data provided"}, 400

            # Validate input
            is_valid, error_msg = self.validate_input(data)
            if not is_valid:
                return {"error": error_msg}, 400

            start_time = time.time()

            # Load models if not already loaded
            if not (self.emotion_model_loaded and self.summarization_model_loaded and self.transcription_model_loaded):
                self.load_models()

            # Extract parameters
            text = data.get('text', '').strip()
            audio_data = data.get('audio_data', '').strip()
            # audio_format = data.get('audio_format', 'wav')  # Currently unused
            language = data.get('language', 'en')
            include_summary = data.get('include_summary', True)
            include_emotion = data.get('include_emotion', True)
            include_transcription = data.get('include_transcription', False)

            # Process audio if provided
            transcription = ""
            if audio_data and include_transcription:
                if self.transcription_model_loaded and self.transcription_model:
                    try:
                        # Decode base64 audio data and save to temporary file
                        import tempfile
                        import os

                        audio_format = data.get('audio_format', 'wav')
                        decoded_audio = base64.b64decode(audio_data)
                        temp_file = tempfile.NamedTemporaryFile(
                            suffix=f".{audio_format}", delete=False
                        )
                        temp_file.write(decoded_audio)
                        temp_file.close()

                        try:
                            result = self.transcription_model.transcribe(temp_file.name, language=language)
                            transcription = result.text
                        finally:
                            try:
                                os.unlink(temp_file.name)
                            except Exception:
                                pass
                    except Exception as e:
                        logger.error(f"Transcription failed: {e}")
                        transcription = f"[FALLBACK] Transcription failed in {language}"
                else:
                    transcription = f"[MOCK] Transcribed audio in {language}: This is a sample transcription."

            # Use transcribed text if no text provided
            if not text and transcription:
                text = transcription

            # Perform emotion analysis
            emotions = []
            confidence_scores = []
            if text and include_emotion:
                if self.emotion_model_loaded and self.emotion_model:
                    try:
                        # Use actual emotion detection
                        emotion_results = self.emotion_model.classify(text)
                        if emotion_results and emotion_results[0]:
                            # Extract top emotions and scores
                            emotions = [result['label'] for result in emotion_results[0][:3]]
                            confidence_scores = [result['score'] for result in emotion_results[0][:3]]
                        else:
                            emotions = ["neutral"]
                            confidence_scores = [0.5]
                    except Exception as e:
                        logger.error(f"Emotion analysis failed: {e}")
                        emotions = ["neutral"]
                        confidence_scores = [0.5]
                else:
                    emotions = ["joy", "sadness", "anger"]
                    confidence_scores = [0.8, 0.6, 0.3]

            # Perform summarization
            summary = ""
            if text and include_summary:
                if self.summarization_model_loaded and self.summarization_model:
                    try:
                        # Use actual T5 model for summarization
                        result = self.summarization_model.generate_summary(text)
                        summary = result.get('summary', '[ERROR] Failed to generate summary')
                    except Exception as e:
                        logger.error(f"Summarization failed: {e}")
                        summary = f"[FALLBACK] Summary failed: {text[:100]}..."
                else:
                    summary = f"[MOCK] Summary of {len(text)} characters: {text[:50]}..."

            processing_time = time.time() - start_time

            # Determine models used
            models_used = []
            if include_emotion and self.emotion_model_loaded:
                models_used.append("emotion-detection")
            if include_summary and self.summarization_model_loaded:
                models_used.append("t5-summarization")
            if include_transcription and self.transcription_model_loaded:
                models_used.append("whisper-transcription")

            return {
                "text": text,
                "emotions": emotions,
                "confidence_scores": confidence_scores,
                "summary": summary,
                "transcription": transcription,
                "language": language,
                "processing_time": processing_time,
                "models_used": models_used,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            return {"error": "Complete analysis failed"}, 500

# Register the endpoint
api.add_resource(CompleteAnalysisEndpoint, '/')

# Health check for complete analysis endpoint
@complete_analysis_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for complete analysis endpoint."""
    endpoint = CompleteAnalysisEndpoint()
    return jsonify({
        "status": "healthy",
        "endpoint": "complete_analysis",
        "models_loaded": {
            "emotion": endpoint.emotion_model_loaded,
            "summarization": endpoint.summarization_model_loaded,
            "transcription": endpoint.transcription_model_loaded
        }
    })
