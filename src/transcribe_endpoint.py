from flask import Blueprint, request, jsonify
from flask_restx import Api, Resource, fields
import logging
from typing import Dict, Any, Optional
import time
import base64

logger = logging.getLogger(__name__)

# Create transcribe endpoint blueprint
transcribe_bp = Blueprint('transcribe', __name__, url_prefix='/api/transcribe')

# Create API namespace
api = Api(transcribe_bp, doc=False, title='Audio Transcription API', version='1.0')

# Define request/response models
transcribe_request = api.model('TranscribeRequest', {
    'audio_data': fields.String(required=True, description='Base64 encoded audio data'),
    'audio_format': fields.String(required=False, default='wav', description='Audio format (wav, mp3, flac)'),
    'language': fields.String(required=False, default='en', description='Language code for transcription'),
    'task': fields.String(required=False, default='transcribe', description='Task type (transcribe, translate)')
})

transcribe_response = api.model('TranscribeResponse', {
    'text': fields.String(description='Transcribed text'),
    'language': fields.String(description='Detected language'),
    'confidence': fields.Float(description='Confidence score'),
    'duration': fields.Float(description='Audio duration in seconds'),
    'processing_time': fields.Float(description='Processing time in seconds'),
    'model_used': fields.String(description='Model used for transcription')
})

class TranscribeEndpoint(Resource):
    """Audio transcription endpoint for voice recordings."""

    def __init__(self):
        super().__init__()
        self.model_loaded = False
        self.model = None

    def load_model(self):
        """Load the Whisper transcription model."""
        try:
            from src.models.voice_processing.samo_whisper_transcriber import SAMOWhisperTranscriber
            self.model = SAMOWhisperTranscriber()
            self.model_loaded = True
            logger.info("Whisper transcription model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model_loaded = False

    @staticmethod
    def validate_audio_data(audio_data: str, audio_format: str) -> bool:
        """Validate audio data format and size."""
        try:
            # Decode base64 data
            decoded_data = base64.b64decode(audio_data)

            # Check file size (max 25MB)
            if len(decoded_data) > 25 * 1024 * 1024:
                return False

            # Check format
            if audio_format.lower() not in ['wav', 'mp3', 'flac', 'm4a']:
                return False

            return True
        except Exception:
            return False

    @api.expect(transcribe_request)
    @api.marshal_with(transcribe_response)
    def post(self):
        """Transcribe audio using Whisper model."""
        try:
            data = request.get_json()
            if not data:
                return {"error": "No JSON data provided"}, 400

            audio_data = data.get('audio_data', '').strip()
            if not audio_data:
                return {"error": "Audio data is required"}, 400

            audio_format = data.get('audio_format', 'wav').lower()
            language = data.get('language', 'en')
            task = data.get('task', 'transcribe')

            # Validate parameters
            if task not in ['transcribe', 'translate']:
                return {"error": "Task must be 'transcribe' or 'translate'"}, 400

            if language not in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']:
                return {"error": "Unsupported language code"}, 400

            # Validate audio data
            if not self.validate_audio_data(audio_data, audio_format):
                return {"error": "Invalid audio data or format"}, 400

            start_time = time.time()

            # Load model if not already loaded
            if not self.model_loaded:
                self.load_model()

            # Transcribe audio
            if self.model_loaded and self.model:
                try:
                    audio_bytes = base64.b64decode(audio_data)

                    # Create temporary file for audio processing
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
                        temp_file.write(audio_bytes)
                        temp_file_path = temp_file.name

                    # Transcribe using the model
                    result = self.model.transcribe_audio(temp_file_path, language=language)

                    # Clean up temporary file
                    import os
                    os.unlink(temp_file_path)

                    text = result.get('text', f"[ERROR] Transcription failed for {language} audio")
                    confidence = result.get('confidence', 0.0)
                    detected_language = result.get('language', language)
                    duration = result.get('duration', 0.0)

                except Exception as e:
                    logger.error(f"Transcription failed: {e}")
                    text = f"[ERROR] Transcription failed: {str(e)}"
                    confidence = 0.0
                    detected_language = language
                    duration = 0.0
            else:
                # Fallback mock transcription
                text = f"[MOCK] Transcribed audio in {language}: This is a sample transcription of audio data."
                confidence = 0.75
                detected_language = language
                duration = 5.0

            processing_time = time.time() - start_time

            return {
                "text": text,
                "language": detected_language,
                "confidence": confidence,
                "duration": duration,
                "processing_time": processing_time,
                "model_used": "whisper-base" if self.model_loaded else "mock"
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"error": "Transcription failed"}, 500

# Register the endpoint
api.add_resource(TranscribeEndpoint, '/')

# Health check for transcribe endpoint
@transcribe_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for transcribe endpoint."""
    return jsonify({
        "status": "healthy",
        "endpoint": "transcribe",
        "model_loaded": TranscribeEndpoint().model_loaded
    })
