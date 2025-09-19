from flask import Blueprint, request
from flask_restx import Api, Resource, fields, abort
import logging
import time

logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 10000  # Maximum text length for analysis (10k characters)

# Create emotion endpoint blueprint
emotion_bp = Blueprint('emotion', __name__, url_prefix='/api/analyze')

# Create API namespace
api = Api(emotion_bp, doc=False, title='Emotion Analysis API', version='1.0')

# Define request/response models
emotion_request = api.model('EmotionRequest', {
    'text': fields.String(required=True, description='Text to analyze for emotions'),
    'generate_summary': fields.Boolean(required=False, default=False, description='Generate text summary')
})

emotion_response = api.model('EmotionResponse', {
    'emotions': fields.List(fields.String, description='Detected emotions'),
    'confidence_scores': fields.List(fields.Float, description='Confidence scores for each emotion'),
    'summary': fields.String(description='Text summary (if requested)'),
    'processing_time': fields.Float(description='Processing time in seconds'),
    'text_length': fields.Integer(description='Length of input text'),
    'timestamp': fields.String(description='Analysis timestamp'),
    'model_used': fields.String(description='Model identifier used for analysis')
})

@api.route('/journal')
class EmotionAnalysis(Resource):
    """Emotion analysis endpoint for journal entries."""

    @api.expect(emotion_request)
    @api.marshal_with(emotion_response)
    def post(self):
        """Analyze emotions in journal text."""
        try:
            start_time = time.time()

            # Get request data
            data = request.get_json()
            if not data or 'text' not in data:
                abort(400, 'Text is required')

            text = data['text']
            generate_summary = data.get('generate_summary', False)

            # Validate input
            if not isinstance(text, str) or len(text.strip()) == 0:
                abort(400, 'Text must be a non-empty string')

            if len(text) > MAX_TEXT_LENGTH:  # 10k character limit
                abort(400, 'Text too long (max 10,000 characters)')

            # Mock emotion analysis (replace with actual model integration)
            emotions, confidence_scores = self._analyze_emotions(text)

            # Generate summary if requested
            summary = None
            if generate_summary:
                summary = self._generate_summary(text)

            processing_time = time.time() - start_time

            # Prepare response
            response = {
                'emotions': emotions,
                'confidence_scores': confidence_scores,
                'summary': summary,
                'processing_time': round(processing_time, 3),
                'text_length': len(text),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'model_used': 'mock-emotion-analyzer-v1.0'
            }

            logger.info(f"Emotion analysis completed: {len(emotions)} emotions detected in {processing_time:.3f}s")
            return response, 200

        except Exception as e:
            logger.exception("Emotion analysis failed")
            abort(500, 'Emotion analysis failed')

    @staticmethod
    def _analyze_emotions(text: str) -> tuple[list[str], list[float]]:
        """Analyze emotions in text (mock implementation)."""
        # Mock emotion detection - replace with actual SAMO BERT model
        emotions = []
        confidence_scores = []

        # Simple keyword-based emotion detection for demo
        text_lower = text.lower()

        emotion_keywords = {
            'joy': ['happy', 'excited', 'joyful', 'cheerful', 'delighted'],
            'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'sorrowful'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed'],
            'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened']
        }

        for emotion, keywords in emotion_keywords.items():
            confidence = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
            if confidence > 0.1:  # Threshold for detection
                emotions.append(emotion)
                confidence_scores.append(min(confidence * 2, 1.0))  # Scale to 0-1

        # If no emotions detected, add neutral
        if not emotions:
            emotions = ['neutral']
            confidence_scores = [0.5]

        return emotions, confidence_scores

    @staticmethod
    def _generate_summary(text: str) -> str:
        """Generate text summary (mock implementation)."""
        # Mock summarization - replace with actual T5 model
        words = text.split()
        if len(words) <= 20:
            return text

        # Simple extractive summary (first 20 words)
        summary_words = words[:20]
        return ' '.join(summary_words) + '...'

def register_emotion_endpoints(app):
    """Register emotion endpoints with the Flask app."""
    app.register_blueprint(emotion_bp)
    logger.info("Emotion endpoints registered: /api/analyze/journal")
