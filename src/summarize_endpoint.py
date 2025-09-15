from flask import Blueprint, request, jsonify
from flask_restx import Api, Resource, fields
import logging
import time

logger = logging.getLogger(__name__)

# Create summarize endpoint blueprint
summarize_bp = Blueprint('summarize', __name__, url_prefix='/api/summarize')

# Create API namespace
api = Api(summarize_bp, doc=False, title='Text Summarization API', version='1.0')

# Define request/response models
summarize_request = api.model('SummarizeRequest', {
    'text': fields.String(required=True, description='Text to summarize'),
    'max_length': fields.Integer(required=False, default=150, description='Maximum summary length'),
    'min_length': fields.Integer(required=False, default=30, description='Minimum summary length'),
    'temperature': fields.Float(required=False, default=0.7, description='Sampling temperature')
})

summarize_response = api.model('SummarizeResponse', {
    'summary': fields.String(description='Generated summary'),
    'original_length': fields.Integer(description='Length of original text'),
    'summary_length': fields.Integer(description='Length of generated summary'),
    'compression_ratio': fields.Float(description='Compression ratio'),
    'processing_time': fields.Float(description='Processing time in seconds'),
    'model_used': fields.String(description='Model used for summarization')
})

class SummarizeEndpoint(Resource):
    """Text summarization endpoint for journal entries."""

    def __init__(self):
        super().__init__()
        self.model_loaded = False
        self.model = None

    def load_model(self):
        """Load the T5 summarization model."""
        try:
            from src.models.summarization.samo_t5_summarizer import SAMOT5Summarizer
            self.model = SAMOT5Summarizer()
            self.model_loaded = True
            logger.info("T5 summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}")
            self.model_loaded = False

    @api.expect(summarize_request)
    @api.marshal_with(summarize_response)
    def post(self):
        """Summarize text using T5 model."""
        try:
            data = request.get_json()
            if not data:
                return {"error": "No JSON data provided"}, 400

            text = data.get('text', '').strip()
            if not text:
                return {"error": "Text is required"}, 400

            if len(text) < 50:
                return {"error": "Text must be at least 50 characters"}, 400

            max_length = data.get('max_length', 150)
            min_length = data.get('min_length', 30)
            temperature = data.get('temperature', 0.7)

            # Validate parameters
            if max_length < min_length:
                return {"error": "max_length must be greater than min_length"}, 400

            if not 0.1 <= temperature <= 2.0:
                return {"error": "temperature must be between 0.1 and 2.0"}, 400

            start_time = time.time()

            # Load model if not already loaded
            if not self.model_loaded:
                self.load_model()

            # Generate summary
            if self.model_loaded and self.model:
                try:
                    # Use actual T5 model for summarization
                    result = self.model.generate_summary(text)
                    summary = result.get('summary', '[ERROR] Failed to generate summary')
                except Exception as e:
                    logger.error(f"T5 summarization failed: {e}")
                    # Fallback to mock result
                    summary = f"[FALLBACK] Summary failed, mock result: {text[:100]}..."
            else:
                # Mock summarization result when model not loaded
                summary = f"[MOCK] Summary of {len(text)} characters: {text[:50]}..."

            processing_time = time.time() - start_time

            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text),
                "processing_time": processing_time,
                "model_used": "t5-base" if self.model_loaded else "mock"
            }

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {"error": "Summarization failed"}, 500

# Register the endpoint
api.add_resource(SummarizeEndpoint, '/')

# Health check for summarize endpoint
@summarize_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for summarize endpoint."""
    return jsonify({
        "status": "healthy",
        "endpoint": "summarize",
        "model_loaded": SummarizeEndpoint().model_loaded
    })
