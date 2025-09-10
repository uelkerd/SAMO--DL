from flask import Blueprint, jsonify
from flask_restx import Api, Resource, fields
import logging
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)

# Create API examples blueprint
api_examples_bp = Blueprint('api_examples', __name__, url_prefix='/api/examples')

# Create API namespace
api = Api(api_examples_bp, doc=False, title='SAMO-DL API Examples', version='1.0')

# Define response models for examples
example_response = api.model('ExampleResponse', {
    'endpoint': fields.String(description='Endpoint path'),
    'description': fields.String(description='Example description'),
    'request': fields.String(description='Example request'),
    'response': fields.String(description='Example response'),
    'curl_command': fields.String(description='cURL command example')
})

class APIExamples(Resource):
    """API examples and usage demonstrations."""
    
    def __init__(self):
        self.examples = {
            "emotion_analysis": {
                "endpoint": "/api/analyze/journal",
                "description": "Analyze journal text for emotions with confidence scores",
                "request": {
                    "text": "I had a wonderful day today! I went for a walk in the park and felt so peaceful and content. The weather was perfect and I met some friendly people. I'm feeling grateful and happy.",
                    "generate_summary": True
                },
                "response": {
                    "emotions": ["joy", "gratitude", "contentment", "peace"],
                    "confidence_scores": [0.92, 0.88, 0.85, 0.78],
                    "summary": "The person had a wonderful day with peaceful activities, feeling grateful and happy.",
                    "processing_time": 1.2,
                    "model_used": "emotion-detection"
                },
                "curl_command": 'curl -X POST "http://localhost:5000/api/analyze/journal" -H "Content-Type: application/json" -d \'{"text": "I had a wonderful day today!", "generate_summary": true}\''
            },
            "text_summarization": {
                "endpoint": "/api/summarize/",
                "description": "Summarize long text using T5 model",
                "request": {
                    "text": "The meeting today was quite productive. We discussed the quarterly goals and made significant progress on the new project. The team was engaged and contributed valuable insights. We also addressed some challenges and came up with solutions. Overall, it was a successful session that moved us forward.",
                    "max_length": 100,
                    "min_length": 30,
                    "temperature": 0.7
                },
                "response": {
                    "summary": "The meeting was productive with team engagement, progress on quarterly goals, and successful problem-solving.",
                    "original_length": 280,
                    "summary_length": 95,
                    "compression_ratio": 0.34,
                    "processing_time": 0.8,
                    "model_used": "t5-base"
                },
                "curl_command": 'curl -X POST "http://localhost:5000/api/summarize/" -H "Content-Type: application/json" -d \'{"text": "Long text to summarize", "max_length": 100}\''
            },
            "audio_transcription": {
                "endpoint": "/api/transcribe/",
                "description": "Transcribe audio recording to text",
                "request": {
                    "audio_data": "UklGRjIAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
                    "audio_format": "wav",
                    "language": "en",
                    "task": "transcribe"
                },
                "response": {
                    "text": "Hello, this is a test recording for the SAMO-DL API transcription service.",
                    "language": "en",
                    "confidence": 0.94,
                    "duration": 3.5,
                    "processing_time": 2.1,
                    "model_used": "whisper-base"
                },
                "curl_command": 'curl -X POST "http://localhost:5000/api/transcribe/" -H "Content-Type: application/json" -d \'{"audio_data": "base64_encoded_audio", "language": "en"}\''
            },
            "complete_analysis": {
                "endpoint": "/api/complete-analysis/",
                "description": "Complete analysis combining emotion detection, summarization, and transcription",
                "request": {
                    "text": "I'm feeling overwhelmed with work lately. There's so much to do and I'm struggling to keep up. I feel stressed and anxious about meeting deadlines. I need to find a better way to manage my time and prioritize tasks.",
                    "include_summary": True,
                    "include_emotion": True,
                    "include_transcription": False
                },
                "response": {
                    "text": "I'm feeling overwhelmed with work lately. There's so much to do and I'm struggling to keep up. I feel stressed and anxious about meeting deadlines. I need to find a better way to manage my time and prioritize tasks.",
                    "emotions": ["overwhelm", "stress", "anxiety", "frustration"],
                    "confidence_scores": [0.89, 0.85, 0.82, 0.78],
                    "summary": "The person feels overwhelmed and stressed about work, struggling with time management and deadlines.",
                    "transcription": "",
                    "language": "en",
                    "processing_time": 3.2,
                    "models_used": ["emotion-detection", "t5-summarization"],
                    "analysis_timestamp": "2025-09-10 12:55:00"
                },
                "curl_command": 'curl -X POST "http://localhost:5000/api/complete-analysis/" -H "Content-Type: application/json" -d \'{"text": "Sample text", "include_summary": true, "include_emotion": true}\''
            },
            "health_check": {
                "endpoint": "/api/health/",
                "description": "Check system health and status",
                "request": {},
                "response": {
                    "status": "healthy",
                    "uptime": 3600,
                    "models_loaded": True,
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "disk_usage": 23.1,
                    "request_count": 1250,
                    "error_count": 5
                },
                "curl_command": 'curl -X GET "http://localhost:5000/api/health/"'
            }
        }
    
    @api.marshal_with(example_response)
    def get(self, example_type: str = None):
        """Get API examples for specific endpoint or all endpoints."""
        try:
            if example_type:
                if example_type not in self.examples:
                    return {"error": "Example type not found"}, 404
                
                example = self.examples[example_type]
                return {
                    "endpoint": example["endpoint"],
                    "description": example["description"],
                    "request": json.dumps(example["request"], indent=2),
                    "response": json.dumps(example["response"], indent=2),
                    "curl_command": example["curl_command"]
                }
            else:
                # Return all examples
                all_examples = []
                for example_type, example in self.examples.items():
                    all_examples.append({
                        "endpoint": example["endpoint"],
                        "description": example["description"],
                        "request": json.dumps(example["request"], indent=2),
                        "response": json.dumps(example["response"], indent=2),
                        "curl_command": example["curl_command"]
                    })
                return all_examples
        except Exception as e:
            logger.error(f"Failed to get examples: {e}")
            return {"error": "Failed to get examples"}, 500
    
    def get_example_types(self):
        """Get list of available example types."""
        try:
            return list(self.examples.keys())
        except Exception as e:
            logger.error(f"Failed to get example types: {e}")
            return {"error": "Failed to get example types"}, 500

# Register the endpoints
api.add_resource(APIExamples, '/')
api.add_resource(APIExamples, '/<string:example_type>')

# Get available example types endpoint
@api_examples_bp.route('/types', methods=['GET'])
def get_example_types():
    """Get list of available example types."""
    try:
        examples = APIExamples()
        return jsonify({
            "example_types": examples.get_example_types(),
            "total_count": len(examples.examples)
        })
    except Exception as e:
        logger.error(f"Failed to get example types: {e}")
        return {"error": "Failed to get example types"}, 500

# Health check for API examples
@api_examples_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for API examples endpoint."""
    return jsonify({
        "status": "healthy",
        "endpoint": "api_examples",
        "examples_available": True
    })
