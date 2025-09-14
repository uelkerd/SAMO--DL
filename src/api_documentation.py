from flask import Blueprint, jsonify
from flask_restx import Api, Resource, fields
import logging
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)

# Create API documentation blueprint
api_docs_bp = Blueprint('api_docs', __name__, url_prefix='/api/docs')

# Create API namespace
api = Api(api_docs_bp, doc=False, title='SAMO-DL API Documentation', version='1.0')

# Define response models for documentation
api_info_response = api.model('APIInfoResponse', {
    'title': fields.String(description='API title'),
    'version': fields.String(description='API version'),
    'description': fields.String(description='API description'),
    'endpoints': fields.List(fields.String, description='Available endpoints'),
    'models': fields.List(fields.String, description='Available models'),
    'status': fields.String(description='API status')
})

endpoint_info_response = api.model('EndpointInfoResponse', {
    'endpoint': fields.String(description='Endpoint path'),
    'method': fields.String(description='HTTP method'),
    'description': fields.String(description='Endpoint description'),
    'parameters': fields.List(fields.String, description='Request parameters'),
    'response': fields.String(description='Response format'),
    'example': fields.String(description='Example request/response')
})

class APIDocumentation(Resource):
    """API documentation and information endpoints."""

    def __init__(self):
        self.api_info = {
            "title": "SAMO-DL API",
            "version": "1.0.0",
            "description": "A deep learning API for nuanced emotion analysis in reflective text",
            "endpoints": [
                "/api/analyze/journal",
                "/api/summarize/",
                "/api/transcribe/",
                "/api/complete-analysis/",
                "/api/health/",
                "/api/docs/"
            ],
            "models": [
                "emotion-detection",
                "t5-summarization",
                "whisper-transcription"
            ],
            "status": "operational"
        }

        self.endpoints_info = {
            "/api/analyze/journal": {
                "method": "POST",
                "description": "Analyze journal text for emotions",
                "parameters": ["text", "generate_summary"],
                "response": "JSON with emotions and confidence scores",
                "example": {
                    "request": {"text": "I feel happy today", "generate_summary": True},
                    "response": {"emotions": ["joy"], "confidence_scores": [0.85]}
                }
            },
            "/api/summarize/": {
                "method": "POST",
                "description": "Summarize text using T5 model",
                "parameters": ["text", "max_length", "min_length", "temperature"],
                "response": "JSON with summary and metrics",
                "example": {
                    "request": {"text": "Long text to summarize", "max_length": 150},
                    "response": {"summary": "Short summary", "compression_ratio": 0.15}
                }
            },
            "/api/transcribe/": {
                "method": "POST",
                "description": "Transcribe audio using Whisper model",
                "parameters": ["audio_data", "audio_format", "language", "task"],
                "response": "JSON with transcribed text and metadata",
                "example": {
                    "request": {"audio_data": "base64_encoded_audio", "language": "en"},
                    "response": {"text": "Transcribed text", "confidence": 0.85}
                }
            },
            "/api/complete-analysis/": {
                "method": "POST",
                "description": "Complete analysis combining all models",
                "parameters": ["text", "audio_data", "include_summary", "include_emotion", "include_transcription"],
                "response": "JSON with comprehensive analysis results",
                "example": {
                    "request": {"text": "Sample text", "include_summary": True, "include_emotion": True},
                    "response": {"emotions": ["joy"], "summary": "Summary", "processing_time": 2.5}
                }
            },
            "/api/health/": {
                "method": "GET",
                "description": "Health check and system status",
                "parameters": [],
                "response": "JSON with system health metrics",
                "example": {
                    "request": {},
                    "response": {"status": "healthy", "uptime": 3600, "models_loaded": True}
                }
            }
        }

    @api.marshal_with(api_info_response)
    def get(self):
        """Get API information and overview."""
        try:
            return self.api_info
        except Exception as e:
            logger.error(f"Failed to get API info: {e}")
            return {"error": "Failed to get API information"}, 500

    @api.marshal_with(endpoint_info_response)
    def get_endpoint(self, endpoint_path: str):
        """Get detailed information about a specific endpoint."""
        try:
            if endpoint_path not in self.endpoints_info:
                return {"error": "Endpoint not found"}, 404

            endpoint_info = self.endpoints_info[endpoint_path]
            return {
                "endpoint": endpoint_path,
                "method": endpoint_info["method"],
                "description": endpoint_info["description"],
                "parameters": endpoint_info["parameters"],
                "response": endpoint_info["response"],
                "example": json.dumps(endpoint_info["example"], indent=2)
            }
        except Exception as e:
            logger.error(f"Failed to get endpoint info: {e}")
            return {"error": "Failed to get endpoint information"}, 500

# Register the endpoints
api.add_resource(APIDocumentation, '/')
api.add_resource(APIDocumentation, '/<string:endpoint_path>')

# OpenAPI/Swagger documentation endpoint
@api_docs_bp.route('/openapi.json', methods=['GET'])
def openapi_spec():
    """Generate OpenAPI specification for the API."""
    try:
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "SAMO-DL API",
                "version": "1.0.0",
                "description": "A deep learning API for nuanced emotion analysis in reflective text"
            },
            "servers": [
                {"url": "http://localhost:5000", "description": "Development server"},
                {"url": "https://api.samo-dl.com", "description": "Production server"}
            ],
            "paths": {
                "/api/analyze/journal": {
                    "post": {
                        "summary": "Analyze journal text for emotions",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string", "description": "Text to analyze"},
                                            "generate_summary": {"type": "boolean", "description": "Generate summary"}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful analysis",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "emotions": {"type": "array", "items": {"type": "string"}},
                                                "confidence_scores": {"type": "array", "items": {"type": "number"}}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/summarize/": {
                    "post": {
                        "summary": "Summarize text using T5 model",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string", "description": "Text to summarize"},
                                            "max_length": {"type": "integer", "description": "Maximum summary length"},
                                            "min_length": {"type": "integer", "description": "Minimum summary length"},
                                            "temperature": {"type": "number", "description": "Sampling temperature"}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful summarization",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "summary": {"type": "string"},
                                                "compression_ratio": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/transcribe/": {
                    "post": {
                        "summary": "Transcribe audio using Whisper model",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "audio_data": {"type": "string", "description": "Base64 encoded audio"},
                                            "audio_format": {"type": "string", "description": "Audio format"},
                                            "language": {"type": "string", "description": "Language code"},
                                            "task": {"type": "string", "description": "Task type"}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful transcription",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "text": {"type": "string"},
                                                "confidence": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/complete-analysis/": {
                    "post": {
                        "summary": "Complete analysis combining all models",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string", "description": "Text to analyze"},
                                            "audio_data": {"type": "string", "description": "Base64 encoded audio"},
                                            "include_summary": {"type": "boolean", "description": "Include summarization"},
                                            "include_emotion": {"type": "boolean", "description": "Include emotion analysis"},
                                            "include_transcription": {"type": "boolean", "description": "Include transcription"}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful complete analysis",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "emotions": {"type": "array", "items": {"type": "string"}},
                                                "summary": {"type": "string"},
                                                "transcription": {"type": "string"},
                                                "processing_time": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/api/health/": {
                    "get": {
                        "summary": "Health check and system status",
                        "responses": {
                            "200": {
                                "description": "System health status",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "status": {"type": "string"},
                                                "uptime": {"type": "number"},
                                                "models_loaded": {"type": "boolean"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return jsonify(openapi_spec)
    except Exception as e:
        logger.error(f"Failed to generate OpenAPI spec: {e}")
        return {"error": "Failed to generate OpenAPI specification"}, 500

# Health check for API documentation
@api_docs_bp.route('/health', methods=['GET'])
def health_check():
    """Health check for API documentation endpoint."""
    return jsonify({
        "status": "healthy",
        "endpoint": "api_documentation",
        "openapi_available": True
    })
