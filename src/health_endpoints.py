from flask import Blueprint, jsonify
from health_monitor import health_monitor
import logging

logger = logging.getLogger(__name__)

# Create health endpoints blueprint
health_bp = Blueprint('health', __name__, url_prefix='/api/health')

@health_bp.route('/', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    try:
        summary = health_monitor.get_health_summary()
        status_code = 200 if summary["status"] in ["healthy", "warning"] else 503
        return jsonify(summary), status_code
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": "Health check failed"
        }), 500

@health_bp.route('/detailed', methods=['GET'])
def detailed_health():
    """Detailed health check with system metrics."""
    try:
        health_data = health_monitor.get_system_health()
        status_code = 200 if health_data["status"] in ["healthy", "warning"] else 503
        return jsonify(health_data), status_code
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": "Detailed health check failed"
        }), 500

@health_bp.route('/ready', methods=['GET'])
def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        health_data = health_monitor.get_system_health()
        if health_data["status"] in ["healthy", "warning"]:
            return jsonify({"ready": True}), 200
        return jsonify({"ready": False, "reason": health_data["status"]}), 503
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({"ready": False, "reason": "error"}), 503

@health_bp.route('/live', methods=['GET'])
def liveness_check():
    """Kubernetes liveness probe endpoint."""
    try:
        # Simple liveness check - just verify the service is responding
        return jsonify({"alive": True}), 200
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return jsonify({"alive": False}), 500

@health_bp.route('/metrics', methods=['GET'])
def health_metrics():
    """Health metrics endpoint for monitoring systems."""
    try:
        health_data = health_monitor.get_system_health()
        metrics = {
            "api_requests_total": health_data["process"]["request_count"],
            "api_errors_total": health_data["process"]["error_count"],
            "api_error_rate_percent": health_data["process"]["error_rate"],
            "system_cpu_percent": health_data["system"]["cpu_percent"],
            "system_memory_percent": health_data["system"]["memory_percent"],
            "system_disk_percent": health_data["system"]["disk_percent"],
            "uptime_seconds": health_data["uptime_hours"] * 3600
        }
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return jsonify({"error": "Metrics collection failed"}), 500

def register_health_endpoints(app):
    """Register health endpoints with the Flask app."""
    app.register_blueprint(health_bp)
    logger.info("Health endpoints registered: /api/health/*")
