"""
Cloud Run Health Monitor - Phase 3 Optimization
Provides comprehensive health checks, graceful shutdown, and monitoring
"""

import os
import sys
import time
import signal
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """Health check metrics"""
    status: str
    response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_requests: int
    timestamp: datetime
    error_message: Optional[str] = None

class HealthMonitor:
    """Comprehensive health monitoring for Cloud Run"""

    def __init__(self):
        self.start_time = datetime.now()
        self.is_shutting_down = False
        self.active_requests = 0
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.shutdown_timeout = int(os.getenv('GRACEFUL_SHUTDOWN_TIMEOUT', '30') or '30')

        # Register graceful shutdown handlers
        signal.signal(signal.SIGTERM, self._graceful_shutdown)
        signal.signal(signal.SIGINT, self._graceful_shutdown)

        logger.info(f"Health monitor initialized with {self.shutdown_timeout}s shutdown timeout")

    def _graceful_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received shutdown signal {signum}, starting graceful shutdown...")
        self.is_shutting_down = True

        # Wait for active requests to complete
        start_wait = time.time()
        while self.active_requests > 0 and (time.time() - start_wait) < self.shutdown_timeout:
            logger.info(f"Waiting for {self.active_requests} active requests to complete...")
            time.sleep(1)

        if self.active_requests > 0:
            logger.warning(f"Force shutdown after {self.shutdown_timeout}s timeout with {self.active_requests} active requests")
        else:
            logger.info("Graceful shutdown completed successfully")

        raise SystemExit(0)

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'cpu_usage_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                'memory_usage_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'memory_percent': 0.0,
                'uptime_seconds': 0.0
            }

    @staticmethod
    def check_model_health() -> Dict[str, Any]:
        """Check if ML models are loaded and responding"""
        try:
            # Import models (this will fail if models aren't loaded)
            from secure_api_server import app

            # Test model loading
            start_time = time.time()

            # Simple health check - try to import key components
            import importlib
            modules_to_check = [
                'src.models.emotion_detection.bert_classifier',
                'src.models.summarization.t5_summarizer', 
                'src.models.voice_processing.whisper_transcriber'
            ]

            for module_name in modules_to_check:
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    return {
                        'status': 'unhealthy',
                        'error': f'Model module {module_name} not available: {e}',
                        'response_time_ms': (time.time() - start_time) * 1000
                    }

            return {
                'status': 'healthy',
                'response_time_ms': (time.time() - start_time) * 1000,
                'models_loaded': len(modules_to_check)
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': f'Model health check failed: {e}',
                'response_time_ms': 0
            }

    @staticmethod
    def check_api_health() -> Dict[str, Any]:
        """Check API endpoint health"""
        try:
            start_time = time.time()

            # Test internal health endpoint
            from secure_api_server import app

            # Use Flask test client instead of FastAPI TestClient
            with app.test_client() as client:
                response = client.get("/health")

            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time_ms': response_time,
                    'status_code': response.status_code
                }
            return {
                'status': 'unhealthy',
                'error': f'Health endpoint returned {response.status_code}',
                'response_time_ms': response_time,
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': f'API health check failed: {e}',
                'response_time_ms': 0
            }

    def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        if self.is_shutting_down:
            return {
                'status': 'shutting_down',
                'message': 'Service is shutting down gracefully',
                'active_requests': self.active_requests,
                'timestamp': datetime.now().isoformat()
            }

        # Get system metrics
        system_metrics = self.get_system_metrics()

        # Check model health
        model_health = self.check_model_health()

        # Check API health
        api_health = self.check_api_health()

        # Determine overall health
        overall_status = 'healthy'
        if model_health['status'] != 'healthy' or api_health['status'] != 'healthy':
            overall_status = 'unhealthy'

        # Check resource thresholds
        if system_metrics['memory_usage_mb'] > 1500:  # 1.5GB threshold
            overall_status = 'degraded'

        if system_metrics['cpu_usage_percent'] > 80:  # 80% CPU threshold
            overall_status = 'degraded'

        health_data = {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': system_metrics['uptime_seconds'],
            'system': {
                'memory_usage_mb': round(system_metrics['memory_usage_mb'], 2),
                'cpu_usage_percent': round(system_metrics['cpu_usage_percent'], 2),
                'memory_percent': round(system_metrics['memory_percent'], 2)
            },
            'models': model_health,
            'api': api_health,
            'requests': {
                'active': self.active_requests,
                'total_processed': len(self.health_metrics)
            }
        }

        # Store metrics for trend analysis
        self.health_metrics[datetime.now().isoformat()] = HealthMetrics(
            status=overall_status,
            response_time_ms=api_health.get('response_time_ms', 0),
            memory_usage_mb=system_metrics['memory_usage_mb'],
            cpu_usage_percent=system_metrics['cpu_usage_percent'],
            active_requests=self.active_requests,
            timestamp=datetime.now(),
            error_message=model_health.get('error') or api_health.get('error')
        )

        # Keep only last 100 metrics
        if len(self.health_metrics) > 100:
            oldest_key = min(self.health_metrics.keys())
            del self.health_metrics[oldest_key]

        return health_data

    def request_started(self):
        """Track request start"""
        with self.lock:
            self.active_requests += 1

    def request_completed(self):
        """Track request completion"""
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)

# Global health monitor instance
health_monitor = HealthMonitor()

def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance"""
    return health_monitor 
