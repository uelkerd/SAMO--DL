"""
Comprehensive Monitoring Dashboard for SAMO Deep Learning API

This module provides real-time monitoring capabilities including:
- System resource monitoring
- Model performance tracking
- API usage analytics
- Error tracking and alerting
- Performance metrics visualization
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import psutil

# Monitoring thresholds constants
CRITICAL_CPU_THRESHOLD = 90
CRITICAL_MEMORY_THRESHOLD = 90
CRITICAL_DISK_THRESHOLD = 95
WARNING_CPU_THRESHOLD = 80
WARNING_MEMORY_THRESHOLD = 80
WARNING_DISK_THRESHOLD = 85
CRITICAL_ERROR_RATE_THRESHOLD = 0.1  # 10%
MODEL_ERROR_COUNT_THRESHOLD = 10

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    last_used: Optional[float]
    is_loaded: bool
    error_count: int

@dataclass
class APIMetrics:
    """API usage metrics."""
    total_requests: int
    requests_per_minute: float
    average_response_time_ms: float
    error_rate: float
    active_connections: int
    uptime_seconds: float

class MonitoringDashboard:
    """Comprehensive monitoring dashboard for SAMO Deep Learning API."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.start_time = time.time()

        # Metrics storage
        self.system_metrics_history = deque(maxlen=history_size)
        self.model_metrics = defaultdict(lambda: ModelMetrics(
            model_name="",
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_response_time_ms=0.0,
            last_used=None,
            is_loaded=False,
            error_count=0
        ))
        self.api_metrics = APIMetrics(
            total_requests=0,
            requests_per_minute=0.0,
            average_response_time_ms=0.0,
            error_rate=0.0,
            active_connections=0,
            uptime_seconds=0.0
        )

        # Request tracking
        self.request_times = deque(maxlen=history_size)
        self.error_log = deque(maxlen=history_size)
        self.total_errors = 0  # Track total errors for accurate error rate

        # Performance tracking
        self.response_times = deque(maxlen=history_size)

        logger.info("Monitoring dashboard initialized")

    def update_system_metrics(self) -> SystemMetrics:
        """Update and store current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network metrics
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024 * 1024)
            network_recv_mb = network.bytes_recv / (1024 * 1024)

            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )

            self.system_metrics_history.append(metrics)
            return metrics

        except Exception as exc:
            logger.error(f"Failed to update system metrics: {exc}")
            return None

    def record_model_request(self, model_name: str, success: bool, response_time_ms: float):
        """Record a model request for metrics tracking."""
        metrics = self.model_metrics[model_name]
        metrics.model_name = model_name
        metrics.total_requests += 1

        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
            metrics.error_count += 1

        # Update average response time
        if metrics.average_response_time_ms == 0:
            metrics.average_response_time_ms = response_time_ms
        else:
            metrics.average_response_time_ms = (
                (metrics.average_response_time_ms * (metrics.total_requests - 1) + response_time_ms)
                / metrics.total_requests
            )

        metrics.last_used = time.time()

    def record_api_request(self, response_time_ms: float, success: bool):
        """Record an API request for metrics tracking."""
        self.api_metrics.total_requests += 1
        self.request_times.append(time.time())
        self.response_times.append(response_time_ms)

        if not success:
            self.error_log.append({
                "timestamp": time.time(),
                "error": "API request failed"
            })
            self.total_errors += 1

        # Update metrics
        self._update_api_metrics()

    def _update_api_metrics(self):
        """Update API metrics based on recent data."""
        current_time = time.time()

        # Calculate requests per minute
        one_minute_ago = current_time - 60
        recent_requests = sum(bool(t > one_minute_ago) for t in self.request_times)
        self.api_metrics.requests_per_minute = recent_requests

        # Calculate average response time
        if self.response_times:
            self.api_metrics.average_response_time_ms = sum(self.response_times) / len(self.response_times)

        # Calculate error rate
        if self.api_metrics.total_requests > 0:
            self.api_metrics.error_rate = self.total_errors / self.api_metrics.total_requests

        # Update uptime
        self.api_metrics.uptime_seconds = current_time - self.start_time

    def set_model_loaded_status(self, model_name: str, is_loaded: bool):
        """Set the loaded status of a model."""
        if model_name in self.model_metrics:
            self.model_metrics[model_name].is_loaded = is_loaded

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring metrics."""
        # Update system metrics
        current_system_metrics = self.update_system_metrics()

        # Update API metrics
        self._update_api_metrics()

        # Prepare model metrics
        model_metrics_dict = {model_name: asdict(metrics) for model_name, metrics in self.model_metrics.items()}

        # Calculate trends
        trends = self._calculate_trends()

        # Health status
        health_status = self._calculate_health_status()

        return {
            "timestamp": time.time(),
            "health_status": health_status,
            "system": asdict(current_system_metrics) if current_system_metrics else {},
            "models": model_metrics_dict,
            "api": asdict(self.api_metrics),
            "trends": trends,
            "alerts": self._generate_alerts()
        }

    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends."""
        if len(self.system_metrics_history) < 2:
            return {}

        recent_metrics = list(self.system_metrics_history)[-10:]  # Last 10 measurements

        cpu_trend = "stable"
        memory_trend = "stable"

        if len(recent_metrics) >= 2:
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]

            # Simple trend calculation
            cpu_slope = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
            memory_slope = (memory_values[-1] - memory_values[0]) / len(memory_values)

            if cpu_slope > 5:
                cpu_trend = "increasing"
            elif cpu_slope < -5:
                cpu_trend = "decreasing"

            if memory_slope > 2:
                memory_trend = "increasing"
            elif memory_slope < -2:
                memory_trend = "decreasing"

        return {
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "response_time_trend": "stable"  # Could be enhanced with more sophisticated analysis
        }

    def _calculate_health_status(self) -> str:
        """Calculate overall system health status."""
        if not self.system_metrics_history:
            return "unknown"

        current_metrics = self.system_metrics_history[-1]

        # Check critical thresholds first
        if (current_metrics.cpu_percent > CRITICAL_CPU_THRESHOLD or
            current_metrics.memory_percent > CRITICAL_MEMORY_THRESHOLD or
            current_metrics.disk_percent > CRITICAL_DISK_THRESHOLD):
            return "critical"

        # Check warning thresholds
        if (current_metrics.cpu_percent > WARNING_CPU_THRESHOLD or
            current_metrics.memory_percent > WARNING_MEMORY_THRESHOLD or
            current_metrics.disk_percent > WARNING_DISK_THRESHOLD or
            self.api_metrics.error_rate > CRITICAL_ERROR_RATE_THRESHOLD):
            return "warning"

        return "healthy"

    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on current metrics."""
        alerts = []

        if not self.system_metrics_history:
            return alerts

        current_metrics = self.system_metrics_history[-1]

        # System alerts
        if current_metrics.cpu_percent > CRITICAL_CPU_THRESHOLD:
            alerts.append({
                "level": "critical",
                "message": f"High CPU usage: {current_metrics.cpu_percent:.1f}%",
                "timestamp": current_metrics.timestamp
            })

        if current_metrics.memory_percent > CRITICAL_MEMORY_THRESHOLD:
            alerts.append({
                "level": "critical",
                "message": f"High memory usage: {current_metrics.memory_percent:.1f}%",
                "timestamp": current_metrics.timestamp
            })

        if current_metrics.disk_percent > CRITICAL_DISK_THRESHOLD:
            alerts.append({
                "level": "critical",
                "message": f"Low disk space: {100 - current_metrics.disk_percent:.1f}% free",
                "timestamp": current_metrics.timestamp
            })

        # API alerts
        if self.api_metrics.error_rate > CRITICAL_ERROR_RATE_THRESHOLD:
            alerts.append({
                "level": "warning",
                "message": f"High error rate: {self.api_metrics.error_rate:.1%}",
                "timestamp": time.time()
            })

        # Model alerts
        for model_name, metrics in self.model_metrics.items():
            if metrics.error_count > MODEL_ERROR_COUNT_THRESHOLD:
                alerts.append({
                    "level": "warning",
                    "message": f"High error count for {model_name}: {metrics.error_count} errors",
                    "timestamp": time.time()
                })

        return alerts

    def get_historical_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical data for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)

        # Filter system metrics
        historical_system = [
            asdict(metrics) for metrics in self.system_metrics_history
            if metrics.timestamp > cutoff_time
        ]

        # Filter response times
        historical_response_times = [
            rt for rt in self.response_times
            if rt > cutoff_time
        ]

        return {
            "system_metrics": historical_system,
            "response_times": historical_response_times,
            "period_hours": hours
        }

    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.system_metrics_history.clear()
        self.model_metrics.clear()
        self.request_times.clear()
        self.error_log.clear()
        self.response_times.clear()
        self.start_time = time.time()

        logger.info("Monitoring metrics reset")

# Global dashboard instance
dashboard = MonitoringDashboard()
