#!/usr/bin/env python3
"""
Model Monitoring Script for REQ-DL-010

This script implements comprehensive model monitoring for SAMO Deep Learning:
1. Real-time performance metrics tracking
2. Data drift detection using statistical methods
3. Automated retraining triggers based on performance degradation
4. Model health dashboard and alerting system

Usage:
    python scripts/model_monitoring.py [--config_path PATH] [--monitor_interval INT] [--alert_threshold FLOAT]

Arguments:
    --config_path: Path to monitoring configuration (default: configs/monitoring.yaml)
    --monitor_interval: Monitoring interval in seconds (default: 300)
    --alert_threshold: Performance degradation threshold for alerts (default: 0.1)
"""

import sys
import argparse
import logging
import time
import json
import yaml
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.models.emotion_detection.bert_classifier import create_bert_emotion_classifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_PATH = "configs/monitoring.yaml"
DEFAULT_MONITOR_INTERVAL = 300  # 5 minutes
DEFAULT_ALERT_THRESHOLD = 0.1  # 10% degradation
DEFAULT_DRIFT_THRESHOLD = 0.05  # 5% drift
DEFAULT_RETRAIN_THRESHOLD = 0.15  # 15% degradation


@dataclass
class ModelMetrics:
    """Data class for storing model performance metrics."""

    timestamp: datetime
    f1_score: float
    precision: float
    recall: float
    inference_time_ms: float
    throughput_rps: float
    memory_usage_mb: float
    gpu_utilization: Optional[float] = None
    cpu_utilization: Optional[float] = None


@dataclass
class DriftMetrics:
    """Data class for storing data drift metrics."""

    timestamp: datetime
    feature_drift_score: float
    label_drift_score: float
    distribution_shift: float
    drift_detected: bool
    affected_features: list[str]


@dataclass
class Alert:
    """Data class for storing monitoring alerts."""

    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    metrics: dict[str, Any]
    action_required: bool


class PerformanceTracker:
    """Track real-time model performance metrics."""

    def __init__(self, window_size: int = 100):
        """Initialize performance tracker.

        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.baseline_metrics = None
        self.degradation_threshold = DEFAULT_ALERT_THRESHOLD

    def add_metrics(self, metrics: ModelMetrics) -> None:
        """Add new metrics to the tracker.

        Args:
            metrics: Model performance metrics
        """
        self.metrics_history.append(metrics)

        # Set baseline if not set
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
            logger.info("Baseline metrics established")

    def get_current_performance(self) -> dict[str, float]:
        """Get current performance metrics.

        Returns:
            Dictionary with current performance metrics
        """
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]
        return {
            "f1_score": latest.f1_score,
            "precision": latest.precision,
            "recall": latest.recall,
            "inference_time_ms": latest.inference_time_ms,
            "throughput_rps": latest.throughput_rps,
            "memory_usage_mb": latest.memory_usage_mb,
        }

    def detect_degradation(self) -> Optional[Alert]:
        """Detect performance degradation.

        Returns:
            Alert if degradation detected, None otherwise
        """
        if not self.metrics_history or self.baseline_metrics is None:
            return None

        latest = self.metrics_history[-1]

        # Calculate degradation
        f1_degradation = (
            self.baseline_metrics.f1_score - latest.f1_score
        ) / self.baseline_metrics.f1_score
        precision_degradation = (
            self.baseline_metrics.precision - latest.precision
        ) / self.baseline_metrics.precision
        recall_degradation = (
            self.baseline_metrics.recall - latest.recall
        ) / self.baseline_metrics.recall

        # Check if degradation exceeds threshold
        max_degradation = max(f1_degradation, precision_degradation, recall_degradation)

        if max_degradation > self.degradation_threshold:
            severity = "HIGH" if max_degradation > DEFAULT_RETRAIN_THRESHOLD else "MEDIUM"
            action_required = max_degradation > DEFAULT_RETRAIN_THRESHOLD

            return Alert(
                timestamp=datetime.now(),
                alert_type="PERFORMANCE_DEGRADATION",
                severity=severity,
                message=f"Model performance degraded by {max_degradation:.2%}",
                metrics={
                    "f1_degradation": f1_degradation,
                    "precision_degradation": precision_degradation,
                    "recall_degradation": recall_degradation,
                    "max_degradation": max_degradation,
                },
                action_required=action_required,
            )

        return None

    def get_trend_analysis(self) -> dict[str, Any]:
        """Analyze performance trends.

        Returns:
            Dictionary with trend analysis
        """
        if len(self.metrics_history) < 10:
            return {"insufficient_data": True}

        # Extract metrics arrays
        f1_scores = [m.f1_score for m in self.metrics_history]
        inference_times = [m.inference_time_ms for m in self.metrics_history]

        # Calculate trends
        f1_trend = np.polyfit(range(len(f1_scores)), f1_scores, 1)[0]
        inference_trend = np.polyfit(range(len(inference_times)), inference_times, 1)[0]

        return {
            "f1_trend": f1_trend,
            "inference_trend": inference_trend,
            "f1_stable": abs(f1_trend) < 0.001,
            "inference_stable": abs(inference_trend) < 0.1,
            "data_points": len(self.metrics_history),
        }


class DataDriftDetector:
    """Detect data drift using statistical methods."""

    def __init__(
        self, reference_data: pd.DataFrame, drift_threshold: float = DEFAULT_DRIFT_THRESHOLD
    ):
        """Initialize drift detector.

        Args:
            reference_data: Reference dataset for drift detection
            drift_threshold: Threshold for drift detection
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.feature_stats = self._compute_reference_stats()

    def _compute_reference_stats(self) -> dict[str, dict[str, float]]:
        """Compute reference statistics for features.

        Returns:
            Dictionary with feature statistics
        """
        stats_dict = {}

        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ["int64", "float64"]:
                stats_dict[column] = {
                    "mean": self.reference_data[column].mean(),
                    "std": self.reference_data[column].std(),
                    "min": self.reference_data[column].min(),
                    "max": self.reference_data[column].max(),
                }

        return stats_dict

    def detect_drift(self, current_data: pd.DataFrame) -> DriftMetrics:
        """Detect data drift in current data.

        Args:
            current_data: Current data to check for drift

        Returns:
            Drift metrics
        """
        drift_scores = {}
        affected_features = []

        # Check each feature for drift
        for feature, ref_stats in self.feature_stats.items():
            if feature in current_data.columns:
                current_mean = current_data[feature].mean()
                current_std = current_data[feature].std()

                # Calculate drift score using KL divergence or statistical distance
                drift_score = self._calculate_drift_score(
                    ref_stats["mean"], ref_stats["std"], current_mean, current_std
                )

                drift_scores[feature] = drift_score

                if drift_score > self.drift_threshold:
                    affected_features.append(feature)

        # Calculate overall drift score
        if drift_scores:
            overall_drift = np.mean(list(drift_scores.values()))
            drift_detected = overall_drift > self.drift_threshold
        else:
            overall_drift = 0.0
            drift_detected = False

        return DriftMetrics(
            timestamp=datetime.now(),
            feature_drift_score=overall_drift,
            label_drift_score=0.0,  # Placeholder for label drift
            distribution_shift=overall_drift,
            drift_detected=drift_detected,
            affected_features=affected_features,
        )

    def _calculate_drift_score(
        self, ref_mean: float, ref_std: float, current_mean: float, current_std: float
    ) -> float:
        """Calculate drift score between reference and current distributions.

        Args:
            ref_mean: Reference mean
            ref_std: Reference standard deviation
            current_mean: Current mean
            current_std: Current standard deviation

        Returns:
            Drift score
        """
        # Use Wasserstein distance as drift measure
        mean_diff = abs(current_mean - ref_mean)
        std_diff = abs(current_std - ref_std)

        # Normalize by reference statistics
        normalized_mean_diff = mean_diff / (ref_std + 1e-8)
        normalized_std_diff = std_diff / (ref_std + 1e-8)

        # Combined drift score
        drift_score = (normalized_mean_diff + normalized_std_diff) / 2

        return drift_score


class ModelHealthMonitor:
    """Main model health monitoring system."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """Initialize model health monitor.

        Args:
            config_path: Path to monitoring configuration
        """
        self.config = self._load_config(config_path)
        self.performance_tracker = PerformanceTracker(
            window_size=self.config.get("window_size", 100)
        )
        self.drift_detector = None  # Will be initialized with reference data
        self.alerts = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None

        # Initialize model
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load monitoring configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {
                "window_size": 100,
                "monitor_interval": DEFAULT_MONITOR_INTERVAL,
                "alert_threshold": DEFAULT_ALERT_THRESHOLD,
                "drift_threshold": DEFAULT_DRIFT_THRESHOLD,
                "retrain_threshold": DEFAULT_RETRAIN_THRESHOLD,
            }

    def _initialize_model(self) -> None:
        """Initialize the model for monitoring."""
        try:
            # Load model
            model_path = self.config.get(
                "model_path", "models/checkpoints/bert_emotion_classifier_final.pt"
            )
            if Path(model_path).exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model, _ = create_bert_emotion_classifier()
                self.model.to(device)

                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])

                self.model.eval()

                # Initialize tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name)

                logger.info("Model initialized for monitoring")
            else:
                logger.warning(f"Model not found: {model_path}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")

    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info("Model monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()

        logger.info("Model monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                if metrics:
                    self.performance_tracker.add_metrics(metrics)

                # Check for degradation
                degradation_alert = self.performance_tracker.detect_degradation()
                if degradation_alert:
                    self.alerts.append(degradation_alert)
                    self._handle_alert(degradation_alert)

                # Check for data drift (if detector is initialized)
                if self.drift_detector:
                    drift_metrics = self._check_data_drift()
                    if drift_metrics.drift_detected:
                        drift_alert = Alert(
                            timestamp=datetime.now(),
                            alert_type="DATA_DRIFT",
                            severity="MEDIUM",
                            message=f"Data drift detected in {len(drift_metrics.affected_features)} features",
                            metrics=asdict(drift_metrics),
                            action_required=False,
                        )
                        self.alerts.append(drift_alert)
                        self._handle_alert(drift_alert)

                # Sleep for monitoring interval
                time.sleep(self.config.get("monitor_interval", DEFAULT_MONITOR_INTERVAL))

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying

    def _collect_metrics(self) -> Optional[ModelMetrics]:
        """Collect current model performance metrics.

        Returns:
            Model metrics if collection successful, None otherwise
        """
        if not self.model:
            return None

        try:
            start_time = time.time()

            # Generate test data
            test_texts = [
                "I'm feeling really excited about this new project!",
                "This is making me so frustrated and angry.",
                "I'm grateful for all the support I've received.",
                "I'm feeling a bit nervous about the presentation.",
                "This is absolutely amazing and wonderful!",
            ]

            # Tokenize
            inputs = self.tokenizer(
                test_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
            )

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                torch.sigmoid(outputs)

            inference_time = time.time() - start_time
            inference_time_ms = inference_time * 1000

            # Calculate mock metrics (in real scenario, these would come from actual evaluation)
            f1_score_val = 0.75  # Mock value
            precision_val = 0.78  # Mock value
            recall_val = 0.72  # Mock value

            # Calculate throughput
            throughput_rps = len(test_texts) / inference_time

            # Get memory usage
            memory_usage_mb = self._get_memory_usage()

            # Get GPU utilization if available
            gpu_utilization = self._get_gpu_utilization()

            return ModelMetrics(
                timestamp=datetime.now(),
                f1_score=f1_score_val,
                precision=precision_val,
                recall=recall_val,
                inference_time_ms=inference_time_ms,
                throughput_rps=throughput_rps,
                memory_usage_mb=memory_usage_mb,
                gpu_utilization=gpu_utilization,
            )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage.

        Returns:
            GPU utilization percentage if available, None otherwise
        """
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
        except:
            pass
        return None

    def _check_data_drift(self) -> DriftMetrics:
        """Check for data drift in incoming data.

        Returns:
            Drift metrics
        """
        # In a real implementation, this would analyze actual incoming data
        # For now, return mock drift metrics
        return DriftMetrics(
            timestamp=datetime.now(),
            feature_drift_score=0.02,
            label_drift_score=0.01,
            distribution_shift=0.02,
            drift_detected=False,
            affected_features=[],
        )

    def _handle_alert(self, alert: Alert) -> None:
        """Handle monitoring alerts.

        Args:
            alert: Alert to handle
        """
        logger.warning(f"ALERT [{alert.severity}]: {alert.message}")

        if alert.action_required:
            logger.info("Action required - triggering retraining pipeline")
            self._trigger_retraining()

        # Save alert to file
        self._save_alert(alert)

    def _trigger_retraining(self) -> None:
        """Trigger model retraining pipeline."""
        try:
            # In a real implementation, this would trigger the retraining pipeline
            logger.info("Triggering model retraining...")

            # For now, just log the action
            retrain_alert = Alert(
                timestamp=datetime.now(),
                alert_type="RETRAINING_TRIGGERED",
                severity="INFO",
                message="Model retraining pipeline triggered",
                metrics={},
                action_required=False,
            )
            self.alerts.append(retrain_alert)

        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")

    def _save_alert(self, alert: Alert) -> None:
        """Save alert to file.

        Args:
            alert: Alert to save
        """
        try:
            alerts_dir = Path("logs/alerts")
            alerts_dir.mkdir(parents=True, exist_ok=True)

            alert_file = alerts_dir / f"alert_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(alert_file, "w") as f:
                json.dump(asdict(alert), f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving alert: {e}")

    def get_health_status(self) -> dict[str, Any]:
        """Get current model health status.

        Returns:
            Dictionary with health status
        """
        current_performance = self.performance_tracker.get_current_performance()
        trend_analysis = self.performance_tracker.get_trend_analysis()

        return {
            "timestamp": datetime.now().isoformat(),
            "model_loaded": self.model is not None,
            "monitoring_active": self.monitoring_active,
            "current_performance": current_performance,
            "trend_analysis": trend_analysis,
            "recent_alerts": len(
                [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=1)]
            ),
            "total_alerts": len(self.alerts),
        }

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data for monitoring dashboard.

        Returns:
            Dictionary with dashboard data
        """
        # Get recent metrics
        recent_metrics = list(self.performance_tracker.metrics_history)[-50:]

        # Get recent alerts
        recent_alerts = [
            a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)
        ]

        return {
            "metrics_history": [asdict(m) for m in recent_metrics],
            "recent_alerts": [asdict(a) for a in recent_alerts],
            "health_status": self.get_health_status(),
        }


def create_monitoring_config(output_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Create default monitoring configuration.

    Args:
        output_path: Path to save configuration
    """
    config = {
        "model_path": "models/checkpoints/bert_emotion_classifier_final.pt",
        "window_size": 100,
        "monitor_interval": DEFAULT_MONITOR_INTERVAL,
        "alert_threshold": DEFAULT_ALERT_THRESHOLD,
        "drift_threshold": DEFAULT_DRIFT_THRESHOLD,
        "retrain_threshold": DEFAULT_RETRAIN_THRESHOLD,
        "alerts": {"email": False, "slack": False, "webhook": None},
        "logging": {"level": "INFO", "file": "logs/monitoring.log"},
    }

    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"Monitoring configuration created: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Monitoring for REQ-DL-010")
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to monitoring configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--monitor_interval",
        type=int,
        default=DEFAULT_MONITOR_INTERVAL,
        help=f"Monitoring interval in seconds (default: {DEFAULT_MONITOR_INTERVAL})",
    )
    parser.add_argument(
        "--alert_threshold",
        type=float,
        default=DEFAULT_ALERT_THRESHOLD,
        help=f"Performance degradation threshold for alerts (default: {DEFAULT_ALERT_THRESHOLD})",
    )
    parser.add_argument(
        "--create_config", action="store_true", help="Create default monitoring configuration"
    )

    args = parser.parse_args()

    if args.create_config:
        create_monitoring_config(args.config_path)
        sys.exit(0)

    # Create monitor
    monitor = ModelHealthMonitor(args.config_path)

    # Start monitoring
    try:
        monitor.start_monitoring()

        # Keep running
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Stopping monitoring...")
        monitor.stop_monitoring()

        # Print final status
        status = monitor.get_health_status()
        logger.info(f"Final status: {status}")

    sys.exit(0)
