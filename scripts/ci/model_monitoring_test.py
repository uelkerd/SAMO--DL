#!/usr/bin/env python3
"""
CI Test for Model Monitoring System

This script tests the model monitoring system to ensure it works correctly
in the CI pipeline without requiring actual model training or long-running processes.

Usage:
    python scripts/ci/model_monitoring_test.py
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_monitoring_imports():
    """Test that all monitoring dependencies can be imported."""
    logger.info("Testing monitoring imports...")

    try:
        import numpy as np
        import pandas as pd
        from scipy import stats
        from sklearn.metrics import f1_score
        import torch
        from transformers import AutoTokenizer
        import yaml

        logger.info("✅ All monitoring dependencies imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False


def test_performance_tracker():
    """Test the PerformanceTracker class."""
    logger.info("Testing PerformanceTracker...")

    try:
        # Import the monitoring module
        sys.path.append(str(Path(__file__).parent.parent.resolve()))
        from model_monitoring import PerformanceTracker, ModelMetrics

        # Create tracker
        tracker = PerformanceTracker(window_size=10)

        # Create mock metrics
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            f1_score=0.75,
            precision=0.78,
            recall=0.72,
            inference_time_ms=150.0,
            throughput_rps=33.3,
            memory_usage_mb=512.0,
        )

        # Add metrics
        tracker.add_metrics(metrics)

        # Test current performance
        current_perf = tracker.get_current_performance()
        if "f1_score" not in current_perf:
            raise AssertionError
        if current_perf["f1_score"] != 0.75:
            raise AssertionError

        # Test trend analysis
        trend = tracker.get_trend_analysis()
        if "insufficient_data" not in trend:
            raise AssertionError

        # Test degradation detection
        degradation_alert = tracker.detect_degradation()
        assert degradation_alert is None  # No degradation with single data point

        logger.info("✅ PerformanceTracker tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ PerformanceTracker test failed: {e}")
        return False


def test_drift_detector():
    """Test the DataDriftDetector class."""
    logger.info("Testing DataDriftDetector...")

    try:
        import pandas as pd
        from model_monitoring import DataDriftDetector, DriftMetrics

        # Create mock reference data
        reference_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 1000),
                "feature2": np.random.normal(5, 2, 1000),
                "feature3": np.random.uniform(0, 10, 1000),
            }
        )

        # Create drift detector
        detector = DataDriftDetector(reference_data, drift_threshold=0.05)

        # Test with similar data (no drift)
        current_data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(5, 2, 100),
                "feature3": np.random.uniform(0, 10, 100),
            }
        )

        drift_metrics = detector.detect_drift(current_data)
        assert isinstance(drift_metrics, DriftMetrics)
        if drift_metrics.drift_detected:
            raise AssertionError

        # Test with drifted data
        drifted_data = pd.DataFrame(
            {
                "feature1": np.random.normal(2, 1, 100),  # Shifted mean
                "feature2": np.random.normal(5, 2, 100),
                "feature3": np.random.uniform(0, 10, 100),
            }
        )

        drifted_metrics = detector.detect_drift(drifted_data)
        assert isinstance(drifted_metrics, DriftMetrics)

        logger.info("✅ DataDriftDetector tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ DataDriftDetector test failed: {e}")
        return False


def test_alert_system():
    """Test the alert system."""
    logger.info("Testing alert system...")

    try:
        from model_monitoring import Alert

        # Create test alert
        alert = Alert(
            timestamp=datetime.now(),
            alert_type="TEST_ALERT",
            severity="MEDIUM",
            message="Test alert message",
            metrics={"test_metric": 0.5},
            action_required=False,
        )

        # Test alert properties
        if alert.alert_type != "TEST_ALERT":
            raise AssertionError
        if alert.severity != "MEDIUM":
            raise AssertionError
        if alert.message != "Test alert message":
            raise AssertionError
        if alert.action_required:
            raise AssertionError

        # Test alert serialization
        alert_dict = {
            "timestamp": alert.timestamp.isoformat(),
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "metrics": alert.metrics,
            "action_required": alert.action_required,
        }

        # Test JSON serialization
        alert_json = json.dumps(alert_dict)
        assert isinstance(alert_json, str)

        logger.info("✅ Alert system tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Alert system test failed: {e}")
        return False


def test_config_creation():
    """Test monitoring configuration creation."""
    logger.info("Testing configuration creation...")

    try:
        from model_monitoring import create_monitoring_config

        # Create test config
        test_config_path = "test_monitoring_config.yaml"
        create_monitoring_config(test_config_path)

        # Check if config file was created
        config_path = Path(test_config_path)
        if not config_path.exists():
            raise AssertionError

        # Load and validate config
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check required fields
        if "model_path" not in config:
            raise AssertionError
        if "window_size" not in config:
            raise AssertionError
        if "monitor_interval" not in config:
            raise AssertionError
        if "alert_threshold" not in config:
            raise AssertionError

        # Clean up
        config_path.unlink()

        logger.info("✅ Configuration creation tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Configuration creation test failed: {e}")
        return False


def test_monitoring_initialization():
    """Test monitoring system initialization."""
    logger.info("Testing monitoring initialization...")

    try:
        from model_monitoring import ModelHealthMonitor

        # Create monitor with test config
        test_config = {
            "model_path": "nonexistent_model.pt",  # Use nonexistent model for testing
            "window_size": 10,
            "monitor_interval": 60,
            "alert_threshold": 0.1,
            "drift_threshold": 0.05,
            "retrain_threshold": 0.15,
        }

        # Save test config
        test_config_path = "test_config.yaml"
        import yaml

        with open(test_config_path, "w") as f:
            yaml.dump(test_config, f)

        # Create monitor
        monitor = ModelHealthMonitor(test_config_path)

        # Test basic properties
        if monitor.config["window_size"] != 10:
            raise AssertionError
        if monitor.config["alert_threshold"] != 0.1:
            raise AssertionError
        if monitor.monitoring_active:
            raise AssertionError

        # Test health status
        health_status = monitor.get_health_status()
        if "timestamp" not in health_status:
            raise AssertionError
        if "model_loaded" not in health_status:
            raise AssertionError
        if "monitoring_active" not in health_status:
            raise AssertionError

        # Clean up
        Path(test_config_path).unlink()

        logger.info("✅ Monitoring initialization tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Monitoring initialization test failed: {e}")
        return False


def test_metrics_collection():
    """Test metrics collection functionality."""
    logger.info("Testing metrics collection...")

    try:
        from model_monitoring import ModelMetrics

        # Create mock metrics
        metrics = ModelMetrics(
            timestamp=datetime.now(),
            f1_score=0.75,
            precision=0.78,
            recall=0.72,
            inference_time_ms=150.0,
            throughput_rps=33.3,
            memory_usage_mb=512.0,
            gpu_utilization=45.0,
        )

        # Test metrics properties
        if metrics.f1_score != 0.75:
            raise AssertionError
        if metrics.precision != 0.78:
            raise AssertionError
        if metrics.recall != 0.72:
            raise AssertionError
        if metrics.inference_time_ms != 150.0:
            raise AssertionError
        if metrics.throughput_rps != 33.3:
            raise AssertionError
        if metrics.memory_usage_mb != 512.0:
            raise AssertionError
        if metrics.gpu_utilization != 45.0:
            raise AssertionError

        # Test metrics serialization
        metrics_dict = {
            "timestamp": metrics.timestamp.isoformat(),
            "f1_score": metrics.f1_score,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "inference_time_ms": metrics.inference_time_ms,
            "throughput_rps": metrics.throughput_rps,
            "memory_usage_mb": metrics.memory_usage_mb,
            "gpu_utilization": metrics.gpu_utilization,
        }

        # Test JSON serialization
        metrics_json = json.dumps(metrics_dict)
        assert isinstance(metrics_json, str)

        logger.info("✅ Metrics collection tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Metrics collection test failed: {e}")
        return False


def main():
    """Run all monitoring tests."""
    logger.info("🧪 Starting Model Monitoring Tests...")

    tests = [
        ("Monitoring Imports", test_monitoring_imports),
        ("Performance Tracker", test_performance_tracker),
        ("Drift Detector", test_drift_detector),
        ("Alert System", test_alert_system),
        ("Configuration Creation", test_config_creation),
        ("Monitoring Initialization", test_monitoring_initialization),
        ("Metrics Collection", test_metrics_collection),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")

    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")

    if failed == 0:
        logger.info("🎉 All monitoring tests passed!")
        return 0
    else:
        logger.error(f"❌ {failed} monitoring tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
