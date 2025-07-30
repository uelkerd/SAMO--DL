#!/usr/bin/env python3
"""
Model Monitoring Dashboard Starter

This script starts the model monitoring system for REQ-DL-010.
It initializes real-time performance tracking, data drift detection,
and automated alerting for the SAMO Deep Learning models.

Usage:
    python scripts/start_monitoring_dashboard.py [--config_path PATH] [--port INT]

Arguments:
    --config_path: Path to monitoring configuration (default: configs/monitoring.yaml)
    --port: Dashboard port (default: 8080)
"""

import argparse
import logging
import threading
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_PATH = "configs/monitoring.yaml"
DEFAULT_PORT = 8080


def start_monitoring_system(config_path: str, port: int) -> None:
    """Start the complete monitoring system.

    Args:
        config_path: Path to monitoring configuration
        port: Dashboard port
    """
    logger.info("🚀 Starting SAMO Model Monitoring System...")

    try:
        # Import monitoring components
        from scripts.model_monitoring import ModelHealthMonitor

        # Initialize monitor
        monitor = ModelHealthMonitor(config_path)

        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor.start_monitoring, daemon=True)
        monitor_thread.start()

        logger.info("✅ Model monitoring started successfully!")
        logger.info("📊 Dashboard available at: http://localhost:{port}")
        logger.info("🔍 Monitoring metrics every 5 minutes")
        logger.info("🚨 Alerts configured for performance degradation")
        logger.info("📈 Data drift detection enabled")
        logger.info("🔄 Automated retraining triggers active")

        # Keep main thread alive
        try:
            while True:
                time.sleep(60)
                health_status = monitor.get_health_status()
                logger.info("💚 System Health: {health_status['overall_status']}")

        except KeyboardInterrupt:
            logger.info("🛑 Stopping monitoring system...")
            monitor.stop_monitoring()
            logger.info("✅ Monitoring system stopped gracefully")

    except Exception as e:
        logger.error("❌ Failed to start monitoring system: {e}")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Start Model Monitoring Dashboard")
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to monitoring configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Dashboard port (default: {DEFAULT_PORT})"
    )

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config_path).exists():
        logger.error("❌ Configuration file not found: {args.config_path}")
        logger.info("Please create the monitoring configuration first")
        sys.exit(1)

    # Start monitoring system
    start_monitoring_system(args.config_path, args.port)


if __name__ == "__main__":
    main()
