#!/usr/bin/env python3
"""
SAMO Unified API Server Startup Script

This script provides a convenient way to start the SAMO unified API server
with proper configuration and error handling.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.unified_api_server import SAMOUnifiedAPIServer


def main():
    """Main entry point for starting the API server."""
    parser = argparse.ArgumentParser(description="Start SAMO Unified API Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)"
    )
    parser.add_argument(
        "--config",
        default="configs/samo_api_config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Load configuration file if it exists
    config = {}
    if Path(args.config).exists():
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"✅ Loaded configuration from {args.config}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load config file {args.config}: {e}")
            logger.info("Using default configuration")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    try:
        logger.info("🚀 Starting SAMO Unified API Server")
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Reload: {args.reload}")
        logger.info(f"Config: {args.config}")

        # Create and start server with configuration
        server = SAMOUnifiedAPIServer()
        
        # Apply configuration if available
        if config:
            # Apply server configuration
            if 'server' in config:
                server_config = config['server']
                if 'cors_origins' in server_config:
                    os.environ['API_ALLOWED_ORIGINS'] = ','.join(server_config['cors_origins'])
                logger.info("✅ Applied server configuration")

        logger.info("✅ Server initialized successfully")
        logger.info("📖 API Documentation: http://localhost:8000/docs")
        logger.info("🔄 ReDoc Documentation: http://localhost:8000/redoc")
        logger.info("💚 Health Check: http://localhost:8000/health")

        # Start the server
        server.run(host=args.host, port=args.port)

    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
