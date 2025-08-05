#!/usr/bin/env python3
"""
Local Container Test Script
===========================

This script tests the Vertex AI container locally before deployment
to catch any startup issues early.
"""

import os
import sys
import subprocess
import time
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True, 
            timeout=300
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def test_docker_build():
    """Test Docker build process."""
    logger.info("🔨 Testing Docker build...")
    
    success, stdout, stderr = run_command(
        "docker build -t emotion-detection-test .",
        cwd="deployment/gcp"
    )
    
    if not success:
        logger.error(f"❌ Docker build failed: {stderr}")
        return False
    
    logger.info("✅ Docker build successful")
    return True

def test_container_startup():
    """Test container startup and health check."""
    logger.info("🚀 Testing container startup...")
    
    # Start container
    success, stdout, stderr = run_command(
        "docker run -d --name emotion-test -p 8080:8080 emotion-detection-test"
    )
    
    if not success:
        logger.error(f"❌ Container startup failed: {stderr}")
        return False
    
    logger.info("✅ Container started successfully")
    
    # Wait for container to be ready
    logger.info("⏳ Waiting for container to be ready...")
    time.sleep(30)
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8080/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Health check passed")
            health_data = response.json()
            logger.info(f"📋 Health data: {json.dumps(health_data, indent=2)}")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return False
    
    # Test prediction endpoint
    try:
        test_data = {"text": "I am feeling happy today!"}
        response = requests.post(
            "http://localhost:8080/predict", 
            json=test_data, 
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info("✅ Prediction test passed")
            prediction_data = response.json()
            logger.info(f"📋 Prediction: {prediction_data['predicted_emotion']} (confidence: {prediction_data['confidence']:.3f})")
        else:
            logger.error(f"❌ Prediction test failed: {response.status_code}")
            logger.error(f"📋 Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Prediction test error: {str(e)}")
        return False
    
    return True

def cleanup_container():
    """Clean up test container."""
    logger.info("🧹 Cleaning up test container...")
    
    # Stop and remove container
    run_command("docker stop emotion-test")
    run_command("docker rm emotion-test")
    run_command("docker rmi emotion-detection-test")
    
    logger.info("✅ Cleanup completed")

def main():
    """Main test function."""
    logger.info("🧪 Starting local container test...")
    
    try:
        # Test Docker build
        if not test_docker_build():
            logger.error("💥 Docker build test failed")
            return False
        
        # Test container startup
        if not test_container_startup():
            logger.error("💥 Container startup test failed")
            return False
        
        logger.info("🎉 All tests passed! Container is ready for deployment.")
        return True
        
    except Exception as e:
        logger.error(f"💥 Test failed with error: {str(e)}")
        return False
    
    finally:
        cleanup_container()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 