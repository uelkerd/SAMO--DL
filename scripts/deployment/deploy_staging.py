#!/usr/bin/env python3
"""
🚀 STAGING DEPLOYMENT SCRIPT
============================
Deploy SAMO-DL API to staging environment with comprehensive testing.
"""

import os
import json
import subprocess
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_ID = "the-tendril-466607-n8"
REGION = "us-central1"
SERVICE_NAME = "samo-dl-api-staging"
IMAGE_NAME = f"us-central1-docker.pkg.dev/{PROJECT_ID}/samo-dl/samo-dl-api-staging"
PORT = 8080

def print_banner():
    """Print deployment banner"""
    print("🚀" * 50)
    print("🎯 SAMO-DL STAGING DEPLOYMENT")
    print("📅", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🚀" * 50)

def check_prerequisites():
    """Check deployment prerequisites"""
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 40)
    
    # Check gcloud CLI
    try:
        result = subprocess.run(['gcloud', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ gcloud CLI installed")
        else:
            print("❌ gcloud CLI not working")
            return False
    except FileNotFoundError:
        print("❌ gcloud CLI not installed")
        return False
    
    # Check authentication
    try:
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'], 
                              capture_output=True, text=True)
        if 'ACTIVE' in result.stdout:
            print("✅ gcloud authenticated")
        else:
            print("❌ gcloud not authenticated")
            return False
    except Exception as e:
        print(f"❌ Authentication check failed: {e}")
        return False
    
    # Check project
    try:
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                              capture_output=True, text=True)
        if PROJECT_ID in result.stdout:
            print(f"✅ Project set to {PROJECT_ID}")
        else:
            print(f"❌ Project not set to {PROJECT_ID}")
            return False
    except Exception as e:
        print(f"❌ Project check failed: {e}")
        return False
    
    return True

def build_docker_image():
    """Build Docker image for staging"""
    print("\n🐳 BUILDING DOCKER IMAGE")
    print("=" * 40)
    
    try:
        # Build the image
        cmd = [
            'docker', 'build',
            '-f', 'Dockerfile.optimized',
            '-t', f'{IMAGE_NAME}:latest',
            '-t', f'{IMAGE_NAME}:{int(time.time())}',
            '.'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker image built successfully")
            return True
        else:
            print("❌ Docker build failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False

def push_docker_image():
    """Push Docker image to Artifact Registry"""
    print("\n📤 PUSHING DOCKER IMAGE")
    print("=" * 40)
    
    try:
        # Configure Docker authentication
        auth_cmd = ['gcloud', 'auth', 'configure-docker', 'us-central1-docker.pkg.dev']
        subprocess.run(auth_cmd, check=True)
        
        # Push the image
        push_cmd = ['docker', 'push', f'{IMAGE_NAME}:latest']
        print(f"Running: {' '.join(push_cmd)}")
        result = subprocess.run(push_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker image pushed successfully")
            return True
        else:
            print("❌ Docker push failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Docker push error: {e}")
        return False

def deploy_to_cloud_run():
    """Deploy to Cloud Run staging"""
    print("\n🚀 DEPLOYING TO CLOUD RUN STAGING")
    print("=" * 40)
    
    try:
        # Deploy command
        deploy_cmd = [
            'gcloud', 'run', 'deploy', SERVICE_NAME,
            '--image', f'{IMAGE_NAME}:latest',
            '--region', REGION,
            '--platform', 'managed',
            '--allow-unauthenticated',
            '--port', str(PORT),
            '--memory', '2Gi',
            '--cpu', '2',
            '--max-instances', '5',
            '--min-instances', '0',
            '--timeout', '300',
            '--concurrency', '40',
            '--set-env-vars', 'ENVIRONMENT=staging,DEBUG=true,LOG_LEVEL=debug'
        ]
        
        print(f"Running: {' '.join(deploy_cmd)}")
        result = subprocess.run(deploy_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Cloud Run deployment successful")
            return True
        else:
            print("❌ Cloud Run deployment failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Cloud Run deployment error: {e}")
        return False

def get_service_url():
    """Get the deployed service URL"""
    try:
        cmd = [
            'gcloud', 'run', 'services', 'describe', SERVICE_NAME,
            '--region', REGION,
            '--format', 'value(status.url)'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        print(f"❌ Error getting service URL: {e}")
        return None

def run_integration_tests(service_url):
    """Run comprehensive integration tests"""
    print("\n🧪 RUNNING INTEGRATION TESTS")
    print("=" * 40)
    
    if not service_url:
        print("❌ No service URL available for testing")
        return False
    
    print(f"Testing service at: {service_url}")
    
    # Test cases
    test_cases = [
        {
            'name': 'Health Check',
            'url': f'{service_url}/health',
            'method': 'GET',
            'expected_status': 200
        },
        {
            'name': 'Root Endpoint',
            'url': f'{service_url}/',
            'method': 'GET',
            'expected_status': 200
        },
        {
            'name': 'Emotion Analysis',
            'url': f'{service_url}/analyze/emotion',
            'method': 'POST',
            'expected_status': 200,
            'data': {'text': 'I am feeling happy today!'}
        },
        {
            'name': 'Text Summarization',
            'url': f'{service_url}/analyze/summarize',
            'method': 'POST',
            'expected_status': 200,
            'data': {'text': 'This is a long text that should be summarized properly by the API.'}
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for test in test_cases:
        print(f"\n🔍 Testing: {test['name']}")
        try:
            if test['method'] == 'GET':
                response = requests.get(test['url'], timeout=30)
            else:
                response = requests.post(
                    test['url'], 
                    json=test.get('data', {}), 
                    timeout=30
                )
            
            if response.status_code == test['expected_status']:
                print(f"✅ {test['name']} - Status: {response.status_code}")
                passed_tests += 1
            else:
                print(f"❌ {test['name']} - Expected: {test['expected_status']}, Got: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {test['name']} - Request failed: {e}")
        except Exception as e:
            print(f"❌ {test['name']} - Error: {e}")
    
    print(f"\n📊 TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests

def main():
    """Main deployment function"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Build Docker image
    if not build_docker_image():
        print("\n❌ Docker build failed. Exiting.")
        sys.exit(1)
    
    # Push Docker image
    if not push_docker_image():
        print("\n❌ Docker push failed. Exiting.")
        sys.exit(1)
    
    # Deploy to Cloud Run
    if not deploy_to_cloud_run():
        print("\n❌ Cloud Run deployment failed. Exiting.")
        sys.exit(1)
    
    # Get service URL
    service_url = get_service_url()
    if not service_url:
        print("\n❌ Could not get service URL. Exiting.")
        sys.exit(1)
    
    print(f"\n🎉 DEPLOYMENT SUCCESSFUL!")
    print(f"🌐 Service URL: {service_url}")
    
    # Wait for service to be ready
    print("\n⏳ Waiting for service to be ready...")
    time.sleep(30)
    
    # Run integration tests
    if run_integration_tests(service_url):
        print("\n🎉 ALL TESTS PASSED! Staging deployment is ready.")
    else:
        print("\n⚠️ Some tests failed. Check the service logs.")
    
    print(f"\n📋 STAGING DEPLOYMENT SUMMARY")
    print(f"   Service: {SERVICE_NAME}")
    print(f"   URL: {service_url}")
    print(f"   Region: {REGION}")
    print(f"   Project: {PROJECT_ID}")

if __name__ == "__main__":
    main()
