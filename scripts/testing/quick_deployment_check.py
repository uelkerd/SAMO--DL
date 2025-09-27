#!/usr/bin/env python3
"""
⚡ QUICK DEPLOYMENT STATUS CHECK
================================

Fast check of Cloud Run deployment status without full monitoring.
"""

import sys
import subprocess
import requests
from pathlib import Path

def check_deployment_status(service_name: str, region: str = "us-central1"):
    """Quick check of Cloud Run deployment status."""
    print("⚡ QUICK DEPLOYMENT CHECK")
    print("=" * 30)

    try:
        # Get service URL
        cmd = [
            "gcloud", "run", "services", "describe", service_name,
            "--region", region,
            "--format", "value(status.url)"
        ]

        print(f"🔍 Checking service: {service_name}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            print("❌ Service not found or not accessible")
            print(f"   Error: {result.stderr}")
            return False

        service_url = result.stdout.strip()
        if not service_url:
            print("❌ No service URL returned")
            return False

        print(f"✅ Service URL: {service_url}")

        # Quick health check
        try:
            response = requests.get(service_url, timeout=5)
            if response.status_code == 200:
                print("✅ Service responding (HTTP 200)")
            else:
                print(f"⚠️  Service responding (HTTP {response.status_code})")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Service not accessible: {e}")
            return False

        # Quick API test
        try:
            api_url = f"{service_url}/analyze"
            payload = {"text": "I feel happy!"}
            response = requests.post(api_url, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "primary_emotion" in data:
                    emotion = data.get("primary_emotion", "unknown")
                    print("🎯 API working!"                    print(f"   Sample result: {emotion}")
                    print()
                    print("🚀 DEPLOYMENT COMPLETE AND READY FOR TESTING!")
                    print(f"🌐 Service URL: {service_url}")
                    return service_url
                else:
                    print("⚠️  API responding but unexpected format")
                    return False
            else:
                print(f"❌ API error (HTTP {response.status_code})")
                return False

        except Exception as e:
            print(f"❌ API test failed: {e}")
            return False

    except subprocess.TimeoutExpired:
        print("⏱️  Command timed out")
        return False
    except Exception as e:
        print(f"❌ Check failed: {e}")
        return False


def main():
    """Quick deployment check."""
    service_name = "samo-emotion-api-deberta"  # Update if different
    region = "us-central1"

    result = check_deployment_status(service_name, region)

    if result:
        print("\n📋 NEXT STEPS:")
        print("1. ✅ Deployment confirmed working")
        print("2. 🔬 Ready for comprehensive testing")
        print("3. 📊 Run scientific test suite")
        print()
        print("🎯 Run comprehensive testing:")
        print("   python scripts/testing/scientific_cloud_run_testing.py")
        print()
        print("Or use the deployment monitor:")
        print("   python scripts/testing/cloud_run_deployment_monitor.py")
    else:
        print("\n❌ DEPLOYMENT ISSUES DETECTED")
        print("🔍 Check Cloud Run console")
        print("🔧 Troubleshoot deployment")
        print("⏰ Try again in a few minutes")


if __name__ == "__main__":
    main()
