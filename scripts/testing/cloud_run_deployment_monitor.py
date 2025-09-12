#!/usr/bin/env python3
"""
🚀 CLOUD RUN DEPLOYMENT MONITOR
==============================

Monitors Cloud Run deployment status and automatically triggers comprehensive testing.
"""

import sys
import time
import requests
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

class CloudRunMonitor:
    """Monitor Cloud Run deployment and trigger testing when ready."""

    def __init__(self, service_name: str, region: str = "us-central1"):
        self.service_name = service_name
        self.region = region
        self.service_url: Optional[str] = None

    def get_service_url(self) -> Optional[str]:
        """Get the Cloud Run service URL using gcloud CLI."""
        try:
            cmd = [
                "gcloud", "run", "services", "describe", self.service_name,
                "--region", self.region,
                "--format", "value(status.url)"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0 and result.stdout.strip():
                url = result.stdout.strip()
                print(f"✅ Service URL found: {url}")
                return url
            else:
                print(f"❌ Failed to get service URL: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("⏱️  Timeout getting service URL")
            return None
        except Exception as e:
            print(f"❌ Error getting service URL: {e}")
            return None

    def test_service_health(self, url: str) -> bool:
        """Test if the service is responding and healthy."""
        try:
            # Test root endpoint
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                print("✅ Service is responding (HTTP 200)")
                return True
            else:
                print(f"⚠️  Service responding but not healthy (HTTP {response.status_code})")
                return False

        except requests.exceptions.ConnectionError:
            print("🔌 Service not yet accessible (connection error)")
            return False
        except requests.exceptions.Timeout:
            print("⏱️  Service timeout")
            return False
        except Exception as e:
            print(f"❌ Service health check error: {e}")
            return False

    def test_api_endpoint(self, url: str) -> bool:
        """Test the actual API endpoint with a sample request."""
        try:
            api_url = f"{url}/analyze"  # Assuming standard endpoint
            payload = {"text": "I feel happy today!"}
            headers = {"Content-Type": "application/json"}

            response = requests.post(api_url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "primary_emotion" in data:
                    print("🎯 API endpoint working correctly!")
                    print(f"   Primary emotion: {data.get('primary_emotion', 'unknown')}")
                    return True
                else:
                    print("⚠️  API responding but unexpected response format")
                    return False
            else:
                print(f"❌ API endpoint error (HTTP {response.status_code})")
                return False

        except Exception as e:
            print(f"❌ API test error: {e}")
            return False

    def wait_for_deployment(self, max_wait_minutes: int = 15) -> Optional[str]:
        """Wait for deployment to complete and return service URL."""
        print("🚀 MONITORING CLOUD RUN DEPLOYMENT")
        print("=" * 50)
        print(f"📍 Service: {self.service_name}")
        print(f"🏗️  Region: {self.region}")
        print(f"⏱️  Max wait: {max_wait_minutes} minutes")
        print()

        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60

        while time.time() - start_time < max_wait_seconds:
            elapsed = time.time() - start_time

            print(f"🔍 Checking deployment status... ({elapsed:.0f}s elapsed)")

            # Get service URL
            service_url = self.get_service_url()

            if service_url:
                print(f"📡 Found service URL: {service_url}")

                # Test service health
                if self.test_service_health(service_url):
                    print("🏥 Service health check passed")

                    # Test API endpoint
                    if self.test_api_endpoint(service_url):
                        print("🎉 DEPLOYMENT COMPLETE AND FULLY FUNCTIONAL!")
                        print(f"🌐 Service URL: {service_url}")
                        return service_url
                    else:
                        print("⚠️  Service healthy but API not working yet")
                else:
                    print("⏳ Service found but not healthy yet")
            else:
                print("⏳ Service not yet available")

            print("⏰ Waiting 30 seconds before next check...")
            time.sleep(30)

        print("❌ DEPLOYMENT TIMEOUT - Service not ready within time limit")
        return None

    def trigger_comprehensive_testing(self, service_url: str):
        """Trigger the comprehensive scientific testing suite."""
        print("\n🚀 INITIATING COMPREHENSIVE SCIENTIFIC TESTING")
        print("=" * 60)

        # Update the test script with the service URL
        test_script_path = Path(__file__).parent / "scientific_cloud_run_testing.py"

        print(f"📝 Test script: {test_script_path}")
        print(f"🎯 Target URL: {service_url}")
        print()

        # Run the comprehensive test suite
        print("🔬 STARTING SCIENTIFIC TESTING PROTOCOL:")
        print("   1. ✅ Comprehensive API testing (10 runs × 5 entries)")
        print("   2. ✅ Load testing (20 concurrent requests)")
        print("   3. ✅ Reliability testing (50 iterations)")
        print("   4. ✅ Statistical analysis with confidence intervals")
        print("   5. ✅ Performance benchmarking")
        print("   6. ✅ Edge case testing")
        print("   7. ✅ Consistency analysis")
        print()

        # Execute the testing
        try:
            cmd = [sys.executable, str(test_script_path)]
            env = os.environ.copy()
            env["CLOUD_RUN_URL"] = service_url

            print("⚡ EXECUTING COMPREHENSIVE TEST SUITE...")
            result = subprocess.run(cmd, env=env, timeout=600)  # 10 minute timeout

            if result.returncode == 0:
                print("🎉 COMPREHENSIVE TESTING COMPLETED SUCCESSFULLY!")
            else:
                print(f"⚠️  Testing completed with exit code: {result.returncode}")

        except subprocess.TimeoutExpired:
            print("⏱️  Testing timed out (10 minutes)")
        except Exception as e:
            print(f"❌ Testing execution error: {e}")


def main():
    """Main deployment monitoring execution."""
    print("🚀 CLOUD RUN DEPLOYMENT MONITOR")
    print("=" * 40)

    # Configuration
    service_name = "samo-emotion-api-deberta"  # Update if different
    region = "us-central1"

    monitor = CloudRunMonitor(service_name, region)

    print("📋 DEPLOYMENT MONITORING PLAN:")
    print("   1. Monitor deployment progress")
    print("   2. Wait for service to become healthy")
    print("   3. Test API endpoints")
    print("   4. Trigger comprehensive scientific testing")
    print("   5. Generate detailed performance report")
    print()

    # Wait for deployment
    service_url = monitor.wait_for_deployment(max_wait_minutes=15)

    if service_url:
        print("\n🎯 DEPLOYMENT SUCCESSFUL!")
        print(f"🌐 Service URL: {service_url}")

        # Ask user if they want to proceed with testing
        response = input("\n🔬 Ready to start comprehensive scientific testing? (y/n): ").lower().strip()

        if response in ['y', 'yes']:
            monitor.trigger_comprehensive_testing(service_url)
        else:
            print("⏸️  Testing postponed. You can run testing manually later.")
            print(f"💡 When ready, run: python scripts/testing/scientific_cloud_run_testing.py")
    else:
        print("\n❌ DEPLOYMENT FAILED OR TIMED OUT")
        print("🔍 Check Cloud Run console for deployment status")
        print("🔧 Troubleshoot any deployment issues")
        print("📞 Contact DevOps if needed")


if __name__ == "__main__":
    main()
