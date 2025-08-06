#!/usr/bin/env python3
"""
Cloud Run API Endpoint Testing Script
Tests the deployed SAMO Emotion Detection API for functionality, security, and performance.
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudRunAPITester:
    def __init__(self, base_url: str, admin_api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.admin_api_key = admin_api_key
        self.session = requests.Session()
        
        # Test data
        self.test_texts = [
            "I am feeling really happy today!",
            "This makes me so angry and frustrated.",
            "I'm feeling sad and lonely.",
            "I'm excited about the new project!",
            "This is absolutely terrifying.",
            "I feel neutral about this situation.",
            "I'm so grateful for your help.",
            "This is disgusting and revolting.",
            "I'm feeling optimistic about the future.",
            "This is really confusing and puzzling."
        ]
        
        # Expected emotions for validation
        self.expected_emotions = [
            ["joy", "excitement"],
            ["anger", "frustration"],
            ["sadness", "grief"],
            ["excitement", "joy"],
            ["fear", "nervousness"],
            ["neutral"],
            ["gratitude", "joy"],
            ["disgust"],
            ["optimism", "joy"],
            ["confusion", "surprise"]
        ]

    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health/status endpoint"""
        logger.info("Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Health endpoint response: {data}")
            
            # Validate expected fields
            required_fields = ["status", "service", "version", "security", "rate_limit"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return {
                    "success": False,
                    "error": f"Missing required fields: {missing_fields}",
                    "response": data
                }
            
            return {
                "success": True,
                "status": data.get("status"),
                "version": data.get("version"),
                "security_enabled": data.get("security") == "enabled",
                "rate_limit": data.get("rate_limit")
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Health endpoint failed: {str(e)}"
            }

    def test_emotion_detection_endpoint(self) -> Dict[str, Any]:
        """Test the emotion detection endpoint"""
        logger.info("Testing emotion detection endpoint...")
        
        test_text = "I am feeling really happy and excited today!"
        
        try:
            payload = {"text": test_text}
            response = self.session.post(f"{self.base_url}/predict", json=payload)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Emotion detection response: {data}")
            
            # Validate response structure
            if "emotions" not in data or "confidence" not in data:
                return {
                    "success": False,
                    "error": "Missing required fields in emotion detection response",
                    "response": data
                }
            
            # Check if emotions were detected
            emotions = data.get("emotions", [])
            confidence = data.get("confidence", 0)
            
            return {
                "success": True,
                "emotions_detected": len(emotions) > 0,
                "confidence": confidence,
                "emotions": emotions,
                "response_time": response.elapsed.total_seconds()
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Emotion detection failed: {str(e)}"
            }

    def test_model_loading(self) -> Dict[str, Any]:
        """Test if models are properly loaded"""
        logger.info("Testing model loading...")
        
        # Test multiple emotion detection requests to verify model loading
        results = []
        
        for i, text in enumerate(self.test_texts[:3]):  # Test first 3 texts
            try:
                payload = {"text": text}
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/predict", json=payload)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    emotions = data.get("emotions", [])
                    confidence = data.get("confidence", 0)
                    
                    results.append({
                        "text_index": i,
                        "success": True,
                        "emotions_detected": len(emotions) > 0,
                        "confidence": confidence,
                        "response_time": response_time
                    })
                else:
                    results.append({
                        "text_index": i,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    })
                    
            except Exception as e:
                results.append({
                    "text_index": i,
                    "success": False,
                    "error": str(e)
                })
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        models_loaded = len(successful_requests) > 0 and all(r["emotions_detected"] for r in successful_requests)
        
        return {
            "success": models_loaded,
            "total_tests": len(results),
            "successful_tests": len(successful_requests),
            "models_loaded": models_loaded,
            "results": results
        }

    def test_security_features(self) -> Dict[str, Any]:
        """Test security features like rate limiting and authentication"""
        logger.info("Testing security features...")
        
        results = {}
        
        # Test rate limiting by making multiple rapid requests
        logger.info("Testing rate limiting...")
        rapid_requests = []
        for i in range(5):  # Make 5 rapid requests
            try:
                payload = {"text": f"Test request {i}"}
                response = self.session.post(f"{self.base_url}/predict", json=payload)
                rapid_requests.append({
                    "request": i,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                })
            except Exception as e:
                rapid_requests.append({
                    "request": i,
                    "error": str(e),
                    "success": False
                })
        
        # Check if any requests were rate limited (429 status)
        rate_limited = any(r.get("status_code") == 429 for r in rapid_requests)
        results["rate_limiting"] = {
            "tested": True,
            "rate_limited": rate_limited,
            "requests": rapid_requests
        }
        
        # Test security headers
        logger.info("Testing security headers...")
        try:
            response = self.session.get(f"{self.base_url}/")
            headers = response.headers
            
            security_headers = {
                "content_security_policy": "Content-Security-Policy" in headers,
                "x_frame_options": "X-Frame-Options" in headers,
                "x_content_type_options": "X-Content-Type-Options" in headers,
                "strict_transport_security": "Strict-Transport-Security" in headers
            }
            
            results["security_headers"] = security_headers
            
        except Exception as e:
            results["security_headers"] = {"error": str(e)}
        
        return results

    def test_performance(self) -> Dict[str, Any]:
        """Test API performance metrics"""
        logger.info("Testing performance...")
        
        performance_results = []
        
        for i, text in enumerate(self.test_texts[:5]):  # Test first 5 texts
            try:
                payload = {"text": text}
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/predict", json=payload)
                end_time = time.time()
                
                if response.status_code == 200:
                    performance_results.append({
                        "request": i,
                        "response_time": end_time - start_time,
                        "success": True
                    })
                else:
                    performance_results.append({
                        "request": i,
                        "response_time": end_time - start_time,
                        "success": False,
                        "status_code": response.status_code
                    })
                    
            except Exception as e:
                performance_results.append({
                    "request": i,
                    "error": str(e),
                    "success": False
                })
        
        # Calculate performance metrics
        successful_requests = [r for r in performance_results if r["success"]]
        
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        return {
            "total_requests": len(performance_results),
            "successful_requests": len(successful_requests),
            "success_rate": len(successful_requests) / len(performance_results) if performance_results else 0,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "min_response_time": min_response_time,
            "results": performance_results
        }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        logger.info("Starting comprehensive API testing...")
        
        test_results = {
            "timestamp": time.time(),
            "base_url": self.base_url,
            "tests": {}
        }
        
        # Run all tests
        test_results["tests"]["health"] = self.test_health_endpoint()
        test_results["tests"]["emotion_detection"] = self.test_emotion_detection_endpoint()
        test_results["tests"]["model_loading"] = self.test_model_loading()
        test_results["tests"]["security"] = self.test_security_features()
        test_results["tests"]["performance"] = self.test_performance()
        
        # Generate summary
        test_results["summary"] = self.generate_summary(test_results["tests"])
        
        return test_results

    def generate_summary(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of test results"""
        summary = {
            "overall_success": True,
            "passed_tests": 0,
            "failed_tests": 0,
            "critical_issues": []
        }
        
        for test_name, result in tests.items():
            if isinstance(result, dict) and result.get("success", False):
                summary["passed_tests"] += 1
            else:
                summary["failed_tests"] += 1
                if test_name in ["health", "model_loading"]:
                    summary["critical_issues"].append(f"{test_name}: {result.get('error', 'Unknown error')}")
        
        # Check for critical failures
        if summary["failed_tests"] > 0:
            summary["overall_success"] = False
        
        return summary

def main():
    """Main function to run the API tests"""
    # Configuration
    BASE_URL = "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app"
    
    print("🧪 SAMO Cloud Run API Testing")
    print("=" * 50)
    print(f"Testing URL: {BASE_URL}")
    print()
    
    # Create tester instance
    tester = CloudRunAPITester(BASE_URL)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Print results
    print("📊 Test Results Summary")
    print("=" * 50)
    
    summary = results["summary"]
    print(f"Overall Success: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}")
    print(f"Tests Passed: {summary['passed_tests']}")
    print(f"Tests Failed: {summary['failed_tests']}")
    
    if summary["critical_issues"]:
        print("\n🚨 Critical Issues:")
        for issue in summary["critical_issues"]:
            print(f"  - {issue}")
    
    # Print detailed results
    print("\n📋 Detailed Results:")
    print("-" * 30)
    
    for test_name, result in results["tests"].items():
        status = "✅ PASS" if isinstance(result, dict) and result.get("success", False) else "❌ FAIL"
        print(f"{test_name.upper()}: {status}")
        
        if isinstance(result, dict):
            if "error" in result:
                print(f"  Error: {result['error']}")
            elif test_name == "performance" and "avg_response_time" in result:
                print(f"  Avg Response Time: {result['avg_response_time']:.3f}s")
                print(f"  Success Rate: {result['success_rate']:.1%}")
    
    # Save results to file
    output_file = "test_reports/cloud_run_api_test_results.json"
    try:
        import os
        os.makedirs("test_reports", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if summary["overall_success"] else 1)

if __name__ == "__main__":
    main() 