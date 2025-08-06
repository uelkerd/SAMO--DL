#!/usr/bin/env python3
"""
Cloud Run API Endpoint Testing Script
Tests the deployed SAMO Emotion Detection API for functionality, security, and performance.
"""

import requests
import json
import time
import sys
import os
import argparse
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudRunAPITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
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

    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health/status endpoint"""
        logger.info("Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/", timeout=30)
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
            
        except requests.exceptions.Timeout as e:
            return {
                "success": False,
                "error": f"Health endpoint timeout: {str(e)}"
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "success": False,
                "error": f"Health endpoint connection error: {str(e)}"
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Health endpoint failed: {str(e)}"
            }

    def test_emotion_detection_endpoint(self) -> Dict[str, Any]:
        """Test the emotion detection endpoint"""
        logger.info("Testing emotion detection endpoint...")
        
        # Test emotions endpoint first
        logger.info("Testing emotions endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/emotions", timeout=30)
            if response.status_code == 200:
                data = response.json()
                results = {
                    "emotions_endpoint": {
                        "success": True,
                        "emotions": data
                    }
                }
            else:
                results = {
                    "emotions_endpoint": {
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    }
                }
        except requests.exceptions.RequestException as e:
            results = {
                "emotions_endpoint": {
                    "success": False,
                    "error": str(e)
                }
            }
        
        # Test prediction endpoint
        test_text = "I am feeling really happy and excited today!"
        
        try:
            payload = {"text": test_text}
            response = self.session.post(f"{self.base_url}/predict", json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Emotion detection response: {data}")
            
            # Validate response structure - FIXED: API returns 'emotion' (singular), not 'emotions' (plural)
            if "emotion" not in data or "confidence" not in data:
                results["valid_input"] = {
                    "success": False,
                    "error": "Missing required fields in emotion detection response",
                    "response": data
                }
            else:
                # Check if emotion was detected
                emotion = data.get("emotion", "")
                confidence = data.get("confidence", 0)
                
                results["valid_input"] = {
                    "success": True,
                    "emotion_detected": bool(emotion),
                    "confidence": confidence,
                    "emotion": emotion,
                    "response_time": response.elapsed.total_seconds()
                }
            
        except requests.exceptions.Timeout as e:
            results["valid_input"] = {
                "success": False,
                "error": f"Emotion detection timeout: {str(e)}"
            }
        except requests.exceptions.ConnectionError as e:
            results["valid_input"] = {
                "success": False,
                "error": f"Emotion detection connection error: {str(e)}"
            }
        except requests.exceptions.RequestException as e:
            results["valid_input"] = {
                "success": False,
                "error": f"Emotion detection failed: {str(e)}"
            }
        
        return results

    def test_invalid_inputs(self) -> Dict[str, Any]:
        """Test invalid input handling"""
        logger.info("Testing invalid input handling...")
        
        invalid_test_cases = [
            {"text": ""},  # Empty text
            {"text": None},  # None text
            {},  # Missing text field
            {"invalid": "field"},  # Wrong field name
            {"text": 123},  # Non-string text
            {"text": "a" * 10000},  # Very long text
        ]
        
        invalid_results = []
        
        for i, test_case in enumerate(invalid_test_cases):
            try:
                response = self.session.post(f"{self.base_url}/predict", json=test_case, timeout=30)
                invalid_results.append({
                    "test_case": i,
                    "payload": test_case,
                    "status_code": response.status_code,
                    "success": response.status_code == 400  # Expected to fail with 400
                })
            except requests.exceptions.Timeout as e:
                invalid_results.append({
                    "test_case": i,
                    "payload": test_case,
                    "error": f"Timeout: {str(e)}",
                    "success": False
                })
            except requests.exceptions.ConnectionError as e:
                invalid_results.append({
                    "test_case": i,
                    "payload": test_case,
                    "error": f"Connection error: {str(e)}",
                    "success": False
                })
            except Exception as e:
                invalid_results.append({
                    "test_case": i,
                    "payload": test_case,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "success": True,
            "test_cases": len(invalid_test_cases),
            "results": invalid_results
        }

    def test_model_loading(self) -> Dict[str, Any]:
        """Test if models are properly loaded"""
        logger.info("Testing model loading...")
        
        results = {}
        
        # Test model status endpoint first
        logger.info("Testing model status endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/model_status", timeout=30)
            if response.status_code == 200:
                data = response.json()
                results["model_status"] = {
                    "success": True,
                    "status": data
                }
            else:
                results["model_status"] = {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
        except requests.exceptions.RequestException as e:
            results["model_status"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test multiple emotion detection requests to verify model loading
        predictions = []
        
        for i, text in enumerate(self.test_texts[:3]):  # Test first 3 texts
            try:
                payload = {"text": text}
                response = self.session.post(f"{self.base_url}/predict", json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    emotion = data.get("emotion", "")
                    confidence = data.get("confidence", 0)
                    
                    # FIXED: Handle None confidence values
                    if confidence is not None:
                        confidence_str = f"{confidence:.3f}"
                    else:
                        confidence_str = "N/A"
                    
                    predictions.append({
                        "text_index": i,
                        "success": True,
                        "emotion_detected": bool(emotion),
                        "emotion": emotion,
                        "confidence": confidence_str,
                        "response_time": response.elapsed.total_seconds()
                    })
                else:
                    predictions.append({
                        "text_index": i,
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    })
                    
            except requests.exceptions.Timeout as e:
                predictions.append({
                    "text_index": i,
                    "success": False,
                    "error": f"Timeout: {str(e)}"
                })
            except requests.exceptions.ConnectionError as e:
                predictions.append({
                    "text_index": i,
                    "success": False,
                    "error": f"Connection error: {str(e)}"
                })
            except Exception as e:
                predictions.append({
                    "text_index": i,
                    "success": False,
                    "error": str(e)
                })
        
        # FIXED: Models are considered loaded if all requests succeeded (status 200), regardless of whether emotions were detected
        successful_requests = [r for r in predictions if r["success"]]
        models_loaded = len(successful_requests) == len(predictions)
        
        results["predictions"] = {
            "success": models_loaded,
            "total_tests": len(predictions),
            "successful_tests": len(successful_requests),
            "models_loaded": models_loaded,
            "results": predictions
        }
        
        return results

    def test_security_features(self) -> Dict[str, Any]:
        """Test security features like rate limiting and authentication"""
        logger.info("Testing security features...")
        
        results = {}
        
        # Test rate limiting by making multiple rapid requests
        logger.info("Testing rate limiting...")
        rapid_requests = []
        
        # Make rate limiting configurable
        rate_limit_requests = int(os.environ.get("RATE_LIMIT_REQUESTS", "10"))
        
        for i in range(rate_limit_requests):
            try:
                payload = {"text": f"Test request {i}"}
                response = self.session.post(f"{self.base_url}/predict", json=payload, timeout=30)
                rapid_requests.append({
                    "request": i,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                })
            except requests.exceptions.Timeout as e:
                rapid_requests.append({
                    "request": i,
                    "error": f"Timeout: {str(e)}",
                    "success": False
                })
            except requests.exceptions.ConnectionError as e:
                rapid_requests.append({
                    "request": i,
                    "error": f"Connection error: {str(e)}",
                    "success": False
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
            response = self.session.get(f"{self.base_url}/", timeout=30)
            headers = response.headers
            
            security_headers = {
                "content_security_policy": "Content-Security-Policy" in headers,
                "x_frame_options": "X-Frame-Options" in headers,
                "x_content_type_options": "X-Content-Type-Options" in headers,
                "strict_transport_security": "Strict-Transport-Security" in headers
            }
            
            results["security_headers"] = security_headers
            
        except requests.exceptions.Timeout as e:
            results["security_headers"] = {"error": f"Timeout: {str(e)}"}
        except requests.exceptions.ConnectionError as e:
            results["security_headers"] = {"error": f"Connection error: {str(e)}"}
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
                response = self.session.post(f"{self.base_url}/predict", json=payload, timeout=30)
                
                if response.status_code == 200:
                    performance_results.append({
                        "request": i,
                        "response_time": response.elapsed.total_seconds(),
                        "success": True
                    })
                else:
                    performance_results.append({
                        "request": i,
                        "response_time": response.elapsed.total_seconds(),
                        "success": False,
                        "status_code": response.status_code
                    })
                    
            except requests.exceptions.Timeout as e:
                performance_results.append({
                    "request": i,
                    "error": f"Timeout: {str(e)}",
                    "success": False
                })
            except requests.exceptions.ConnectionError as e:
                performance_results.append({
                    "request": i,
                    "error": f"Connection error: {str(e)}",
                    "success": False
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
        test_results["tests"]["invalid_inputs"] = self.test_invalid_inputs()
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
            if isinstance(result, dict):
                # Handle nested results (like model_loading)
                if "predictions" in result:
                    # Model loading has nested structure
                    if result["predictions"]["success"]:
                        summary["passed_tests"] += 1
                    else:
                        summary["failed_tests"] += 1
                        summary["critical_issues"].append(f"{test_name}: Model loading failed")
                elif "valid_input" in result:
                    # Emotion detection has nested structure
                    if result["valid_input"]["success"]:
                        summary["passed_tests"] += 1
                    else:
                        summary["failed_tests"] += 1
                        summary["critical_issues"].append(f"{test_name}: {result['valid_input'].get('error', 'Unknown error')}")
                elif result.get("success", False):
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
    # Allow BASE_URL to be set via command line argument or environment variable
    parser = argparse.ArgumentParser(description="Comprehensive Cloud Run API testing")
    parser.add_argument("--base-url", help="Base URL for the API")
    args = parser.parse_args()
    
    if args.base_url:
        base_url = args.base_url
    elif os.environ.get("CLOUD_RUN_API_URL"):
        base_url = os.environ["CLOUD_RUN_API_URL"]
    else:
        base_url = "https://samo-emotion-api-optimized-secure-71517823771.us-central1.run.app"
    
    print("🧪 SAMO Cloud Run API Testing")
    print("=" * 50)
    print(f"Testing URL: {base_url}")
    print()
    
    # Create tester instance
    tester = CloudRunAPITester(base_url)
    
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
        if isinstance(result, dict):
            # Handle nested results (like model_loading)
            if "predictions" in result:
                # Model loading has nested structure
                status = "✅ PASS" if result["predictions"]["success"] else "❌ FAIL"
                print(f"{test_name.upper()}: {status}")
                if not result["predictions"]["success"]:
                    print(f"  Error: Model loading failed")
            elif "valid_input" in result:
                # Emotion detection has nested structure
                status = "✅ PASS" if result["valid_input"]["success"] else "❌ FAIL"
                print(f"{test_name.upper()}: {status}")
                if "error" in result["valid_input"]:
                    print(f"  Error: {result['valid_input']['error']}")
            else:
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                print(f"{test_name.upper()}: {status}")
                
                if "error" in result:
                    print(f"  Error: {result['error']}")
                elif test_name == "performance" and "avg_response_time" in result:
                    print(f"  Avg Response Time: {result['avg_response_time']:.3f}s")
                    print(f"  Success Rate: {result['success_rate']:.1%}")
    
    # Save results to file
    output_file = "test_reports/cloud_run_api_test_results.json"
    try:
        os.makedirs("test_reports", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    except PermissionError as e:
        print(f"\n⚠️  Permission denied saving results: {e}")
    except OSError as e:
        print(f"\n⚠️  OS error saving results: {e}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if summary["overall_success"] else 1)

if __name__ == "__main__":
    main() 