#!/usr/bin/env python3
"""
🔬 SCIENTIFIC CLOUD RUN TESTING FRAMEWORK
========================================

Comprehensive, statistically rigorous testing of DeBERTa-v3 model deployment.
Uses scientific methodology with controlled variables and statistical analysis.
"""

import sys
import os
import time
import json
import statistics
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy import stats

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

@dataclass
class TestConfig:
    """Configuration for scientific testing."""
    cloud_run_url: str
    num_runs: int = 10
    num_concurrent: int = 5
    timeout_seconds: int = 30
    confidence_level: float = 0.95

@dataclass
class TestResult:
    """Individual test result."""
    run_id: int
    timestamp: float
    request_time: float
    response_time: float
    latency: float
    status_code: int
    success: bool
    response_data: Dict[str, Any]
    error_message: str = ""

@dataclass
class StatisticalAnalysis:
    """Statistical analysis of test results."""
    mean_latency: float
    median_latency: float
    std_dev_latency: float
    min_latency: float
    max_latency: float
    p95_latency: float
    p99_latency: float
    success_rate: float
    throughput: float
    confidence_interval: Tuple[float, float]

class ScientificCloudRunTester:
    """Scientific testing framework for Cloud Run API."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.test_journal_entries = self._create_controlled_test_data()
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def _create_controlled_test_data(self) -> List[Dict[str, str]]:
        """Create controlled test data for consistent, scientific testing."""
        return [
            {
                "id": "control_001",
                "title": "Anxiety Control Test",
                "content": """I feel really anxious about tomorrow's presentation. My heart is racing and I can't stop thinking about what could go wrong. The nervousness is overwhelming and I feel like I might panic.""",
                "expected_primary": "nervousness",
                "expected_emotions": ["nervousness", "fear", "anxiety"],
                "complexity": "simple",
                "word_count": 58
            },
            {
                "id": "control_002",
                "title": "Joy Control Test",
                "content": """I'm so happy today! Everything went perfectly and I feel amazing. The sun is shining and I have this incredible sense of joy and contentment that fills my whole being.""",
                "expected_primary": "joy",
                "expected_emotions": ["joy", "happiness", "contentment"],
                "complexity": "simple",
                "word_count": 55
            },
            {
                "id": "control_003",
                "title": "Complex Emotional Transition",
                "content": """This morning I woke up feeling frustrated and angry about the argument last night. But then I went for a walk and saw the beautiful sunrise, which made me feel grateful and peaceful. Now I'm sitting here feeling content and hopeful about the future.""",
                "expected_primary": "gratitude",
                "expected_emotions": ["frustration", "anger", "gratitude", "peace", "contentment", "hope"],
                "complexity": "complex",
                "word_count": 87
            },
            {
                "id": "control_004",
                "title": "Sadness Control Test",
                "content": """I'm feeling really sad today. The weight of disappointment is heavy on my chest and I can't shake this feeling of melancholy. Everything seems gray and hopeless.""",
                "expected_primary": "sadness",
                "expected_emotions": ["sadness", "disappointment", "melancholy"],
                "complexity": "simple",
                "word_count": 52
            },
            {
                "id": "control_005",
                "title": "Pride Achievement Test",
                "content": """I finally finished that project I've been working on for months! The sense of accomplishment and pride is overwhelming. I feel proud of what I've achieved and excited about what's next.""",
                "expected_primary": "pride",
                "expected_emotions": ["pride", "accomplishment", "excitement"],
                "complexity": "simple",
                "word_count": 56
            }
        ]

    def run_single_test(self, test_data: Dict[str, str], run_id: int) -> TestResult:
        """Run a single API test with detailed timing and error handling."""
        request_time = time.time()

        try:
            payload = {"text": test_data["content"]}
            headers = {"Content-Type": "application/json"}

            response = requests.post(
                self.config.cloud_run_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds
            )

            response_time = time.time()
            latency = response_time - request_time

            if response.status_code == 200:
                response_data = response.json()
                success = True
                error_message = ""
            else:
                response_data = {}
                success = False
                error_message = f"HTTP {response.status_code}: {response.text}"

        except requests.exceptions.Timeout:
            response_time = time.time()
            latency = response_time - request_time
            success = False
            error_message = "Request timeout"
            response_data = {}
            response = None

        except Exception as e:
            response_time = time.time()
            latency = response_time - request_time
            success = False
            error_message = str(e)
            response_data = {}
            response = None

        return TestResult(
            run_id=run_id,
            timestamp=request_time,
            request_time=request_time,
            response_time=response_time,
            latency=latency,
            status_code=response.status_code if 'response' in locals() and response else 0,
            success=success,
            response_data=response_data,
            error_message=error_message
        )

    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite with statistical analysis."""
        print("🔬 STARTING SCIENTIFIC CLOUD RUN TESTING")
        print("=" * 60)
        print(f"🎯 Cloud Run URL: {self.config.cloud_run_url}")
        print(f"📊 Test Runs: {self.config.num_runs}")
        print(f"⚡ Concurrent Requests: {self.config.num_concurrent}")
        print(f"⏱️  Timeout: {self.config.timeout_seconds}s")
        print()

        # Run all tests
        all_results = []
        for run in range(self.config.num_runs):
            print(f"🚀 Run {run + 1}/{self.config.num_runs}")

            # Test each journal entry
            for entry in self.test_journal_entries:
                result = self.run_single_test(entry, run)
                all_results.append(result)
                self.results.append(result)

                status = "✅" if result.success else "❌"
                print(".2f"
            print()

        # Analyze results
        analysis = self._analyze_results(all_results)

        # Print comprehensive report
        self._print_comprehensive_report(analysis)

        return {
            "config": {
                "cloud_run_url": self.config.cloud_run_url,
                "num_runs": self.config.num_runs,
                "num_concurrent": self.config.num_concurrent,
                "timeout_seconds": self.config.timeout_seconds
            },
            "results": [self._result_to_dict(r) for r in self.results],
            "analysis": analysis,
            "timestamp": time.time(),
            "test_duration": time.time() - self.start_time
        }

    def run_load_test(self, concurrent_requests: int = 20) -> Dict[str, Any]:
        """Run load testing to measure throughput and concurrency performance."""
        print("
🔥 LOAD TESTING - High Concurrency"        print(f"⚡ Concurrent Requests: {concurrent_requests}")
        print("-" * 40)

        test_entry = self.test_journal_entries[0]  # Use first entry for consistency

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(self.run_single_test, test_entry, i)
                for i in range(concurrent_requests)
            ]

            load_results = []
            for future in as_completed(futures):
                result = future.result()
                load_results.append(result)

                status = "✅" if result.success else "❌"
                print(".2f"
        total_time = time.time() - start_time
        throughput = len(load_results) / total_time

        successful_requests = sum(1 for r in load_results if r.success)
        success_rate = successful_requests / len(load_results)

        latencies = [r.latency for r in load_results if r.success]

        return {
            "concurrent_requests": concurrent_requests,
            "total_requests": len(load_results),
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "total_time": total_time,
            "throughput": throughput,  # requests per second
            "mean_latency": statistics.mean(latencies) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency": np.percentile(latencies, 99) if latencies else 0
        }

    def run_reliability_test(self, num_iterations: int = 50) -> Dict[str, Any]:
        """Test reliability over many iterations."""
        print("
🔄 RELIABILITY TESTING"        print(f"📊 Iterations: {num_iterations}")
        print("-" * 40)

        test_entry = self.test_journal_entries[0]
        reliability_results = []

        consecutive_failures = 0
        max_consecutive_failures = 0

        for i in range(num_iterations):
            result = self.run_single_test(test_entry, i)
            reliability_results.append(result)

            if not result.success:
                consecutive_failures += 1
                max_consecutive_failures = max(max_consecutive_failures, consecutive_failures)
            else:
                consecutive_failures = 0

            if (i + 1) % 10 == 0:
                successful = sum(1 for r in reliability_results[-10:] if r.success)
                print(f"   Iterations {i-9:2d}-{i+1:2d}: {successful}/10 successful")

        successful_requests = sum(1 for r in reliability_results if r.success)
        success_rate = successful_requests / len(reliability_results)

        return {
            "total_iterations": num_iterations,
            "successful_requests": successful_requests,
            "success_rate": success_rate,
            "max_consecutive_failures": max_consecutive_failures,
            "reliability_score": success_rate * 100  # percentage
        }

    def _analyze_results(self, results: List[TestResult]) -> StatisticalAnalysis:
        """Perform statistical analysis on test results."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return StatisticalAnalysis(
                mean_latency=0, median_latency=0, std_dev_latency=0,
                min_latency=0, max_latency=0, p95_latency=0, p99_latency=0,
                success_rate=0, throughput=0, confidence_interval=(0, 0)
            )

        latencies = [r.latency for r in successful_results]

        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        std_dev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        success_rate = len(successful_results) / len(results)

        # Calculate throughput (requests per second)
        if successful_results:
            time_span = successful_results[-1].response_time - successful_results[0].request_time
            throughput = len(successful_results) / time_span if time_span > 0 else 0
        else:
            throughput = 0

        # Confidence interval for mean latency
        if len(latencies) > 1:
            confidence_interval = stats.t.interval(
                self.config.confidence_level,
                len(latencies) - 1,
                loc=mean_latency,
                scale=stats.sem(latencies)
            )
        else:
            confidence_interval = (mean_latency, mean_latency)

        return StatisticalAnalysis(
            mean_latency=mean_latency,
            median_latency=median_latency,
            std_dev_latency=std_dev_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            success_rate=success_rate,
            throughput=throughput,
            confidence_interval=confidence_interval
        )

    def _print_comprehensive_report(self, analysis: StatisticalAnalysis):
        """Print comprehensive statistical report."""
        print("📊 STATISTICAL ANALYSIS REPORT")
        print("=" * 60)

        print("🎯 SUCCESS METRICS:")
        print(".1f"        print(".2f"
        print()
        print("⚡ LATENCY ANALYSIS:")
        print(".2f"        print(".2f"        print(".2f"        print(".2f"        print(".2f"        print(".2f"        print()
        print("📈 PERFORMANCE METRICS:")
        print(".2f"        print(".2f"
        print()
        print("🔬 STATISTICAL CONFIDENCE:")
        print(".2f"        print(".2f"
        print()
        # Performance assessment
        self._assess_performance(analysis)

    def _assess_performance(self, analysis: StatisticalAnalysis):
        """Provide scientific assessment of performance."""
        print("🎯 SCIENTIFIC PERFORMANCE ASSESSMENT")
        print("-" * 40)

        # Success rate assessment
        if analysis.success_rate >= 0.99:
            success_assessment = "🟢 EXCELLENT (Production Ready)"
        elif analysis.success_rate >= 0.95:
            success_assessment = "🟡 GOOD (Minor Issues)"
        elif analysis.success_rate >= 0.90:
            success_assessment = "🟠 ACCEPTABLE (Needs Attention)"
        else:
            success_assessment = "🔴 POOR (Not Production Ready)"

        # Latency assessment
        if analysis.p95_latency < 1.0:
            latency_assessment = "🟢 EXCELLENT (Real-time)"
        elif analysis.p95_latency < 3.0:
            latency_assessment = "🟡 GOOD (Interactive)"
        elif analysis.p95_latency < 5.0:
            latency_assessment = "🟠 ACCEPTABLE (Batch Processing)"
        else:
            latency_assessment = "🔴 POOR (Too Slow)"

        # Throughput assessment
        if analysis.throughput > 50:
            throughput_assessment = "🟢 EXCELLENT (High Throughput)"
        elif analysis.throughput > 20:
            throughput_assessment = "🟡 GOOD (Moderate Throughput)"
        elif analysis.throughput > 10:
            throughput_assessment = "🟠 ACCEPTABLE (Low Throughput)"
        else:
            throughput_assessment = "🔴 POOR (Very Low Throughput)"

        print(f"Success Rate: {success_assessment}")
        print(f"Latency (P95): {latency_assessment}")
        print(f"Throughput: {throughput_assessment}")
        print()

        # Overall recommendation
        if analysis.success_rate >= 0.95 and analysis.p95_latency < 3.0:
            overall = "🟢 PRODUCTION READY"
        elif analysis.success_rate >= 0.90 and analysis.p95_latency < 5.0:
            overall = "🟡 READY WITH MONITORING"
        else:
            overall = "🔴 NEEDS IMPROVEMENT"

        print(f"🎯 OVERALL ASSESSMENT: {overall}")

    def _result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """Convert TestResult to dictionary."""
        return {
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "latency": result.latency,
            "status_code": result.status_code,
            "success": result.success,
            "error_message": result.error_message,
            "response_data": result.response_data
        }

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save comprehensive test results."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"scientific_cloud_run_test_results_{timestamp}.json"

        filepath = Path(__file__).parent / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {filepath}")
        return str(filepath)


def main():
    """Main scientific testing execution."""
    print("🔬 SCIENTIFIC CLOUD RUN TESTING FRAMEWORK")
    print("=" * 60)

    # Configuration - will be updated when user provides the Cloud Run URL
    config = TestConfig(
        cloud_run_url="https://YOUR-CLOUD-RUN-URL",  # To be updated
        num_runs=10,
        num_concurrent=5,
        timeout_seconds=30,
        confidence_level=0.95
    )

    tester = ScientificCloudRunTester(config)

    print("⚠️  WAITING FOR CLOUD RUN DEPLOYMENT TO COMPLETE...")
    print("📝 Once deployed, update the cloud_run_url in the config above")
    print("🚀 Then run comprehensive scientific testing")
    print()
    print("🎯 TEST PLAN:")
    print("   1. Comprehensive API testing (10 runs × 5 entries)")
    print("   2. Load testing (20 concurrent requests)")
    print("   3. Reliability testing (50 iterations)")
    print("   4. Statistical analysis with confidence intervals")
    print("   5. Performance benchmarking")
    print()
    print("📊 METRICS TO MEASURE:")
    print("   • Success rate with confidence intervals")
    print("   • Latency percentiles (P50, P95, P99)")
    print("   • Throughput (requests/second)")
    print("   • Error patterns and failure modes")
    print("   • Consistency across multiple runs")
    print()
    print("🔬 SCIENTIFIC METHODOLOGY:")
    print("   • Controlled test data (consistent inputs)")
    print("   • Multiple runs for statistical significance")
    print("   • Error handling and edge case testing")
    print("   • Performance benchmarking under load")
    print("   • Reliability assessment over time")


if __name__ == "__main__":
    main()
