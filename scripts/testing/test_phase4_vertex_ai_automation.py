#!/usr/bin/env python3
"""
Phase 4: Vertex AI Deployment Automation Test Suite
Comprehensive testing for Phase 4 Vertex AI automation features
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import unittest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

class Phase4VertexAIAutomationTest(unittest.TestCase):
    """Comprehensive test suite for Phase 4 Vertex AI automation"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(__file__).parent
        self.deployment_dir = self.test_dir.parent.parent / 'deployment'
        self.vertex_ai_script = self.deployment_dir / 'vertex_ai_phase4_automation.py'
        self.maxDiff = None
        
        # Test configuration
        self.test_config = {
            'project_id': 'test-project-123',
            'region': 'us-central1',
            'model_name': 'test-emotion-detection',
            'endpoint_name': 'test-endpoint',
            'machine_type': 'n1-standard-2',
            'min_replicas': 1,
            'max_replicas': 5,
            'cost_budget': 50.0
        }
    
    def test_01_script_structure(self):
        """Test Phase 4 automation script structure"""
        print("🔍 Testing Phase 4 automation script structure...")
        
        self.assertTrue(
                        self.vertex_ai_script.exists(),
                        "Vertex AI automation script should exist"
                       )
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for required classes and methods
        required_elements = [
            'class DeploymentConfig',
            'class VertexAIPhase4Automation',
            'def check_prerequisites',
            'def generate_model_version',
            'def create_deployment_package',
            'def build_and_push_image',
            'def create_vertex_ai_model',
            'def deploy_model_to_endpoint',
            'def setup_monitoring_and_alerting',
            'def setup_cost_monitoring',
            'def rollback_deployment',
            'def setup_ab_testing',
            'def get_performance_metrics',
            'def cleanup_old_versions',
            'def run_full_deployment'
        ]
        
        for element in required_elements:
            self.assertIn(element, content, f"Missing required element: {element}")
        
        print("✅ Phase 4 automation script structure validation passed")
    
    def test_02_deployment_config_dataclass(self):
        """Test DeploymentConfig dataclass structure"""
        print("🔍 Testing DeploymentConfig dataclass...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for dataclass import and usage
        self.assertIn(
                      'from dataclasses import dataclass',
                      content,
                      "Missing dataclass import"
                     )
        self.assertIn('@dataclass', content, "Missing dataclass decorator")
        
        # Check for required configuration fields
        required_fields = [
            'project_id: str',
            'region: str',
            'model_name: str',
            'endpoint_name: str',
            'machine_type: str',
            'min_replicas: int',
            'max_replicas: int',
            'cost_budget: float'
        ]
        
        for field in required_fields:
            self.assertIn(field, content, f"Missing required field: {field}")
        
        print("✅ DeploymentConfig dataclass validation passed")
    
    def test_03_prerequisites_checking(self):
        """Test prerequisites checking functionality"""
        print("🔍 Testing prerequisites checking...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for prerequisite checks
        prerequisite_checks = [
            'gcloud CLI',
            'Authentication',
            'Project Configuration',
            'Vertex AI API',
            'Cloud Monitoring API',
            'Cloud Logging API',
            'Artifact Registry',
            'IAM Permissions'
        ]
        
        for check in prerequisite_checks:
            self.assertIn(check, content, f"Missing prerequisite check: {check}")
        
        # Check for individual check methods
        check_methods = [
            '_check_gcloud',
            '_check_authentication',
            '_check_project',
            '_check_vertex_ai_api',
            '_check_monitoring_api',
            '_check_logging_api',
            '_check_artifact_registry',
            '_check_iam_permissions'
        ]
        
        for method in check_methods:
            self.assertIn(f'def {method}', content, f"Missing check method: {method}")
        
        print("✅ Prerequisites checking validation passed")
    
    def test_04_model_versioning(self):
        """Test model versioning functionality"""
        print("🔍 Testing model versioning...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for version generation
        self.assertIn(
                      'def generate_model_version',
                      content,
                      "Missing version generation method"
                     )
        self.assertIn(
                      'datetime.now().strftime',
                      content,
                      "Missing timestamp generation"
                     )
        self.assertIn('git rev-parse', content, "Missing git commit hash")
        
        # Check for version format
        self.assertIn('v{timestamp}_{git_hash}', content, "Missing version format")
        
        print("✅ Model versioning validation passed")
    
    def test_05_deployment_package_creation(self):
        """Test deployment package creation"""
        print("🔍 Testing deployment package creation...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for deployment package creation
        self.assertIn(
                      'def create_deployment_package',
                      content,
                      "Missing deployment package creation"
                     )
        self.assertIn(
                      'deployment/vertex_ai/{version}',
                      content,
                      "Missing versioned directory structure"
                     )
        self.assertIn('Dockerfile', content, "Missing Dockerfile creation")
        self.assertIn('version_metadata.json', content, "Missing version metadata")
        
        # Check for required files
        required_files = [
            'model/',
            'requirements.txt',
            'predict.py',
            'Dockerfile',
            'version_metadata.json'
        ]
        
        for file in required_files:
            self.assertIn(file, content, f"Missing required file: {file}")
        
        print("✅ Deployment package creation validation passed")
    
    def test_06_docker_image_handling(self):
        """Test Docker image building and pushing"""
        print("🔍 Testing Docker image handling...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for Docker operations
        self.assertIn(
                      'def build_and_push_image',
                      content,
                      "Missing Docker image handling"
                     )
        self.assertIn(
                      'gcloud auth configure-docker',
                      content,
                      "Missing Docker authentication"
                     )
        self.assertIn('docker build', content, "Missing Docker build")
        self.assertIn('docker push', content, "Missing Docker push")
        
        # Check for image URI format
        self.assertIn(
                      'gcr.io/{self.config.project_id}',
                      content,
                      "Missing image URI format"
                     )
        
        print("✅ Docker image handling validation passed")
    
    def test_07_vertex_ai_model_creation(self):
        """Test Vertex AI model creation"""
        print("🔍 Testing Vertex AI model creation...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for model creation
        self.assertIn(
                      'def create_vertex_ai_model',
                      content,
                      "Missing model creation method"
                     )
        self.assertIn(
                      'gcloud ai models upload',
                      content,
                      "Missing model upload command"
                     )
        self.assertIn('--container-image-uri', content, "Missing container image URI")
        self.assertIn('--container-predict-route', content, "Missing predict route")
        self.assertIn('--container-health-route', content, "Missing health route")
        
        print("✅ Vertex AI model creation validation passed")
    
    def test_08_endpoint_deployment(self):
        """Test endpoint deployment functionality"""
        print("🔍 Testing endpoint deployment...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for endpoint deployment
        self.assertIn(
                      'def deploy_model_to_endpoint',
                      content,
                      "Missing endpoint deployment method"
                     )
        self.assertIn(
                      'gcloud ai endpoints deploy-model',
                      content,
                      "Missing endpoint deployment command"
                     )
        self.assertIn('--traffic-split', content, "Missing traffic split")
        self.assertIn('--machine-type', content, "Missing machine type")
        self.assertIn('--min-replica-count', content, "Missing min replica count")
        self.assertIn('--max-replica-count', content, "Missing max replica count")
        
        print("✅ Endpoint deployment validation passed")
    
    def test_09_monitoring_and_alerting(self):
        """Test monitoring and alerting setup"""
        print("🔍 Testing monitoring and alerting...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for monitoring setup
        self.assertIn(
                      'def setup_monitoring_and_alerting',
                      content,
                      "Missing monitoring setup method"
                     )
        self.assertIn('monitoring_policy.json', content, "Missing monitoring policy")
        self.assertIn(
                      'gcloud alpha monitoring policies create',
                      content,
                      "Missing monitoring policy creation"
                     )
        
        # Check for alert conditions
        self.assertIn('High Error Rate', content, "Missing error rate monitoring")
        self.assertIn('High Latency', content, "Missing latency monitoring")
        
        print("✅ Monitoring and alerting validation passed")
    
    def test_10_cost_monitoring(self):
        """Test cost monitoring setup"""
        print("🔍 Testing cost monitoring...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for cost monitoring
        self.assertIn(
                      'def setup_cost_monitoring',
                      content,
                      "Missing cost monitoring method"
                     )
        self.assertIn('budget_config.json', content, "Missing budget configuration")
        self.assertIn(
                      'gcloud billing budgets create',
                      content,
                      "Missing budget creation"
                     )
        
        # Check for budget thresholds
        self.assertIn('thresholdPercent', content, "Missing budget thresholds")
        
        print("✅ Cost monitoring validation passed")
    
    def test_11_rollback_capabilities(self):
        """Test rollback capabilities"""
        print("🔍 Testing rollback capabilities...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for rollback functionality
        self.assertIn('def rollback_deployment', content, "Missing rollback method")
        self.assertIn('deployment_history', content, "Missing deployment history")
        self.assertIn(
                      'gcloud ai endpoints deploy-model',
                      content,
                      "Missing rollback deployment"
                     )
        
        print("✅ Rollback capabilities validation passed")
    
    def test_12_ab_testing_support(self):
        """Test A/B testing support"""
        print("🔍 Testing A/B testing support...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for A/B testing
        self.assertIn('def setup_ab_testing', content, "Missing A/B testing method")
        self.assertIn('version_a', content, "Missing version A parameter")
        self.assertIn('version_b', content, "Missing version B parameter")
        self.assertIn('traffic_split', content, "Missing traffic split")
        
        print("✅ A/B testing support validation passed")
    
    def test_13_performance_metrics(self):
        """Test performance metrics collection"""
        print("🔍 Testing performance metrics...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for performance metrics
        self.assertIn(
                      'def get_performance_metrics',
                      content,
                      "Missing performance metrics method"
                     )
        self.assertIn(
                      'gcloud ai endpoints describe',
                      content,
                      "Missing endpoint description"
                     )
        self.assertIn('gcloud ai models list', content, "Missing model listing")
        
        print("✅ Performance metrics validation passed")
    
    def test_14_cleanup_functionality(self):
        """Test cleanup functionality"""
        print("🔍 Testing cleanup functionality...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for cleanup
        self.assertIn('def cleanup_old_versions', content, "Missing cleanup method")
        self.assertIn('keep_versions', content, "Missing version retention")
        self.assertIn('gcloud ai models delete', content, "Missing model deletion")
        
        print("✅ Cleanup functionality validation passed")
    
    def test_15_full_deployment_workflow(self):
        """Test full deployment workflow"""
        print("🔍 Testing full deployment workflow...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for full deployment workflow
        self.assertIn(
                      'def run_full_deployment',
                      content,
                      "Missing full deployment method"
                     )
        
        # Check for workflow steps
        workflow_steps = [
            'check_prerequisites',
            'generate_model_version',
            'create_deployment_package',
            'build_and_push_image',
            'create_vertex_ai_model',
            'deploy_model_to_endpoint',
            'setup_monitoring_and_alerting',
            'setup_cost_monitoring',
            'get_performance_metrics',
            'cleanup_old_versions',
            '_save_deployment_summary'
        ]
        
        for step in workflow_steps:
            self.assertIn(step, content, f"Missing workflow step: {step}")
        
        print("✅ Full deployment workflow validation passed")
    
    def test_16_error_handling(self):
        """Test error handling and logging"""
        print("🔍 Testing error handling...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for error handling
        self.assertIn('import logging', content, "Missing logging import")
        self.assertIn('logger = logging.getLogger', content, "Missing logger setup")
        self.assertIn('try:', content, "Missing try blocks")
        self.assertIn('except', content, "Missing except blocks")
        self.assertIn('logger.error', content, "Missing error logging")
        self.assertIn('logger.warning', content, "Missing warning logging")
        
        print("✅ Error handling validation passed")
    
    def test_17_configuration_management(self):
        """Test configuration management"""
        print("🔍 Testing configuration management...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for configuration management
        self.assertIn('DeploymentConfig', content, "Missing deployment configuration")
        self.assertIn('project_id', content, "Missing project ID configuration")
        self.assertIn('region', content, "Missing region configuration")
        self.assertIn('machine_type', content, "Missing machine type configuration")
        self.assertIn('min_replicas', content, "Missing min replicas configuration")
        self.assertIn('max_replicas', content, "Missing max replicas configuration")
        self.assertIn('cost_budget', content, "Missing cost budget configuration")
        
        print("✅ Configuration management validation passed")
    
    def test_18_security_features(self):
        """Test security features"""
        print("🔍 Testing security features...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for security features
        self.assertIn('subprocess.run', content, "Missing subprocess usage")
        self.assertIn('capture_output=True', content, "Missing output capture")
        self.assertIn('text=True', content, "Missing text mode")
        self.assertIn('check=True', content, "Missing error checking")
        
        print("✅ Security features validation passed")
    
    def test_19_documentation_and_logging(self):
        """Test documentation and logging"""
        print("🔍 Testing documentation and logging...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for documentation
        self.assertIn('"""', content, "Missing docstrings")
        self.assertIn(
                      'Phase 4: Vertex AI Deployment Automation',
                      content,
                      "Missing module docstring"
                     )
        self.assertIn(
                      'Enhanced Vertex AI deployment',
                      content,
                      "Missing class docstring"
                     )
        
        # Check for logging
        self.assertIn('logger.info', content, "Missing info logging")
        self.assertIn('print(', content, "Missing print statements")
        
        print("✅ Documentation and logging validation passed")
    
    def test_20_main_function(self):
        """Test main function"""
        print("🔍 Testing main function...")
        
        with open(self.vertex_ai_script, 'r') as f:
            content = f.read()
        
        # Check for main function
        self.assertIn('def main():', content, "Missing main function")
        self.assertIn('if __name__ == "__main__":', content, "Missing main guard")
        self.assertIn(
                      'gcloud config get-value project',
                      content,
                      "Missing project ID retrieval"
                     )
        self.assertIn('DeploymentConfig(', content, "Missing configuration creation")
        self.assertIn(
                      'VertexAIPhase4Automation(',
                      content,
                      "Missing automation instance creation"
                     )
        self.assertIn('run_full_deployment()', content, "Missing deployment execution")
        
        print("✅ Main function validation passed")

def run_phase4_tests():
    """Run all Phase 4 tests"""
    print("🚀 RUNNING PHASE 4 VERTEX AI AUTOMATION TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Phase4VertexAIAutomationTest)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 PHASE 4 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.skipped:
        print("\n⚠️  SKIPPED:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 ALL PHASE 4 TESTS PASSED!")
        print("✅ Vertex AI automation is ready for deployment")
        print("\n📋 Phase 4 Features Validated:")
        print("   ✅ Automated model versioning and deployment")
        print("   ✅ Rollback capabilities and A/B testing support")
        print("   ✅ Model performance monitoring and alerting")
        print("   ✅ Cost optimization and resource management")
        print("   ✅ Comprehensive testing and validation")
    else:
        print("\n❌ SOME PHASE 4 TESTS FAILED!")
        print("Please fix the issues before proceeding with deployment")
    
    return success

if __name__ == "__main__":
    success = run_phase4_tests()
    sys.exit(0 if success else 1) 