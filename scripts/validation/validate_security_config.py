#!/usr/bin/env python3
"""
Security Configuration Validator

This script validates the security configuration file to ensure all required
settings are present and valid according to the security schema.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class SecurityConfigValidator:
    """Validator for security configuration files."""
    
    def __init__(self, config_path: str = "configs/security.yaml"):
        self.config_path = Path(config_path)
        self.errors = []
        self.warnings = []
        
    def validate(self) -> bool:
        """Validate the security configuration file."""
        print("🔍 Validating security configuration...")
        
        # Check if file exists
        if not self.config_path.exists():
            self.errors.append(f"Security configuration file not found: {self.config_path}")
            return False
        
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in security configuration: {e}")
            return False
        
        # Validate required sections
        self._validate_required_sections(config)
        
        # Validate API security settings
        self._validate_api_security(config.get('api', {}))
        
        # Validate security headers
        self._validate_security_headers(config.get('security_headers', {}))
        
        # Validate logging configuration
        self._validate_logging(config.get('logging', {}))
        
        # Validate environment settings
        self._validate_environment(config.get('environment', {}))
        
        # Validate dependency security
        self._validate_dependencies(config.get('dependencies', {}))
        
        # Validate model security
        self._validate_model_security(config.get('model', {}))
        
        # Validate database security
        self._validate_database_security(config.get('database', {}))
        
        # Validate deployment security
        self._validate_deployment_security(config.get('deployment', {}))
        
        return len(self.errors) == 0
    
    def _validate_required_sections(self, config: Dict[str, Any]) -> None:
        """Validate that all required sections are present."""
        required_sections = [
            'api', 'security_headers', 'logging', 'environment',
            'dependencies', 'model', 'database', 'deployment'
        ]
        
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
    
    def _validate_api_security(self, api_config: Dict[str, Any]) -> None:
        """Validate API security configuration."""
        if not api_config:
            self.errors.append("API configuration is empty")
            return
        
        # Check rate limiting
        rate_limiting = api_config.get('rate_limiting', {})
        if not rate_limiting.get('enabled', False):
            self.warnings.append("Rate limiting is disabled - security risk")
        
        # Check CORS
        cors = api_config.get('cors', {})
        if not cors.get('enabled', False):
            self.warnings.append("CORS is disabled - may cause issues")
        
        # Check authentication
        auth = api_config.get('authentication', {})
        if not auth.get('enabled', False):
            self.errors.append("Authentication is disabled - security risk")
        
        # Check input validation
        input_validation = api_config.get('input_validation', {})
        if not input_validation:
            self.errors.append("Input validation configuration is missing")
    
    def _validate_security_headers(self, headers_config: Dict[str, Any]) -> None:
        """Validate security headers configuration."""
        if not headers_config.get('enabled', False):
            self.warnings.append("Security headers are disabled")
            return
        
        headers = headers_config.get('headers', {})
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection'
        ]
        
        for header in required_headers:
            if header not in headers:
                self.warnings.append(f"Missing recommended security header: {header}")
    
    def _validate_logging(self, logging_config: Dict[str, Any]) -> None:
        """Validate logging configuration."""
        if not logging_config:
            self.errors.append("Logging configuration is missing")
            return
        
        # Check security events logging
        security_events = logging_config.get('security_events', {})
        if not security_events.get('enabled', False):
            self.warnings.append("Security events logging is disabled")
        
        # Check request logging
        requests = logging_config.get('requests', {})
        if not requests.get('enabled', False):
            self.warnings.append("Request logging is disabled")
        
        # Check error logging
        errors = logging_config.get('errors', {})
        if not errors.get('enabled', False):
            self.warnings.append("Error logging is disabled")
    
    def _validate_environment(self, env_config: Dict[str, Any]) -> None:
        """Validate environment configuration."""
        if not env_config:
            self.errors.append("Environment configuration is missing")
            return
        
        # Check required environment variables
        required_vars = env_config.get('required_vars', [])
        if not required_vars:
            self.warnings.append("No required environment variables specified")
        
        # Check sensitive variables
        sensitive_vars = env_config.get('sensitive_vars', [])
        if not sensitive_vars:
            self.warnings.append("No sensitive variables specified for masking")
        
        # Check environment-specific settings
        for env in ['production', 'development', 'testing']:
            env_settings = env_config.get(env, {})
            if not env_settings:
                self.warnings.append(f"No settings specified for {env} environment")
    
    def _validate_dependencies(self, deps_config: Dict[str, Any]) -> None:
        """Validate dependency security configuration."""
        if not deps_config:
            self.errors.append("Dependency security configuration is missing")
            return
        
        scanning = deps_config.get('scanning', {})
        if not scanning.get('enabled', False):
            self.warnings.append("Dependency security scanning is disabled")
        
        tools = scanning.get('tools', [])
        if not tools:
            self.warnings.append("No security scanning tools specified")
    
    def _validate_model_security(self, model_config: Dict[str, Any]) -> None:
        """Validate model security configuration."""
        if not model_config:
            self.errors.append("Model security configuration is missing")
            return
        
        loading = model_config.get('loading', {})
        if not loading.get('validate_model_files', False):
            self.warnings.append("Model file validation is disabled")
        
        inference = model_config.get('inference', {})
        if not inference:
            self.warnings.append("Model inference security settings are missing")
    
    def _validate_database_security(self, db_config: Dict[str, Any]) -> None:
        """Validate database security configuration."""
        if not db_config:
            self.errors.append("Database security configuration is missing")
            return
        
        connection = db_config.get('connection', {})
        if not connection.get('use_ssl', False):
            self.errors.append("Database SSL is disabled - security risk")
        
        data_protection = db_config.get('data_protection', {})
        if not data_protection.get('encrypt_sensitive_data', False):
            self.warnings.append("Sensitive data encryption is disabled")
    
    def _validate_deployment_security(self, deploy_config: Dict[str, Any]) -> None:
        """Validate deployment security configuration."""
        if not deploy_config:
            self.errors.append("Deployment security configuration is missing")
            return
        
        container = deploy_config.get('container', {})
        if not container.get('run_as_non_root', False):
            self.errors.append("Container not configured to run as non-root - security risk")
        
        network = deploy_config.get('network', {})
        if not network.get('use_https', False):
            self.errors.append("HTTPS is disabled - security risk")
    
    def print_results(self) -> None:
        """Print validation results."""
        print("\n📊 Security Configuration Validation Results")
        print("=" * 50)
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ Security configuration is valid!")
        elif not self.errors:
            print(f"\n⚠️  Configuration has {len(self.warnings)} warnings but no errors")
        else:
            print(f"\n❌ Configuration has {len(self.errors)} errors that must be fixed")

def main():
    """Main function to run security configuration validation."""
    validator = SecurityConfigValidator()
    
    if validator.validate():
        validator.print_results()
        if validator.errors:
            raise ValueError("Security validation errors found")
        else:
            print("\n✅ Security configuration validation passed!")
    else:
        validator.print_results()
        raise ValueError("Security validation failed")

if __name__ == "__main__":
    main()
