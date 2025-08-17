"""
Environment Configuration Management - Phase 3 Cloud Run Optimization
Provides environment-specific settings for development, staging, and production
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class CloudRunConfig:
    """Cloud Run specific configuration"""
    # Resource allocation
    memory_limit_mb: int = 2048
    cpu_limit: int = 2
    max_instances: int = 10
    min_instances: int = 1
    concurrency: int = 80
    timeout_seconds: int = 300

    # Auto-scaling
    target_cpu_utilization: float = 0.7
    target_memory_utilization: float = 0.8
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300

    # Health checks
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    health_check_retries: int = 3

    # Graceful shutdown
    graceful_shutdown_timeout_seconds: int = 30

    # Monitoring
    enable_monitoring: bool = True
    enable_metrics: bool = True
    log_level: str = "info"

    # Rate limiting
    max_requests_per_minute: int = 1000
    rate_limit_window_seconds: int = 60

    # Security
    enable_cors: bool = True
    cors_origins: Optional[List[str]] = None
    enable_rate_limiting: bool = True
    enable_input_sanitization: bool = True

class EnvironmentConfig:
    """Environment-specific configuration management"""

    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config = self._load_environment_config()

    def _load_environment_config(self) -> CloudRunConfig:
        """Load configuration based on environment"""
        if self.environment == 'production':
            return CloudRunConfig(
                memory_limit_mb=int(os.getenv('MEMORY_LIMIT_MB', '2048') or '2048'),
                cpu_limit=int(os.getenv('CPU_LIMIT', '2') or '2'),
                max_instances=int(os.getenv('MAX_INSTANCES', '10') or '10'),
                min_instances=int(os.getenv('MIN_INSTANCES', '1') or '1'),
                concurrency=int(os.getenv('CONCURRENCY', '80') or '80'),
                timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', '300') or '300'),
                target_cpu_utilization=float(
                                             os.getenv('TARGET_CPU_UTILIZATION',
                                             '0.7') or '0.7'),
                                             
                target_memory_utilization=float(
                                                os.getenv('TARGET_MEMORY_UTILIZATION',
                                                '0.8') or '0.8'),
                                                
                health_check_interval_seconds=int(
                                                  os.getenv('HEALTH_CHECK_INTERVAL',
                                                  '30') or '30'),
                                                  
                graceful_shutdown_timeout_seconds=int(
                                                      os.getenv('GRACEFUL_SHUTDOWN_TIMEOUT',
                                                      '30') or '30'),
                                                      
                enable_monitoring=os.getenv(
                                            'ENABLE_MONITORING',
                                            'true').lower() == 'true',
                                            
                enable_metrics=os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
                log_level=os.getenv('LOG_LEVEL', 'info'),
                max_requests_per_minute=int(
                                            os.getenv('MAX_REQUESTS_PER_MINUTE',
                                            '1000') or '1000'),
                                            
                enable_cors=True,
                cors_origins=os.getenv('CORS_ORIGINS', '*').split(','),
                enable_rate_limiting=True,
                enable_input_sanitization=True
            )

        if self.environment == 'staging':
            return CloudRunConfig(
                memory_limit_mb=1024,
                cpu_limit=1,
                max_instances=5,
                min_instances=0,
                concurrency=40,
                timeout_seconds=180,
                target_cpu_utilization=0.6,
                target_memory_utilization=0.7,
                health_check_interval_seconds=60,
                graceful_shutdown_timeout_seconds=15,
                enable_monitoring=True,
                enable_metrics=True,
                log_level='debug',
                max_requests_per_minute=500,
                enable_cors=True,
                cors_origins=['*'],
                enable_rate_limiting=True,
                enable_input_sanitization=True
            )
        return CloudRunConfig(
            memory_limit_mb=512,
            cpu_limit=1,
            max_instances=2,
            min_instances=0,
            concurrency=20,
            timeout_seconds=120,
            target_cpu_utilization=0.5,
            target_memory_utilization=0.6,
            health_check_interval_seconds=120,
            graceful_shutdown_timeout_seconds=10,
            enable_monitoring=False,
            enable_metrics=False,
            log_level='debug',
            max_requests_per_minute=100,
            enable_cors=True,
            cors_origins=['*'],
            enable_rate_limiting=False,
            enable_input_sanitization=False
        )

    def get_gunicorn_config(self) -> Dict[str, Any]:
        """Get Gunicorn configuration for Cloud Run"""
        return {
            'bind': f':{os.getenv("PORT", "8080")}',
            'workers': 1,  # Cloud Run best practice
            'threads': 8,
            'timeout': 0,  # Cloud Run handles timeouts
            'keepalive': 5,
            'max_requests': 1000,
            'max_requests_jitter': 100,
            'access_logfile': '-',
            'error_logfile': '-',
            'loglevel': self.config.log_level,
            'preload_app': True,
            'worker_class': 'sync',
            'worker_connections': self.config.concurrency
        }

    def get_health_check_config(self) -> Dict[str, Any]:
        """Get health check configuration"""
        return {
            'interval_seconds': self.config.health_check_interval_seconds,
            'timeout_seconds': self.config.health_check_timeout_seconds,
            'retries': self.config.health_check_retries,
            'graceful_shutdown_timeout': self.config.graceful_shutdown_timeout_seconds
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            'enabled': self.config.enable_monitoring,
            'metrics_enabled': self.config.enable_metrics,
            'log_level': self.config.log_level,
            'target_cpu_utilization': self.config.target_cpu_utilization,
            'target_memory_utilization': self.config.target_memory_utilization
        }

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'enable_cors': self.config.enable_cors,
            'cors_origins': self.config.cors_origins,
            'enable_rate_limiting': self.config.enable_rate_limiting,
            'enable_input_sanitization': self.config.enable_input_sanitization,
            'max_requests_per_minute': self.config.max_requests_per_minute
        }

    def validate_config(self) -> None:
        """Validate configuration settings"""
        # Validate resource limits
        if not 512 <= self.config.memory_limit_mb <= 8192:
            raise AssertionError("Memory limit must be between 512MB and 8GB")
        if not 1 <= self.config.cpu_limit <= 8:
            raise AssertionError("CPU limit must be between 1 and 8")
        if not 1 <= self.config.max_instances <= 100:
            raise AssertionError("Max instances must be between 1 and 100")
        if not 0 <= self.config.min_instances <= self.config.max_instances:
            raise AssertionError("Min instances cannot exceed max instances")

        # Validate timeouts
        if not 10 <= self.config.timeout_seconds <= 900:
            raise AssertionError("Timeout must be between 10 and 900 seconds")
        if not 5 <= self.config.health_check_interval_seconds <= 300:
            raise AssertionError(
                                 "Health check interval must be between 5 and 300 seconds"
                                )

        # Validate utilization targets
        if not 0.1 <= self.config.target_cpu_utilization <= 0.9:
            raise AssertionError("CPU utilization target must be between 0.1 and 0.9")
        if not 0.1 <= self.config.target_memory_utilization <= 0.9:
            raise AssertionError(
                                 "Memory utilization target must be between 0.1 and 0.9"
                                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': self.environment,
            'cloud_run': {
                'memory_limit_mb': self.config.memory_limit_mb,
                'cpu_limit': self.config.cpu_limit,
                'max_instances': self.config.max_instances,
                'min_instances': self.config.min_instances,
                'concurrency': self.config.concurrency,
                'timeout_seconds': self.config.timeout_seconds,
                'target_cpu_utilization': self.config.target_cpu_utilization,
                'target_memory_utilization': self.config.target_memory_utilization,
'health_check_interval_seconds': self.config.health_check_interval_seconds,
'graceful_shutdown_timeout_seconds': self.config.graceful_shutdown_timeout_seconds
            },
            'monitoring': self.get_monitoring_config(),
            'security': self.get_security_config()
        }

# Global configuration instance
config = EnvironmentConfig()

def get_config() -> EnvironmentConfig:
    """Get the global configuration instance"""
    return config 
