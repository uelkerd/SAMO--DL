import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HealthMonitor:
    """Health monitoring system for API endpoints and system resources."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_health_check = None
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        try:
            # System resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process information
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Uptime calculation
            uptime_seconds = time.time() - self.start_time
            uptime_hours = uptime_seconds / 3600
            
            health_data = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_hours": round(uptime_hours, 2),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / 1024**3, 2),
                    "disk_percent": disk.percent,
                    "disk_free_gb": round(disk.free / 1024**3, 2)
                },
                "process": {
                    "memory_mb": round(process_memory, 2),
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": round(self.error_count / max(self.request_count, 1) * 100, 2)
                },
                "last_health_check": self.last_health_check
            }
            
            # Determine overall health status
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                health_data["status"] = "warning"
            if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                health_data["status"] = "critical"
            if self.error_count > 0 and self.error_count / max(self.request_count, 1) > 0.1:
                health_data["status"] = "degraded"
                
            self.last_health_check = health_data["timestamp"]
            return health_data
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def record_request(self, success: bool = True):
        """Record a request for health monitoring."""
        self.request_count += 1
        if not success:
            self.error_count += 1
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a simplified health summary for quick checks."""
        health = self.get_system_health()
        return {
            "status": health["status"],
            "uptime_hours": health["uptime_hours"],
            "request_count": health["process"]["request_count"],
            "error_rate": health["process"]["error_rate"]
        }

# Global health monitor instance
health_monitor = HealthMonitor()
