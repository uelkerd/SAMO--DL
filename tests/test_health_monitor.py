import unittest
from unittest.mock import patch, MagicMock
from src.health_monitor import HealthMonitor

class TestHealthMonitor(unittest.TestCase):
    """Test cases for health monitoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.health_monitor = HealthMonitor()
        
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        self.assertIsNotNone(self.health_monitor.start_time)
        self.assertEqual(self.health_monitor.request_count, 0)
        self.assertEqual(self.health_monitor.error_count, 0)
        self.assertIsNone(self.health_monitor.last_health_check)
    
    def test_get_system_health(self):
        """Test getting system health metrics."""
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.Process') as mock_process:
            
            # Mock system metrics
            mock_cpu.return_value = 45.2
            mock_memory.return_value = MagicMock(percent=67.8, available=8589934592)
            mock_disk.return_value = MagicMock(percent=23.1, free=107374182400)
            mock_process.return_value.memory_info.return_value.rss = 134217728  # 128MB
            
            health_data = self.health_monitor.get_system_health()
            
            self.assertIn('cpu_percent', health_data)
            self.assertIn('memory_percent', health_data)
            self.assertIn('disk_percent', health_data)
            self.assertIn('process_memory_mb', health_data)
            self.assertIn('uptime', health_data)
    
    def test_get_health_summary(self):
        """Test getting health summary."""
        with patch.object(self.health_monitor, 'get_system_health') as mock_health:
            mock_health.return_value = {
                'cpu_percent': 45.2,
                'memory_percent': 67.8,
                'disk_percent': 23.1,
                'uptime': 3600
            }
            
            summary = self.health_monitor.get_health_summary()
            
            self.assertIn('status', summary)
            self.assertIn('uptime', summary)
            self.assertIn('cpu_usage', summary)
            self.assertIn('memory_usage', summary)
            self.assertIn('disk_usage', summary)
    
    def test_health_summary_healthy_status(self):
        """Test health summary with healthy status."""
        with patch.object(self.health_monitor, 'get_system_health') as mock_health:
            mock_health.return_value = {
                'cpu_percent': 45.2,
                'memory_percent': 67.8,
                'disk_percent': 23.1,
                'uptime': 3600
            }
            
            summary = self.health_monitor.get_health_summary()
            self.assertEqual(summary['status'], 'healthy')
    
    def test_health_summary_warning_status(self):
        """Test health summary with warning status."""
        with patch.object(self.health_monitor, 'get_system_health') as mock_health:
            mock_health.return_value = {
                'cpu_percent': 85.2,
                'memory_percent': 90.8,
                'disk_percent': 23.1,
                'uptime': 3600
            }
            
            summary = self.health_monitor.get_health_summary()
            self.assertEqual(summary['status'], 'warning')
    
    def test_health_summary_critical_status(self):
        """Test health summary with critical status."""
        with patch.object(self.health_monitor, 'get_system_health') as mock_health:
            mock_health.return_value = {
                'cpu_percent': 95.2,
                'memory_percent': 98.8,
                'disk_percent': 95.1,
                'uptime': 3600
            }
            
            summary = self.health_monitor.get_health_summary()
            self.assertEqual(summary['status'], 'critical')
    
    def test_increment_request_count(self):
        """Test incrementing request count."""
        initial_count = self.health_monitor.request_count
        self.health_monitor.increment_request_count()
        self.assertEqual(self.health_monitor.request_count, initial_count + 1)
    
    def test_increment_error_count(self):
        """Test incrementing error count."""
        initial_count = self.health_monitor.error_count
        self.health_monitor.increment_error_count()
        self.assertEqual(self.health_monitor.error_count, initial_count + 1)
    
    def test_update_last_health_check(self):
        """Test updating last health check timestamp."""
        self.health_monitor.update_last_health_check()
        self.assertIsNotNone(self.health_monitor.last_health_check)

if __name__ == '__main__':
    unittest.main()
