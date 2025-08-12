import json
import os
import pytest
import requests

pytestmark = pytest.mark.integration


def _h():
    return {"X-API-Key": os.environ.get('ADMIN_API_KEY', 'test-key-123'), "Content-Type": "application/json"}


def test_rate_limit_triggered(secure_api_server_url_rl):
    base = secure_api_server_url_rl
    url = f"{base}/api/summarize"
    body = {"text": "This text should be summarized."}
    r1 = requests.post(url, headers=_h(), data=json.dumps(body))
    r2 = requests.post(url, headers=_h(), data=json.dumps(body))
    r3 = requests.post(url, headers=_h(), data=json.dumps(body))
    # With limit=2/min we expect the 3rd to be 429
    assert r1.status_code in (200, 503)
    assert r2.status_code in (200, 503)
    assert r3.status_code == 429


def test_monitoring_performance_schema(secure_api_server_url):
    base = secure_api_server_url
    url = f"{base}/api/monitoring/performance"
    r = requests.get(url, headers={"X-API-Key": os.environ.get('ADMIN_API_KEY', 'test-key-123')})
    assert r.status_code == 200
    data = r.json()
    assert 'timestamp' in data
    assert 'system' in data
    sys = data['system']
    assert 'cpu_percent' in sys
    assert 'memory_percent' in sys
    assert 'disk_percent' in sys


def test_monitoring_health_detailed_schema(secure_api_server_url):
    base = secure_api_server_url
    url = f"{base}/api/monitoring/health/detailed"
    r = requests.get(url, headers={"X-API-Key": os.environ.get('ADMIN_API_KEY', 'test-key-123')})
    assert r.status_code == 200
    data = r.json()
    assert 'status' in data
    assert 'issues' in data
    assert isinstance(data['issues'], list)
    assert 'models' in data
    assert 'system' in data