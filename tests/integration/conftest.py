import os
import socket
import threading
import time
import importlib.util
import pytest


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def secure_api_server_url():
    os.environ.setdefault('ADMIN_API_KEY', 'test-key-123')
    os.environ.setdefault('OPENAPI_SPEC_PATH', os.path.abspath('deployment/cloud-run/openapi.yaml'))
    os.environ.setdefault('OPENAPI_ALLOWED_DIR', os.path.abspath('deployment/cloud-run'))
    port = _find_free_port()
    os.environ['PORT'] = str(port)

    spec = importlib.util.spec_from_file_location('secure_api_server', 'deployment/cloud-run/secure_api_server.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    app = mod.app

    def run():
        app.run(host='127.0.0.1', port=port, debug=False)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    time.sleep(1.5)
    base = f"http://127.0.0.1:{port}"
    # Export for tests that rely on env
    os.environ['SECURE_API_BASE'] = base
    yield base
    # teardown: nothing to do, daemon thread exits with process