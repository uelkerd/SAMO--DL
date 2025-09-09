"""Documentation Blueprint for API Documentation Serving.

This module provides Flask blueprint routes for serving OpenAPI specifications
and Swagger UI documentation. Includes security features for path validation
and content type handling to prevent unauthorized access to specification files.

Key Features:
- Serves OpenAPI YAML specification with path validation
- Renders Swagger UI with configurable spec URL
- Implements Content Security Policy (CSP) nonce generation
- Secure file path resolution and access control
"""

from __future__ import annotations
import os
from pathlib import Path
from flask import Blueprint, Response, jsonify, render_template, g


docs_bp = Blueprint('docs', __name__, template_folder='templates')


@docs_bp.route('/openapi.yaml', methods=['GET'])
def serve_openapi_spec() -> Response:
    """Serve OpenAPI spec for Swagger UI with safe path validation."""
    # Restrict spec path to a safe directory
    allowed_dir = Path(os.environ.get('OPENAPI_ALLOWED_DIR', '/app')).resolve()
    spec_path = os.environ.get('OPENAPI_SPEC_PATH', '/app/openapi.yaml')
    abs_spec_path = Path(spec_path).resolve()

    try:
        # Validate that the spec path is within the allowed directory
        if os.path.commonpath([abs_spec_path, allowed_dir]) != allowed_dir:
            return jsonify({'error': 'Invalid OpenAPI spec path'}), 400

        with open(abs_spec_path, encoding='utf-8') as f:
            content = f.read()
        # Use a standard YAML mimetype
        return Response(content, mimetype='application/x-yaml')
    except Exception:
        # Avoid leaking exact path in error; log on server side only if needed
        return jsonify({'error': 'OpenAPI spec not found'}), 404


@docs_bp.route('/docs', methods=['GET'], strict_slashes=False)
def swagger_ui() -> str:
    """Render Swagger UI that loads the OpenAPI spec from /openapi.yaml."""
    # Allow overriding the spec URL (e.g., behind a proxy) but default to local
    spec_url = os.environ.get('OPENAPI_SPEC_URL', '/openapi.yaml')
    # Generate per-request nonce for CSP and pass to template
    import secrets
    nonce = secrets.token_urlsafe(16)
    g.csp_nonce = nonce
    return render_template('docs.html', spec_url=spec_url, csp_nonce=nonce)
