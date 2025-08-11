from __future__ import annotations

import os
from flask import Blueprint, Response, jsonify, render_template


docs_bp = Blueprint('docs', __name__, template_folder='templates')


@docs_bp.route('/openapi.yaml', methods=['GET'])
def serve_openapi_spec():
    """Serve OpenAPI spec for Swagger UI with safe path validation."""
    # Restrict spec path to a safe directory
    allowed_dir = os.path.abspath(os.environ.get('OPENAPI_ALLOWED_DIR', '/app'))
    spec_path = os.environ.get('OPENAPI_SPEC_PATH', '/app/openapi.yaml')
    abs_spec_path = os.path.abspath(spec_path)

    try:
        # Validate that the spec path is within the allowed directory
        if os.path.commonpath([abs_spec_path, allowed_dir]) != allowed_dir:
            return jsonify({'error': 'Invalid OpenAPI spec path'}), 400

        with open(abs_spec_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Use a standard YAML mimetype
        return Response(content, mimetype='application/x-yaml')
    except Exception as e:
        # Avoid leaking exact path in error; log on server side only if needed
        return jsonify({'error': 'OpenAPI spec not found'}), 404


@docs_bp.route('/docs', methods=['GET'], strict_slashes=False)
def swagger_ui():
    """Render Swagger UI that loads the OpenAPI spec from /openapi.yaml."""
    # Allow overriding the spec URL (e.g., behind a proxy) but default to local
    spec_url = os.environ.get('OPENAPI_SPEC_URL', '/openapi.yaml')
    return render_template('docs.html', spec_url=spec_url)