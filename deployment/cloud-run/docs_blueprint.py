from __future__ import annotations

import os
from pathlib import Path
from flask import Blueprint, Response, jsonify, render_template, g


docs_bp = Blueprint('docs', __name__, template_folder='templates')


@docs_bp.route('/openapi.yaml', methods=['GET'])
def serve_openapi_spec():
    """Serve OpenAPI spec for Swagger UI with safe path validation."""
    # Restrict spec path to a safe directory
    allowed_dir = Path(os.environ.get('OPENAPI_ALLOWED_DIR', '/app')).resolve()
    spec_path = os.environ.get('OPENAPI_SPEC_PATH', '/app/openapi.yaml')
    abs_spec_path = Path(spec_path).resolve()

    try:
        # Validate that the spec path is within the allowed directory
        # Use robust containment check compatible with older Python versions
        try:
            is_contained = abs_spec_path.is_relative_to(allowed_dir)
        except AttributeError:
            # Fallback for Python < 3.9
            try:
                os.path.commonpath([str(allowed_dir), str(abs_spec_path)]) == str(allowed_dir)
                is_contained = True
            except ValueError:
                is_contained = False

        if abs_spec_path.parent != allowed_dir and not is_contained:
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
    # Generate per-request nonce for CSP and pass to template
    import secrets
    nonce = secrets.token_urlsafe(16)
    g.csp_nonce = nonce
    return render_template('docs.html', spec_url=spec_url, csp_nonce=nonce)
