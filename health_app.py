#!/usr/bin/env python3
"""Simple health check Flask application for minimal Docker testing."""

import os
from flask import Flask, jsonify, Response

app: Flask = Flask(__name__)


@app.route("/health")
def health() -> Response:
    """Health check endpoint.

    Returns:
        Response: JSON health status response.
    """
    res = jsonify({"status": "healthy"})
    res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return res


@app.route("/")
def root() -> Response:
    """Root endpoint.

    Returns:
        Response: Basic application information.
    """
    res = jsonify({
        "message": "SAMO Health Check Service",
        "status": "running"
    })
    res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return res


if __name__ == "__main__":
    # For production, use HOST environment variable. For security, default to localhost
    # Override with HOST=0.0.0.0 only when explicitly needed
    # (e.g., Docker/Cloud Run)
    host = os.getenv("HOST", "127.0.0.1")  # noqa: S104 - intentional for runtime
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
