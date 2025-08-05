#!/usr/bin/env python3

import os
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    who = os.environ.get('WHO', 'World')
    return f'Hello {who}! This is SAMO Emotion Detection API test.\n'

@app.route('/health')
def health():
    """Health check endpoint."""
    return 'OK\n'

if __name__ == '__main__':
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.environ.get('PORT', 8080))
    
    # Start the server
    app.run(host='0.0.0.0', port=port, debug=False) 