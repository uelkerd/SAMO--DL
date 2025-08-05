import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running'
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'SAMO Test API',
        'version': '1.0.0',
        'status': 'running'
    })

if __name__ == '__main__':
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting simple test server on port {port}")
    
    # Start the server
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    ) 