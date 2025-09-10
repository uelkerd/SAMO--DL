from flask import Flask, jsonify
from flask_cors import CORS
from auth import require_api_key
from rate_limiter import rate_limit

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/api/protected', methods=['POST'])
@require_api_key
@rate_limit(max_requests=10, window_minutes=1)
def protected():
    return jsonify({'message': 'Protected endpoint'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
