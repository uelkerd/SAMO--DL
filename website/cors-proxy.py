#!/usr/bin/env python3
"""
CORS Proxy for SAMO Emotion API
This script creates a local proxy to bypass CORS restrictions
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import json
import sys

class CORSProxyHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_POST(self):
        if self.path == '/emotion':
            try:
                # Read the request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Forward the request to the real API
                api_url = 'https://samo-emotion-api-minimal-71517823771.us-central1.run.app/predict'
                
                req = urllib.request.Request(api_url, data=post_data, method='POST')
                req.add_header('Content-Type', 'application/json')
                
                with urllib.request.urlopen(req) as response:
                    api_response = response.read()
                    
                    # Send CORS headers
                    self.send_response(200)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    
                    # Send the API response
                    self.wfile.write(api_response)
                    
            except Exception as e:
                print(f"Error: {e}")
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = json.dumps({"error": str(e)})
                self.wfile.write(error_response.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

if __name__ == '__main__':
    port = 8081
    server = HTTPServer(('localhost', port), CORSProxyHandler)
    print(f"CORS Proxy running on http://localhost:{port}")
    print("Available endpoints:")
    print(f"  POST http://localhost:{port}/emotion")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy...")
        server.shutdown()
