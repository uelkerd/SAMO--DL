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
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Cache-Control, Pragma, Accept, Origin, X-Requested-With')
        self.end_headers()
    
    def do_GET(self):
        # Handle GET requests (fallback)
        if self.path == '/emotion':
            self.send_response(405)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps({"error": "Method not allowed. Use POST instead."})
            self.wfile.write(response.encode())
        else:
            self.send_response(404)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/emotion':
            try:
                # Read the request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Debug: Print what we received
                print(f"Received data: {post_data.decode('utf-8')}")
                
                # Forward the request to the real API
                api_url = 'https://samo-emotion-api-minimal-71517823771.us-central1.run.app/predict'
                
                req = urllib.request.Request(api_url, data=post_data, method='POST')
                req.add_header('Content-Type', 'application/json')
                
                # Debug: Print what we're sending
                print(f"Sending to API: {post_data.decode('utf-8')}")
                
                with urllib.request.urlopen(req) as response:
                    api_response = response.read()
                    
                    # Send CORS headers
                    self.send_response(200)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    
                    # Send the API response
                    self.wfile.write(api_response)
                    
            except urllib.error.HTTPError as e:
                print(f"HTTP Error {e.code}: {e.reason}")
                # Forward the original error status code instead of converting to 500
                self.send_response(e.code)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = json.dumps({"error": f"API Error {e.code}: {e.reason}"})
                self.wfile.write(error_response.encode())
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
        # Enable logging to see requests
        print(f"[{self.date_time_string()}] {format % args}")

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
