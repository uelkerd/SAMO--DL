#!/usr/bin/env python3
"""
CORS Proxy for SAMO Emotion API
This script creates a local proxy to bypass CORS restrictions
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import json
import requests

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

                # Parse the JSON to extract the text
                request_data = json.loads(post_data.decode('utf-8'))
                text = request_data.get('text', '')

                # URL encode the text for query parameter
                encoded_text = urllib.parse.quote(text)

                # Forward the request to the unified API with query parameters
                api_url = f'https://samo-unified-api-71517823771.us-central1.run.app/analyze/emotion?text={encoded_text}'

                # Make POST request with query parameters using secure requests library
                headers = {
                    'Content-Type': 'application/json',
                    'Content-Length': '0'  # Required for POST requests with query params
                }

                # Debug: Print what we're sending
                print(f"Sending to Unified API: {api_url}")

                # Use requests library for better security and SSL verification
                response = requests.post(api_url, headers=headers, timeout=30, verify=True)
                response.raise_for_status()  # Raise exception for HTTP errors

                # Send CORS headers
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()

                # Send the API response
                self.wfile.write(response.content)

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 500
                print(f"HTTP Error {status_code}: {str(e)}")
                # Forward the original error status code instead of converting to 500
                self.send_response(status_code)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = json.dumps({"error": f"API Error {status_code}: {str(e)}"})
                self.wfile.write(error_response.encode())
            except requests.exceptions.RequestException as e:
                print(f"Request Error: {str(e)}")
                self.send_response(500)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = json.dumps({"error": f"Request failed: {str(e)}"})
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
