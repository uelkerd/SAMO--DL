#!/usr/bin/env python3

import os
import sys
import time

print("=== MINIMAL CLOUD RUN TEST ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Environment variables:")
for key, value in os.environ.items():
    print(f"  {key}: {value}")

# Get port from environment
port = int(os.environ.get('PORT', 8080))
print(f"Starting server on port: {port}")

# Simple HTTP server
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class SimpleHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Hello from Cloud Run!")
        
        def log_message(self, format, *args):
            print(f"Request: {format % args}")
    
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    print(f"Server started on port {port}")
    server.serve_forever()
    
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1) 