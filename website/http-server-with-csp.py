#!/usr/bin/env python3
"""HTTP Server with proper CSP headers"""

from http.server import HTTPServer, SimpleHTTPRequestHandler

class CSPHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CSP header with frame-ancestors
        self.send_header('Content-Security-Policy', 
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "media-src 'self' blob:; "
            "worker-src 'self' blob:; "
            "object-src 'none'; "
            "connect-src 'self' http://localhost:8081 https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app https://api.samo-dl.com https://api.openai.com https://cdn.jsdelivr.net; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        super().end_headers()

if __name__ == '__main__':
    port = 8082
    server = HTTPServer(('localhost', port), CSPHTTPRequestHandler)
    print(f"HTTP Server with CSP running on http://localhost:{port}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
