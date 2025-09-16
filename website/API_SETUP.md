# JWT Authentication Setup for SAMO-DL Demo

## Overview
The comprehensive demo connects to the SAMO Unified API. The service requires JWT (Bearer token) authentication for all non-health endpoints. This document explains how to set up the configuration securely.

## Setup Instructions

### 1. Create the Configuration File
Create a `config.js` file in the `website/` directory with the following content:

```javascript
/**
 * API Configuration for SAMO-DL Demo
 * This file contains the API configuration
 * DO NOT commit this file with real API keys to version control
 */

// API Configuration
const SAMO_CONFIG = {
    baseURL: 'https://samo-unified-api-frrnetyhfa-uc.a.run.app',
    jwtToken: null, // JWT Bearer token required for authentication
    timeout: 30000,
    retryAttempts: 3
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SAMO_CONFIG;
} else {
    window.SAMO_CONFIG = SAMO_CONFIG;
}
```

### 2. JWT Authentication Setup
The SAMO Unified API requires JWT (Bearer token) authentication for all non-health endpoints. To authenticate:

1. **Obtain a JWT token** from your API provider or authentication service
2. **Add the token to your config**:
   ```javascript
   const SAMO_CONFIG = {
       baseURL: 'https://samo-unified-api-frrnetyhfa-uc.a.run.app',
       jwtToken: 'your-jwt-token-here', // Replace with actual JWT
       timeout: 30000,
       retryAttempts: 3
   };
   ```
3. **Include in requests**: The token should be sent in the Authorization header:
   ```
   Authorization: Bearer <your-jwt-token>
   ```

### 3. Secure Token Storage
- Store JWT tokens in environment variables, not in code
- Use secure token management practices
- Rotate tokens regularly for security

### 4. Security Notes
- The `config.js` file is already added to `.gitignore` to prevent accidental commits
- Never commit API keys to version control
- The demo will fall back to mock data if the config file is not available

## Testing the Demo

1. Start the local HTTP server:
   ```bash
   cd website
   python3 -m http.server 8080
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8080/comprehensive-demo.html
   ```

3. Test with the provided journal text samples

## Troubleshooting

### API Key Issues
- Ensure the API key is correctly set in `config.js`
- Check that the API key has the correct permissions
- Verify the API service is running and accessible

### Rate Limiting
- The API has rate limits (300 requests per minute in production)
- Rate limits may vary by environment - check your deployment configuration
- If you hit rate limits, the demo will show mock data
- Wait for the rate limit to reset before trying again
- Rate limit headers are included in responses for monitoring

### Fallback Mode
- If the API is unavailable, the demo will automatically use mock data
- This ensures the demo always works for demonstration purposes
