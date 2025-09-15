# API Key Setup for SAMO-DL Demo

## Overview
The comprehensive demo connects to the SAMO Unified API. The current service doesn't require an API key, but this document explains how to set up the configuration securely.

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
    apiKey: null, // Current service doesn't require API key
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

### 2. Current Service Status
The current SAMO Unified API service doesn't require an API key for authentication. The service is running at:
`https://samo-unified-api-frrnetyhfa-uc.a.run.app`

### 3. Configuration
The `config.js` file is already configured with the correct service URL and no API key requirement.

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
- The API has rate limits (100 requests per minute)
- If you hit rate limits, the demo will show mock data
- Wait for the rate limit to reset before trying again

### Fallback Mode
- If the API is unavailable, the demo will automatically use mock data
- This ensures the demo always works for demonstration purposes
