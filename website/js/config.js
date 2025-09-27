// SAMO-DL API Configuration
// This file centralizes API endpoint configuration for the website

window.SAMO_CONFIG = {
  API: {
    // Base URL for the SAMO-DL API
    // Replace with your actual deployment URL
    BASE_URL: 'https://samo-unified-api-frrnetyhfa-uc.a.run.app',

    // Alternative URLs for different environments
    // Uncomment and modify as needed
    // DEV_URL: 'http://localhost:8000',
    // STAGING_URL: 'https://samo-staging-api.example.com',
    // PROD_URL: 'https://samo-prod-api.example.com',

    // API endpoints
    ENDPOINTS: {
      PREDICT: '/predict',
      PREDICT_BATCH: '/predict_batch',
      TRANSCRIBE_VOICE: '/transcribe/voice',
      TRANSCRIBE_BATCH: '/transcribe/batch',
      SUMMARIZE_TEXT: '/summarize/text',
      ANALYZE_JOURNAL: '/analyze/journal',
      HEALTH: '/health',
      METRICS: '/metrics',
      MONITORING_DASHBOARD: '/monitoring/dashboard',
      VERSION: '/version',
      DOCS: '/docs',
      OPENAPI: '/openapi.json',

      // Authentication endpoints
      AUTH: {
        REGISTER: '/auth/register',
        LOGIN: '/auth/login',
        REFRESH: '/auth/refresh',
        LOGOUT: '/auth/logout',
        PROFILE: '/auth/profile'
      },

      // WebSocket endpoints
      WS: {
        CHAT: '/ws/chat',
        TRANSCRIBE: '/ws/transcribe'
      }
    },

    // Default timeout settings (in milliseconds)
    TIMEOUTS: {
      DEFAULT: 10000,    // 10 seconds
      LONG_RUNNING: 30000, // 30 seconds
      WEBSOCKET: 5000    // 5 seconds
    },

    // Rate limiting configuration
    RATE_LIMITS: {
      MAX_REQUESTS_PER_MINUTE: 60,
      BURST_LIMIT: 10
    }
  },

  // UI Configuration
  UI: {
    // Demo settings
    DEMO: {
      MAX_TEXT_LENGTH: 5000,
      MAX_BATCH_SIZE: 10,
      ENABLE_VOICE_RECORDING: true,
      ENABLE_WEBSOCKET: true
    },

    // Monitoring refresh interval (in milliseconds)
    MONITORING_REFRESH_INTERVAL: 30000, // 30 seconds

    // Animation settings
    ANIMATIONS: {
      ENABLED: true,
      DURATION: 300
    }
  },

  // Feature flags
  FEATURES: {
    ENABLE_AUTH: true,
    ENABLE_VOICE_TRANSCRIPTION: true,
    ENABLE_BATCH_PROCESSING: true,
    ENABLE_TEXT_SUMMARIZATION: true,
    ENABLE_REAL_TIME_MONITORING: true,
    ENABLE_SECURITY_TESTING: true,
    ENABLE_WEBSOCKET_CHAT: true
  }
};

// Helper function to get API URL with fallback
window.SAMO_CONFIG.getApiUrl = function(endpoint) {
  const baseUrl = this.API.BASE_URL;
  if (!baseUrl) {
    console.warn('SAMO_CONFIG.API.BASE_URL is not set. Please configure your API endpoint.');
    return null;
  }

  // Remove trailing slash from base URL and leading slash from endpoint
  const cleanBaseUrl = baseUrl.replace(/\/$/, '');
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : '/' + endpoint;

  return cleanBaseUrl + cleanEndpoint;
};

// Helper function to get WebSocket URL
window.SAMO_CONFIG.getWebSocketUrl = function(endpoint) {
  const baseUrl = this.API.BASE_URL;
  if (!baseUrl) {
    console.warn('SAMO_CONFIG.API.BASE_URL is not set. Cannot create WebSocket URL.');
    return null;
  }

  // Convert HTTPS to WSS
  let wsUrl = baseUrl.replace(/^https:/, 'wss:').replace(/^http:/, 'ws:');

  // Remove trailing slash from base URL and leading slash from endpoint
  wsUrl = wsUrl.replace(/\/$/, '');
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : '/' + endpoint;

  return wsUrl + cleanEndpoint;
};

console.log('SAMO-DL configuration loaded successfully');
