/**
 * SAMO-DL API Configuration
 * Centralized configuration for API endpoints and keys
 * This file should be loaded before other JavaScript files
 */

window.SAMO_CONFIG = {
  // API Configuration
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
      EMOTION: '/analyze/emotion',
      SUMMARIZE: '/analyze/summarize',
      JOURNAL: '/analyze/journal',
      VOICE_JOURNAL: '/analyze/voice-journal',
      HEALTH: '/health',
      READY: '/ready',
      TRANSCRIBE: '/transcribe',
      OPENAI_PROXY: '/proxy/openai',

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
    },

    // Legacy compatibility - keep these for backward compatibility
    TIMEOUT: 45000, // 45 seconds (emotion analysis can take ~28s)
    RETRY_ATTEMPTS: 3,
    API_KEY: null, // Set via server injection or user input // skipcq: SCT-A000
    REQUIRE_AUTH: true // Secure default; overridden to false only for localhost dev
  },

    // OpenAI Configuration - DISABLED for security (use proxy instead)
    // WARNING: Direct client-side OpenAI calls expose API keys!
    // All OpenAI functionality must use the /proxy/openai endpoint
    OPENAI: {
        // API_URL: 'https://api.openai.com/v1/chat/completions', // DISABLED - Security risk
        // MODEL: 'gpt-4o-mini', // DISABLED - Use proxy endpoint
        // MAX_TOKENS: 4000, // DISABLED - Use proxy endpoint
        // TEMPERATURE: 0.7 // DISABLED - Use proxy endpoint
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

  // External Services
  EXTERNAL: {
    HUGGINGFACE: {
      API_URL: 'https://api-inference.huggingface.co/models/gpt2',
      MAX_LENGTH: 150
    },
    GOOGLE_FONTS: 'https://fonts.googleapis.com',
    CDN: {
      BOOTSTRAP: 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
      CHART_JS: 'https://cdn.jsdelivr.net/npm/chart.js',
      FONT_AWESOME: 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
    }
  },

  // Development/Production flags
  ENVIRONMENT: 'production', // 'development' or 'production'
  DEBUG: false,

    // Feature flags
    FEATURES: {
        ENABLE_OPENAI: false, // DISABLED - Security risk (use /proxy/openai endpoint instead)
        ENABLE_MOCK_DATA: false, // Always use real APIs
        ENABLE_ANALYTICS: false,
        ENABLE_AUTH: true,
        ENABLE_VOICE_TRANSCRIPTION: true,
        ENABLE_BATCH_PROCESSING: true,
        ENABLE_TEXT_SUMMARIZATION: true,
        ENABLE_REAL_TIME_MONITORING: true,
        ENABLE_SECURITY_TESTING: true,
        ENABLE_WEBSOCKET_CHAT: true
    }
};

// Environment-specific overrides - USE DEPLOYED API FOR DEVELOPMENT
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.SAMO_CONFIG.ENVIRONMENT = 'development';
    window.SAMO_CONFIG.DEBUG = true;

    // Use local unified API server; match page scheme to avoid cert/mixed-content issues
    const scheme = window.location.protocol === 'https:' ? 'https' : 'http';
    const host = window.location.hostname === '127.0.0.1' ? '127.0.0.1' : 'localhost';
    window.SAMO_CONFIG.API.BASE_URL = `${scheme}://${host}:8002`;
    window.SAMO_CONFIG.API.REQUIRE_AUTH = false; // Disable auth for local dev convenience
    // Note: ENDPOINTS remain unchanged from production config (no override needed)

    console.log(`🔧 Running in localhost development mode - using local API server at ${window.SAMO_CONFIG.API.BASE_URL}`);
}

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

// Deep merge utility function
window.SAMO_CONFIG.deepMerge = function(target, source) {
    const result = { ...target };

    for (const key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                // Recursively merge objects
                result[key] = this.deepMerge(target[key] || {}, source[key]);
            } else {
                // Replace primitives and arrays
                result[key] = source[key];
            }
        }
    }

    return result;
};

// Server-side configuration injection (if available)
if (window.SAMO_SERVER_CONFIG) {
    window.SAMO_CONFIG = window.SAMO_CONFIG.deepMerge(window.SAMO_CONFIG, window.SAMO_SERVER_CONFIG);
}

// Recursive redaction utility function
window.SAMO_CONFIG.redactSensitiveValues = function(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (Array.isArray(obj)) {
        return obj.map(item => this.redactSensitiveValues(item));
    }

    const result = {};
    const SENSITIVE_PATTERNS = [
      /^(api[-_]?key|authorization|x[-_]?api[-_]?key|bearer)$/i, // skipcq: SCT-A000 - security pattern definitions
      /^(token|access[_-]?token|refresh[_-]?token)$/i, // skipcq: SCT-A000 - security pattern definitions
      /^(secret|client[_-]?secret)$/i, // skipcq: SCT-A000 - security pattern definitions
      /^(password|passwd)$/i, // skipcq: SCT-A000 - security pattern definitions
      /^(credential|credentials|auth|authkey)$/i // skipcq: SCT-A000 - security pattern definitions
    ];

    for (const [key, value] of Object.entries(obj)) {
        const isSensitive = SENSITIVE_PATTERNS.some(re => re.test(key));

        if (isSensitive) {
            result[key] = 'REDACTED';
        } else if (value && typeof value === 'object') {
            result[key] = this.redactSensitiveValues(value);
        } else {
            result[key] = value;
        }
    }

    return result;
};

// Only log config in debug mode and redact sensitive fields
if (window.SAMO_CONFIG && window.SAMO_CONFIG.DEBUG) {
    const sanitizedConfig = window.SAMO_CONFIG.redactSensitiveValues(window.SAMO_CONFIG);
    console.log('🔧 SAMO Configuration loaded (debug mode):', sanitizedConfig);
}

console.log('SAMO-DL configuration loaded successfully');
