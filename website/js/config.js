/**
 * SAMO Configuration
 * Centralized configuration for API endpoints and keys
 * This file should be loaded before other JavaScript files
 */

window.SAMO_CONFIG = {
    // API Configuration
    API: {
        BASE_URL: 'https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app',
        ENDPOINTS: {
            EMOTION: '/analyze/emotion',
            SUMMARIZE: '/analyze/summarize',
            JOURNAL: '/analyze/journal',
            VOICE_JOURNAL: '/analyze/voice-journal',
            HEALTH: '/health',
            READY: '/ready',
            TRANSCRIBE: '/transcribe',
            OPENAI_PROXY: '/proxy/openai'
        },
        TIMEOUT: 45000, // 45 seconds (emotion analysis can take ~28s)
        RETRY_ATTEMPTS: 3,
        API_KEY: null, // Set via server injection or user input
        REQUIRE_AUTH: false // Set to true for production with API key requirement
    },

    // OpenAI Configuration (for client-side text generation)
    OPENAI: {
        API_URL: 'https://api.openai.com/v1/chat/completions',
        MODEL: 'gpt-4o-mini',
        MAX_TOKENS: 4000, // Increased for gpt-4o-mini
        TEMPERATURE: 0.7
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
        ENABLE_OPENAI: true, // Enabled - direct OpenAI API calls
        ENABLE_MOCK_DATA: false, // Always use real APIs
        ENABLE_ANALYTICS: false
    }
};

// Environment-specific overrides - USE DEPLOYED API FOR DEVELOPMENT
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.SAMO_CONFIG.ENVIRONMENT = 'development';
    window.SAMO_CONFIG.DEBUG = true;

    // Use local unified API server (has the correct /analyze/emotion endpoints)
    // This server has the exact endpoints the frontend expects
    window.SAMO_CONFIG.API.BASE_URL = 'https://localhost:8002';
    window.SAMO_CONFIG.API.ENDPOINTS = {
        EMOTION: '/analyze/emotion',
        SUMMARIZE: '/analyze/summarize',
        VOICE_JOURNAL: '/analyze/voice-journal',
        HEALTH: '/health',
        JOURNAL: '/analyze/journal',
        READY: '/ready',
        TRANSCRIBE: '/transcribe',
        OPENAI_PROXY: '/proxy/openai'
    };

    console.log('🔧 Running in localhost development mode - using deployed Cloud Run API');
}

// Deep merge utility function
function deepMerge(target, source) {
    const result = { ...target };

    for (const key in source) {
        if (source.hasOwnProperty(key)) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                // Recursively merge objects
                result[key] = deepMerge(target[key] || {}, source[key]);
            } else {
                // Replace primitives and arrays
                result[key] = source[key];
            }
        }
    }

    return result;
}

// Server-side configuration injection (if available)
if (window.SAMO_SERVER_CONFIG) {
    window.SAMO_CONFIG = deepMerge(window.SAMO_CONFIG, window.SAMO_SERVER_CONFIG);
}

// Recursive redaction utility function
function redactSensitiveValues(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }

    if (Array.isArray(obj)) {
        return obj.map(item => redactSensitiveValues(item));
    }

    const result = {};
    const SENSITIVE_PATTERNS = [
      /^(api[-_]?key|authorization|x[-_]?api[-_]?key|bearer)$/i,
      /^(token|access[_-]?token|refresh[_-]?token)$/i,
      /^(secret|client[_-]?secret)$/i,
      /^(password|passwd)$/i,
      /^(credential|credentials|auth|authkey)$/i
    ];

    for (const [key, value] of Object.entries(obj)) {
        const isSensitive = SENSITIVE_PATTERNS.some(re => re.test(key));

        if (isSensitive) {
            result[key] = 'REDACTED';
        } else if (value && typeof value === 'object') {
            result[key] = redactSensitiveValues(value);
        } else {
            result[key] = value;
        }
    }

    return result;
}

// Only log config in debug mode and redact sensitive fields
if (window.SAMO_CONFIG && window.SAMO_CONFIG.DEBUG) {
    const sanitizedConfig = redactSensitiveValues(window.SAMO_CONFIG);
    console.log('🔧 SAMO Configuration loaded (debug mode):', sanitizedConfig);
}
