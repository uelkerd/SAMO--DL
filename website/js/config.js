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
            VOICE_JOURNAL: '/analyze/voice_journal',
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
        ENABLE_OPENAI: false, // Disabled by default - requires server-side proxy
        ENABLE_MOCK_DATA: false, // Always use real APIs
        ENABLE_ANALYTICS: false
    }
};

// Environment-specific overrides - ALWAYS USE REAL APIS
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.SAMO_CONFIG.ENVIRONMENT = 'development';
    window.SAMO_CONFIG.DEBUG = true;

    // For demo testing, use production API directly (CORS is enabled on the server)
    // Keep production URL and endpoints for localhost development
    console.log('🔧 Running in localhost development mode - using production API with CORS');
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
    const sensitiveKeys = [
        'apikey', 'api_key', 'apiKey', 'secret', 'token', 'authorization', 
        'password', 'clientsecret', 'client_secret', 'clientSecret',
        'key', 'keys', 'credential', 'credentials', 'auth', 'authkey'
    ];
    
    for (const [key, value] of Object.entries(obj)) {
        const keyLower = key.toLowerCase();
        const isSensitive = sensitiveKeys.some(sensitiveKey => 
            keyLower.includes(sensitiveKey) || sensitiveKey.includes(keyLower)
        );
        
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
