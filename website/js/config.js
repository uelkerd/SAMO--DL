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
            HEALTH: '/health',
            READY: '/ready',
            TRANSCRIBE: '/transcribe'
        },
        TIMEOUT: 45000, // 45 seconds (emotion analysis can take ~28s)
        RETRY_ATTEMPTS: 3
    },
    
    // OpenAI Configuration (for client-side text generation)
    OPENAI: {
        API_KEY: '', // Set via environment or server injection
        API_URL: 'https://api.openai.com/v1/chat/completions',
        MODEL: 'gpt-3.5-turbo',
        MAX_TOKENS: 200,
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
        ENABLE_OPENAI: true, // Enabled by default for core functionality
        ENABLE_MOCK_DATA: true,
        ENABLE_ANALYTICS: false
    }
};

// Environment-specific overrides
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.SAMO_CONFIG.ENVIRONMENT = 'development';
    window.SAMO_CONFIG.DEBUG = true;
    window.SAMO_CONFIG.FEATURES.ENABLE_MOCK_DATA = true;

    // Use local mock server for development
    if (window.location.port === '8000') {
        window.SAMO_CONFIG.API.BASE_URL = 'http://localhost:8000';
        window.SAMO_CONFIG.API.ENDPOINTS = {
            EMOTION: '/api/emotion',
            SUMMARIZE: '/api/summarize',
            JOURNAL: '/api/summarize', // Use summarize for journal
            HEALTH: '/api/health',
            READY: '/api/health',
            TRANSCRIBE: '/api/transcribe'
        };
    }
}

// Server-side configuration injection (if available)
if (window.SAMO_SERVER_CONFIG) {
    Object.assign(window.SAMO_CONFIG, window.SAMO_SERVER_CONFIG);
}

// Only log config in debug mode and redact sensitive fields
if (window.SAMO_CONFIG && window.SAMO_CONFIG.DEBUG) {
    const sanitizedConfig = { ...window.SAMO_CONFIG };
    const sensitiveKeys = ['apiKey', 'secret', 'token', 'password', 'clientSecret'];
    
    // Redact sensitive fields
    sensitiveKeys.forEach(key => {
        if (sanitizedConfig[key]) {
            sanitizedConfig[key] = 'REDACTED';
        }
    });
    
    // Also check nested objects
    if (sanitizedConfig.API) {
        sensitiveKeys.forEach(key => {
            if (sanitizedConfig.API[key]) {
                sanitizedConfig.API[key] = 'REDACTED';
            }
        });
    }
    
    if (sanitizedConfig.OPENAI) {
        sensitiveKeys.forEach(key => {
            if (sanitizedConfig.OPENAI[key]) {
                sanitizedConfig.OPENAI[key] = 'REDACTED';
            }
        });
    }
    
    console.log('🔧 SAMO Configuration loaded (debug mode):', sanitizedConfig);
}
