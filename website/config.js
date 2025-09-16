// SAMO-DL Demo Configuration
// DO NOT COMMIT TOKENS TO VERSION CONTROL!

window.SAMO_CONFIG = {
    // OpenAI API Configuration
    OPENAI: {
        // Use proxy endpoint instead of direct API access for security
        PROXY_URL: 'https://samo-unified-api-frrnetyhfa-uc.a.run.app/generate/journal',
        MODEL: 'gpt-3.5-turbo',
        MAX_TOKENS: 200,
        TEMPERATURE: 0.8
    },
    
    // Hugging Face API Configuration (fallback)
    HUGGING_FACE: {
        // Replace with your actual token - DO NOT COMMIT THIS FILE WITH REAL TOKENS
        API_TOKEN: 'hf_your_token_here', // Replace with: hf_your_actual_token_here
        MODEL: 'distilgpt2', // Fallback models: 'gpt2', 'microsoft/DialoGPT-medium'
        MAX_LENGTH: 150,
        TEMPERATURE: 0.8
    },
    
    // Emotion API Configuration  
    EMOTION_API: {
        ENDPOINT: 'https://samo-unified-api-71517823771.us-central1.run.app/analyze/emotion',
        TIMEOUT: 10000
    },
    
    // Feature flags
    FEATURES: {
        ENABLE_OPENAI: false  // Set to false for public builds to prevent API key exposure
    },
    
    // Demo Configuration
    DEMO: {
        FALLBACK_TO_STATIC: true,
        SHOW_DEBUG_INFO: true
    }
};

// Security warning
console.warn('🔒 SECURITY: OpenAI integration disabled for public builds - using proxy endpoint instead');
console.warn('🔒 SECURITY: Never commit API keys to version control!');