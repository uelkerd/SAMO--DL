/**
 * API Configuration Example for SAMO-DL Demo
 * Copy this file to config.js and customize for your environment
 * DO NOT commit config.js with real API keys to version control
 * 
 * Environment-specific configuration:
 * - Local development: Uses localhost:8080
 * - Production: Uses relative /api proxy path
 * - Custom: Override via SAMO_CONFIG environment variable
 */

// API Configuration - Environment Detection
const isLocalDev = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' ||
                   window.location.hostname === '';

const SAMO_CONFIG = {
    // Use relative path for production, localhost for development
    baseURL: isLocalDev ? 'http://localhost:8080' : '/api',
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
