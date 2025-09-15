/**
 * SAMO API Client
 * Centralized API client using configuration from config.js
 */

class SAMOAPIClient {
    constructor() {
        this.baseURL = window.SAMO_CONFIG?.API?.BASE_URL || 'https://samo-unified-api-frrnetyhfa-uc.a.run.app';
        this.endpoints = window.SAMO_CONFIG?.API?.ENDPOINTS || {
            EMOTION: '/analyze/emotion',
            JOURNAL: '/analyze/journal',
            HEALTH: '/health',
            TRANSCRIBE: '/transcribe'
        };
        this.timeout = window.SAMO_CONFIG?.API?.TIMEOUT || 30000;
        this.retryAttempts = window.SAMO_CONFIG?.API?.RETRY_ATTEMPTS || 3;
    }

    /**
     * Make a request to the SAMO API
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise<Response>}
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: this.timeout
        };

        const requestOptions = { ...defaultOptions, ...options };

        // Add timeout handling
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        requestOptions.signal = controller.signal;

        try {
            const response = await fetch(url, requestOptions);
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }

    /**
     * Analyze emotion from text
     * @param {string} text - Text to analyze
     * @returns {Promise<Object>}
     */
    async analyzeEmotion(text) {
        return this.request(this.endpoints.EMOTION, {
            body: JSON.stringify({ text })
        });
    }

    /**
     * Analyze journal entry
     * @param {string} text - Journal text
     * @param {boolean} generateSummary - Whether to generate summary
     * @returns {Promise<Object>}
     */
    async analyzeJournal(text, generateSummary = false) {
        return this.request(this.endpoints.JOURNAL, {
            body: JSON.stringify({ 
                text, 
                generate_summary: generateSummary 
            })
        });
    }

    /**
     * Check API health
     * @returns {Promise<Object>}
     */
    async checkHealth() {
        return this.request(this.endpoints.HEALTH, {
            method: 'GET'
        });
    }

    /**
     * Transcribe audio
     * @param {File|Blob} audioFile - Audio file to transcribe
     * @param {string} language - Optional language hint
     * @returns {Promise<Object>}
     */
    async transcribe(audioFile, language = null) {
        const formData = new FormData();
        formData.append('audio_file', audioFile);
        if (language) {
            formData.append('language', language);
        }

        return this.request(this.endpoints.TRANSCRIBE, {
            headers: {}, // Remove Content-Type for FormData
            body: formData
        });
    }

    /**
     * Get API base URL for display purposes
     * @returns {string}
     */
    getBaseURL() {
        return this.baseURL;
    }

    /**
     * Check if API is configured
     * @returns {boolean}
     */
    isConfigured() {
        return !!(this.baseURL && this.endpoints);
    }

    // Backward-compatibility aliases
    async analyzeText(text) {
        return this.analyzeJournal(text);
    }

    async transcribeAudio(audioFile) {
        return this.transcribe(audioFile);
    }

    async detectEmotions(text) {
        return this.analyzeEmotion(text);
    }
}

// Export for use in other scripts
window.SAMOAPIClient = SAMOAPIClient;