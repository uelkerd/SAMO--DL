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
     * Make a request to the SAMO API with retry logic and JSON parsing
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Parsed JSON response
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const defaultOptions = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        };

        const baseDelay = 1000; // 1 second base delay
        let lastError;

        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            let controller = null;
            let timeoutId = null;

            try {
                // Create new controller and timeout for each attempt
                controller = new AbortController();
                timeoutId = setTimeout(() => controller.abort(), this.timeout);

                const requestOptions = { 
                    ...defaultOptions, 
                    ...options,
                    signal: controller.signal
                };

                const response = await fetch(url, requestOptions);
                clearTimeout(timeoutId);

                // Parse response JSON
                let responseData;
                try {
                    responseData = await response.json();
                } catch (parseError) {
                    // Fallback to text if JSON parsing fails
                    const textData = await response.text();
                    responseData = { message: textData, raw: true };
                }

                // Handle success (2xx status codes)
                if (response.ok) {
                    return responseData;
                }

                // Handle transient errors (429, 503) - retry with exponential backoff
                if ((response.status === 429 || response.status === 503) && attempt < this.retryAttempts) {
                    const jitter = Math.random() * 0.1; // 0-10% jitter
                    const delay = baseDelay * Math.pow(2, attempt - 1) * (1 + jitter);
                    
                    console.warn(`Request failed with status ${response.status}, retrying in ${Math.round(delay)}ms (attempt ${attempt}/${this.retryAttempts})`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }

                // Handle final failure or non-retryable errors
                const error = new Error(`HTTP ${response.status}: ${response.statusText}`);
                error.status = response.status;
                error.statusText = response.statusText;
                error.response = responseData;
                throw error;

            } catch (error) {
                clearTimeout(timeoutId);
                lastError = error;

                // Handle network aborts and timeouts - retry if we have attempts left
                if ((error.name === 'AbortError' || error.name === 'TimeoutError') && attempt < this.retryAttempts) {
                    const jitter = Math.random() * 0.1;
                    const delay = baseDelay * Math.pow(2, attempt - 1) * (1 + jitter);
                    
                    console.warn(`Request aborted/timed out, retrying in ${Math.round(delay)}ms (attempt ${attempt}/${this.retryAttempts})`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }

                // For other errors or final attempt, throw immediately
                throw error;
            } finally {
                // Clean up controller and timeout
                if (controller) {
                    controller.abort();
                }
                if (timeoutId) {
                    clearTimeout(timeoutId);
                }
            }
        }

        // If we get here, all retries failed
        throw lastError;
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