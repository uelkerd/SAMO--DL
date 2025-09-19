/**
 * SAMOAPIClient Module for Testing
 * Extracts the SAMOAPIClient class from comprehensive-demo.js for testing
 */

// Mock the browser environment globals
if (typeof window === 'undefined') {
  global.window = {
    SAMO_CONFIG: {
      API: {
        BASE_URL: 'https://test-api.com',
        ENDPOINTS: {
          EMOTION: '/analyze/emotion',
          SUMMARIZE: '/analyze/summarize',
          VOICE_JOURNAL: '/analyze/voice-journal',
          HEALTH: '/health'
        },
        TIMEOUT: 15000,
        COLD_START_TIMEOUT: 45000,
        RETRY_ATTEMPTS: 3,
        API_KEY: null,
        API_KEY_ENV: null
      }
    }
  };
}

if (typeof localStorage === 'undefined') {
  global.localStorage = {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
    clear: () => {}
  };
}

// SAMOAPIClient class extracted from comprehensive-demo.js
class SAMOAPIClient {
    constructor() {
        // Use centralized configuration
        const windowRef = global.window || (typeof window !== 'undefined' ? window : {});
        if (!windowRef.SAMO_CONFIG) {
            console.warn('⚠️ SAMO_CONFIG not found, using fallback configuration');
        }

        this.baseURL = windowRef.SAMO_CONFIG?.API?.BASE_URL || 'https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app';
        this.endpoints = windowRef.SAMO_CONFIG?.API?.ENDPOINTS || {
            EMOTION: '/analyze/emotion',
            SUMMARIZE: '/analyze/summarize',
            JOURNAL: '/analyze/journal',
            HEALTH: '/health',
            READY: '/ready',
            TRANSCRIBE: '/transcribe',
            VOICE_JOURNAL: '/analyze/voice-journal'  // Match actual API endpoint
        };

        // Ensure VOICE_JOURNAL has a fallback if missing from config
        if (!this.endpoints.VOICE_JOURNAL) {
            this.endpoints.VOICE_JOURNAL = '/analyze/voice-journal';
        }

        // Optimized timeout configuration for better UX
        this.timeout = windowRef.SAMO_CONFIG?.API?.TIMEOUT || 15000; // Reduced from 20s to 15s
        this.coldStartTimeout = windowRef.SAMO_CONFIG?.API?.COLD_START_TIMEOUT || 45000; // Reduced from 60s to 45s
        this.retryAttempts = windowRef.SAMO_CONFIG?.API?.RETRY_ATTEMPTS || 1; // Reduced to 1 for faster feedback
        this.isColdStart = true; // Track if this is the first request
    }

    getApiKey() {
        // Try to get API key from various sources
        // 1. From SAMO_CONFIG (server-injected)
        const windowRef = global.window || (typeof window !== 'undefined' ? window : {});
        if (windowRef.SAMO_CONFIG?.API?.API_KEY) {
            return windowRef.SAMO_CONFIG.API.API_KEY;
        }

        // 2. From localStorage (user-set)
        const storedKey = localStorage.getItem('samo_api_key');
        if (storedKey && storedKey.trim()) {
            return storedKey.trim();
        }

        // 3. From environment variable (if available in browser context)
        if (windowRef.SAMO_CONFIG?.API?.API_KEY_ENV) {
            return windowRef.SAMO_CONFIG.API.API_KEY_ENV;
        }

        return null;
    }

    async makeRequest(endpoint, data, method = 'POST', isFormData = false, timeoutMs = null) {
        return this.makeRequestWithRetry(endpoint, data, method, isFormData, timeoutMs, this.retryAttempts);
    }

    // Helper method to build query string for deployed API format
    buildQueryString(data) {
        if (!data || typeof data !== 'object') return '';
        const params = new URLSearchParams();
        for (const [key, value] of Object.entries(data)) {
            if (value !== null && value !== undefined) {
                params.append(key, value);
            }
        }
        return params.toString();
    }

    async makeRequestWithRetry(endpoint, data, method = 'POST', isFormData = false, timeoutMs = null, attemptsLeft = null) {
        // Use class defaults if not specified
        if (attemptsLeft === null) attemptsLeft = this.retryAttempts;

        const config = {
            method,
            headers: {}
        };
        const controller = new AbortController();

        // Use cold start timeout for first request, regular timeout otherwise
        const timeout = timeoutMs || (this.isColdStart ? this.coldStartTimeout : this.timeout);
        const timer = setTimeout(() => {
            controller.abort(new Error(`Request timeout after ${timeout/1000}s`));
        }, timeout);
        config.signal = controller.signal;

        // Track this request in LayoutManager if available
        if (typeof LayoutManager !== 'undefined') {
            LayoutManager.addActiveRequest(controller);
        }

        // Add API key for production endpoints if available
        const apiKey = this.getApiKey();
        if (apiKey) {
            config.headers['X-API-Key'] = apiKey;
        }

        if (data && method === 'POST') {
            if (isFormData) {
                // For FormData, don't set Content-Type header - let browser set it with boundary
                config.body = data;
            } else {
                // For deployed API, use query parameters instead of JSON body
                const queryString = this.buildQueryString(data);
                if (queryString) {
                    endpoint += `?${queryString}`;
                }
                config.headers['Content-Type'] = 'application/json';
            }
        } else if (method === 'GET') {
            config.headers['Content-Type'] = 'application/json';
        }

        try {
            const url = `${this.baseURL}${endpoint}`;

            // Log retry attempt info for user feedback
            const attemptNumber = this.retryAttempts - attemptsLeft + 1;
            if (attemptNumber > 1) {
                console.log(`🔄 Retry attempt ${attemptNumber}/${this.retryAttempts} for ${endpoint}`);
                if (typeof addToProgressConsole === 'function') {
                    addToProgressConsole(`Retry attempt ${attemptNumber}/${this.retryAttempts} - ${endpoint}`, 'warning');
                }
            }

            const response = await fetch(url, config);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const msg = errorData.message || errorData.error || `HTTP ${response.status}`;

                // Handle retryable errors
                if (response.status === 429 || response.status >= 500) {
                    if (attemptsLeft > 1) {
                        const backoffDelay = Math.pow(2, this.retryAttempts - attemptsLeft) * 1000; // Exponential backoff
                        console.warn(`Request failed (${response.status}), retrying in ${backoffDelay}ms. Attempts left: ${attemptsLeft - 1}`);

                        // Provide user feedback about retry
                        if (typeof addToProgressConsole === 'function') {
                            addToProgressConsole(`Request failed (${response.status}), retrying in ${backoffDelay/1000}s...`, 'warning');
                        }

                        await new Promise(resolve => setTimeout(resolve, backoffDelay));
                        return this.makeRequestWithRetry(endpoint, data, method, isFormData, timeoutMs, attemptsLeft - 1);
                    }
                }

                // Non-retryable errors or out of retries
                if (response.status === 429) throw new Error(msg || 'Rate limit exceeded. Please try again shortly.');
                if (response.status === 401) throw new Error(msg || 'API key required.');
                if (response.status === 503) throw new Error(msg || 'Service temporarily unavailable.');
                throw new Error(msg);
            }

            // Mark cold start as complete after first successful request
            if (this.isColdStart) {
                this.isColdStart = false;
                console.log('✅ Cold start completed, future requests will use faster timeout');
            }

            return await response.json();
        } catch (error) {
            // Handle network errors with retry
            if ((error.name === 'AbortError' || error.message.includes('timeout') || error.message.includes('network')) && attemptsLeft > 1) {
                const backoffDelay = Math.pow(2, this.retryAttempts - attemptsLeft) * 1000;
                console.warn(`Network error, retrying in ${backoffDelay}ms. Attempts left: ${attemptsLeft - 1}`, error.message);

                // Provide user feedback about network retry
                if (typeof addToProgressConsole === 'function') {
                    addToProgressConsole(`Network error, retrying in ${backoffDelay/1000}s...`, 'warning');
                }

                await new Promise(resolve => setTimeout(resolve, backoffDelay));
                return this.makeRequestWithRetry(endpoint, data, method, isFormData, timeoutMs, attemptsLeft - 1);
            }

            console.error('API request failed:', error);
            throw error;
        } finally {
            clearTimeout(timer);

            // Remove request from LayoutManager tracking
            if (typeof LayoutManager !== 'undefined') {
                LayoutManager.removeActiveRequest(controller);
            }
        }
    }

    async transcribeAudio(audioFile) {
        const formData = new FormData();
        formData.append('audio_file', audioFile);

        try {
            // Use VOICE_JOURNAL endpoint for audio analysis flows with proper timeout handling
            return await this.makeRequest(this.endpoints.VOICE_JOURNAL, formData, 'POST', true);
        } catch (error) {
            console.error('Transcription error:', error);
            throw error;
        }
    }

    async summarizeText(text) {
        try {
            // Use makeRequest method for proper timeout and error handling
            const response = await this.makeRequest(this.endpoints.SUMMARIZE, { text }, 'POST');

            // The makeRequest method already handles JSON parsing, so response is the data
            return response;
        } catch (error) {
            // If API is not available, return mock data for demo purposes
            if (error.message.includes('Rate limit') || error.message.includes('API key') || error.message.includes('Service temporarily') || error.message.includes('Abuse detected') || error.message.includes('Client blocked')) {
                console.warn('API not available, using mock data for demo:', error.message);
                return this.getMockSummaryResponse(text);
            }
            throw error;
        }
    }

    getMockSummaryResponse(text) {
        // Mock summarization response for demo purposes
        const words = text.split(' ');
        const summaryLength = Math.max(10, Math.floor(words.length * 0.3));
        const summary = words.slice(0, summaryLength).join(' ') + '...';

        return {
            summary: summary,
            original_length: text.length,
            summary_length: summary.length,
            compression_ratio: (summary.length / text.length).toFixed(2),
            request_id: 'demo-' + Date.now(),
            timestamp: Date.now() / 1000,
            mock: true
        };
    }

    async detectEmotions(text) {
        try {
            // Use makeRequest method for proper timeout and error handling
            const data = await this.makeRequest(this.endpoints.EMOTION, { text }, 'POST');

            // Extract top 5 emotions and sort by confidence
            const emotions = data.emotions || {};
            const emotionArray = Object.entries(emotions)
                .map(([emotion, confidence]) => ({ emotion, confidence }))
                .sort((a, b) => b.confidence - a.confidence)
                .slice(0, 5);

            return {
                ...data,
                top_emotions: emotionArray
            };
        } catch (error) {
            // If API is not available, return mock data for demo purposes
            if (error.message.includes('Rate limit') || error.message.includes('API key') || error.message.includes('Service temporarily') || error.message.includes('Abuse detected') || error.message.includes('Client blocked')) {
                console.warn('API not available, using mock data for demo:', error.message);
                return this.getMockEmotionResponse(text);
            }
            throw error;
        }
    }

    getMockEmotionResponse(text) {
        // Mock emotion detection response for demo purposes - matches new API format
        const emotions = {
            'admiration': 0.12,
            'amusement': 0.08,
            'anger': 0.02,
            'annoyance': 0.01,
            'approval': 0.15,
            'caring': 0.05,
            'confusion': 0.03,
            'curiosity': 0.18,
            'desire': 0.04,
            'disappointment': 0.02,
            'disapproval': 0.01,
            'disgust': 0.01,
            'embarrassment': 0.01,
            'excitement': 0.85,
            'fear': 0.02,
            'gratitude': 0.12,
            'grief': 0.01,
            'joy': 0.72,
            'love': 0.08,
            'nervousness': 0.03,
            'optimism': 0.68,
            'pride': 0.05,
            'realization': 0.06,
            'relief': 0.04,
            'remorse': 0.01,
            'sadness': 0.02,
            'surprise': 0.15,
            'neutral': 0.08
        };

        // Create top_emotions array for bar graphs
        const emotionArray = Object.entries(emotions)
            .map(([emotion, confidence]) => ({ emotion, confidence }))
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 5);

        return {
            text: text,
            emotions: emotions,
            predicted_emotion: emotionArray[0].emotion,
            top_emotions: emotionArray,
            request_id: 'demo-' + Date.now(),
            timestamp: Date.now() / 1000,
            mock: true
        };
    }

    async processCompleteWorkflow(audioFile, text) {
        const results = {
            transcription: null,
            summary: null,
            emotions: null,
            processingTime: 0,
            modelsUsed: []
        };

        const startTime = Date.now();
        let currentText = text;

        // Step 1: Transcribe audio if provided
        if (audioFile) {
            try {
                const audioResponse = await this.transcribeAudio(audioFile);
                // Map transcription, summary and emotion_analysis from unified response
                results.transcription = audioResponse.transcription || audioResponse;
                results.summary = audioResponse.summary || null;
                results.emotions = audioResponse.emotion_analysis || null;

                // Extract transcribed text for further processing if needed
                const transcribedText = results.transcription.text || results.transcription.transcription;
                currentText = transcribedText;
                results.modelsUsed.push('SAMO Whisper');
            } catch (error) {
                console.error('Transcription failed:', error);
                throw new Error('Voice transcription failed. Please try again.');
            }
        }

        // Step 2: Summarize text (if not already done in audio processing)
        if (currentText && !results.summary) {
            try {
                results.summary = await this.summarizeText(currentText);
                results.modelsUsed.push('SAMO T5');
            } catch (error) {
                console.error('Summarization failed:', error);
                // Continue without summary
            }
        }

        // Step 3: Detect emotions (if not already done in audio processing)
        if (currentText && !results.emotions) {
            try {
                results.emotions = await this.detectEmotions(currentText);
                results.modelsUsed.push('SAMO DeBERTa v3 Large');
            } catch (error) {
                console.error('Emotion detection failed:', error);
                throw new Error('Emotion detection failed. Please try again.');
            }
        }

        results.processingTime = Date.now() - startTime;
        return results;
    }
}

module.exports = SAMOAPIClient;