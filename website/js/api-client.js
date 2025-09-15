/**
 * API Client Module
 * Handles all API communication for the SAMO Demo
 */
class SAMOAPIClient {
    constructor() {
        // Initialize configuration with proper fallbacks
        this.config = null;
        this.baseURL = null;
        this.apiKey = null;
        this.timeout = 30000;
        this.retryAttempts = 3;
        this.initialized = false;
        
        // Initialize asynchronously
        this.initializeConfig();
    }

    async initializeConfig() {
        try {
            // Priority order: environment variables > config.js > sensible defaults
            const config = {
                baseURL: await this.getBaseURL(),
                apiKey: this.getAPIKey(),
                timeout: this.getTimeout(),
                retryAttempts: this.getRetryAttempts()
            };

            // Validate configuration
            this.validateConfig(config);
            
            this.config = config;
            this.baseURL = config.baseURL;
            this.apiKey = config.apiKey;
            this.timeout = config.timeout;
            this.retryAttempts = config.retryAttempts;
            this.initialized = true;
            
            console.log('API Client initialized with baseURL:', this.baseURL);
        } catch (error) {
            console.error('Failed to initialize API client configuration:', error);
            // Use fallback configuration
            this.baseURL = 'https://samo-unified-api-frrnetyhfa-uc.a.run.app';
            this.initialized = true;
        }
    }

    async getBaseURL() {
        // 1. Check for environment-specific configuration
        if (typeof SAMO_CONFIG !== 'undefined' && SAMO_CONFIG.baseURL) {
            return SAMO_CONFIG.baseURL;
        }

        // 2. Check for build-time environment variables (if available)
        if (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_URL) {
            return process.env.REACT_APP_API_URL;
        }

        // 3. Try to fetch server-side configuration (for production)
        if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
            try {
                const serverConfig = await this.fetchServerConfig();
                if (serverConfig && serverConfig.baseURL) {
                    return serverConfig.baseURL;
                }
            } catch (error) {
                console.warn('Failed to fetch server configuration, using fallback:', error.message);
            }
            // Use relative API proxy path for production fallback
            return '/api';
        }

        // 4. Production GCP Cloud Run fallback
        return 'https://samo-unified-api-frrnetyhfa-uc.a.run.app';
    }

    async fetchServerConfig() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout for config
            
            const response = await fetch('/api/config', {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                },
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`Server config fetch failed: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                console.warn('Server configuration timeout after 5s');
            } else {
                console.warn('Server configuration not available:', error.message);
            }
            return null;
        }
    }

    getAPIKey() {
        if (typeof SAMO_CONFIG !== 'undefined' && SAMO_CONFIG.apiKey) {
            return SAMO_CONFIG.apiKey;
        }
        if (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_KEY) {
            return process.env.REACT_APP_API_KEY;
        }
        return null; // No API key required for current service
    }

    getTimeout() {
        if (typeof SAMO_CONFIG !== 'undefined' && SAMO_CONFIG.timeout) {
            return parseInt(SAMO_CONFIG.timeout, 10);
        }
        return 30000; // 30 seconds default
    }

    getRetryAttempts() {
        if (typeof SAMO_CONFIG !== 'undefined' && SAMO_CONFIG.retryAttempts) {
            return parseInt(SAMO_CONFIG.retryAttempts, 10);
        }
        return 3; // 3 retry attempts default
    }

    validateConfig(config) {
        // Validate baseURL
        if (!config.baseURL || typeof config.baseURL !== 'string') {
            throw new Error('Invalid API base URL configuration');
        }

        // Check for hardcoded production URLs in client code
        if (config.baseURL.includes('samo-unified-api') && config.baseURL.includes('.run.app')) {
            console.warn('Warning: Using hardcoded production URL in client code. Consider using environment configuration.');
        }

        // Validate timeout
        if (config.timeout < 1000 || config.timeout > 120000) {
            console.warn('API timeout should be between 1-120 seconds, using default');
            config.timeout = 30000;
        }

        // Validate retry attempts
        if (config.retryAttempts < 0 || config.retryAttempts > 10) {
            console.warn('Retry attempts should be between 0-10, using default');
            config.retryAttempts = 3;
        }
    }

    async waitForInitialization() {
        const maxWaitTime = 5000; // 5 seconds
        const checkInterval = 100; // 100ms
        let waited = 0;

        while (!this.initialized && waited < maxWaitTime) {
            await new Promise(resolve => setTimeout(resolve, checkInterval));
            waited += checkInterval;
        }

        if (!this.initialized) {
            throw new Error('API client initialization timeout');
        }
    }

    async makeRequest(endpoint, data, method = 'POST', retryAttempt = 0) {
        // Wait for initialization to complete
        if (!this.initialized) {
            await this.waitForInitialization();
        }

        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'SAMO-Demo-Website/1.0'
            }
        };

        if (this.apiKey) {
            config.headers['X-API-Key'] = this.apiKey;
        }

        if (data && method === 'POST') {
            config.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, config);

            if (!response.ok) {
                if (response.status === 429) {
                    const errorData = await response.json().catch(() => ({}));

                    // If we have retry attempts left and a retry_after value
                    if (retryAttempt < this.retryAttempts && errorData.retry_after) {
                        const delayMs = Math.min(errorData.retry_after * 1000, 5000); // Max 5 seconds
                        console.log(`Rate limited, retrying in ${delayMs}ms (attempt ${retryAttempt + 1}/${this.retryAttempts})`);

                        await new Promise(resolve => setTimeout(resolve, delayMs));
                        return this.makeRequest(endpoint, data, method, retryAttempt + 1);
                    }

                    throw new Error(errorData.message || 'Rate limit exceeded. Using demo data instead.');
                } else if (response.status === 401) {
                    throw new Error('API authentication required for this endpoint.');
                } else if (response.status === 403) {
                    throw new Error('API access forbidden. Using demo data instead.');
                } else if (response.status === 503) {
                    throw new Error('Service temporarily unavailable. Using demo data instead.');
                }
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async transcribeAudio(audioFile) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.timeout);
            
            const formData = new FormData();
            formData.append('audio', audioFile);

            const response = await fetch(`${this.baseURL}/transcribe/voice`, {
                method: 'POST',
                body: formData,
                headers: this.apiKey ? { 'X-API-Key': this.apiKey } : {},
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `Transcription failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error(`Transcription timeout after ${this.timeout}ms`);
            }
            console.error('Transcription error:', error);
            throw error;
        }
    }

    async summarizeText(text) {
        try {
            const response = await this.makeRequest('/summarize/text', { text });

            // Handle the actual API response format
            if (response.summary) {
                return {
                    summary: response.summary.summary || response.summary,
                    original_length: response.insights?.text_length || text.length,
                    summary_length: response.summary?.summary?.length || response.summary.length || 0,
                    compression_ratio: response.summary.compression_ratio || 0.5,
                    request_id: 'api-' + Date.now(),
                    timestamp: Date.now() / 1000,
                    mock: false
                };
            }

            return response;
        } catch (error) {
            // If API is not available, return mock data for demo purposes
            console.warn('API not available, using mock data for demo:', error.message);
            return this.getMockSummaryResponse(text);
        }
    }

    getMockSummaryResponse(text) {
        // Extract sentences for analysis
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        
        // Convert to third-person narrative summary
        let summary = '';
        
        // Analyze the overall sentiment and key themes
        const hasExcitement = /excited|happy|thrilled|wonderful|amazing|optimistic/i.test(text);
        const hasConcerns = /nervous|challenges|difficulties|concerned/i.test(text);
        const hasConfidence = /confident|believe|can overcome|together/i.test(text);
        const hasFuture = /future|opportunities|ahead|await/i.test(text);
        
        // Build third-person summary based on detected themes
        const summaryParts = [];
        
        if (hasExcitement) {
            summaryParts.push('The individual expresses strong positive emotions and enthusiasm');
        }
        
        if (hasFuture) {
            summaryParts.push('about upcoming opportunities and future prospects');
        }
        
        if (hasConcerns) {
            summaryParts.push('while acknowledging some apprehension about potential challenges');
        }
        
        if (hasConfidence) {
            summaryParts.push('but maintains confidence in their ability to work through difficulties collaboratively');
        }
        
        // Create the final summary
        if (summaryParts.length > 0) {
            summary = summaryParts.join(' ') + '.';
        } else {
            // Fallback for generic text
            summary = 'The text expresses a mix of emotions and thoughts about current circumstances and future outlook.';
        }
        
        // Ensure summary is not too long
        if (summary.length > text.length * 0.6) {
            summary = 'The individual shares their emotional state and perspective on current and future situations.';
        }
        
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
            const response = await this.makeRequest('/analyze/journal', { text });

            // Handle the actual API response format
            if (response.emotion_analysis && response.emotion_analysis.emotions) {
                const emotions = response.emotion_analysis.emotions;
                const emotionArray = Object.entries(emotions).map(([emotion, confidence]) => ({
                    emotion: emotion,
                    confidence: confidence
                }));

                return {
                    emotions: emotionArray,
                    confidence: response.emotion_analysis.confidence || 0,
                    primary_emotion: response.emotion_analysis.primary_emotion,
                    emotional_intensity: response.emotion_analysis.emotional_intensity,
                    processing_time_ms: response.processing_time_ms,
                    text: text,
                    mock: false
                };
            }

            return response;
        } catch (error) {
            // If API is not available, return mock data for demo purposes
            if (error.message.includes('Rate limit') || error.message.includes('API key') || error.message.includes('Service temporarily') || error.message.includes('Abuse detected') || error.message.includes('Client blocked')) {
                console.warn('API not available, using mock data for demo:', error.message);
                return this.getMockEmotionResponse(text);
            }
            console.warn('Unknown error, using mock data for demo:', error.message);
            return this.getMockEmotionResponse(text);
        }
    }

    getMockEmotionResponse(text) {
        // Mock emotion detection response for demo purposes
        const emotions = [
            { emotion: 'joy', confidence: 0.85 },
            { emotion: 'excitement', confidence: 0.72 },
            { emotion: 'optimism', confidence: 0.68 },
            { emotion: 'gratitude', confidence: 0.45 },
            { emotion: 'neutral', confidence: 0.15 }
        ];
        
        return {
            text: text,
            emotions: emotions,
            confidence: 0.75,
            request_id: 'demo-' + Date.now(),
            timestamp: Date.now() / 1000,
            mock: true
        };
    }
}
