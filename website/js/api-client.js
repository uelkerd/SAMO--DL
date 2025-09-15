/**
 * API Client Module
 * Handles all API communication for the SAMO Demo
 */
class SAMOAPIClient {
    constructor() {
        // Use configuration from config.js if available, otherwise fallback to demo mode
        this.baseURL = (typeof SAMO_CONFIG !== 'undefined') ? SAMO_CONFIG.baseURL : 'https://samo-unified-api-frrnetyhfa-uc.a.run.app';
        this.apiKey = (typeof SAMO_CONFIG !== 'undefined') ? SAMO_CONFIG.apiKey : null;
        this.timeout = (typeof SAMO_CONFIG !== 'undefined') ? SAMO_CONFIG.timeout : 30000;
        this.retryAttempts = (typeof SAMO_CONFIG !== 'undefined') ? SAMO_CONFIG.retryAttempts : 3;
    }

    async makeRequest(endpoint, data, method = 'POST') {
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
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
                    throw new Error(errorData.message || 'Rate limit exceeded. Please try again in a moment.');
                } else if (response.status === 401) {
                    throw new Error('API key required. Please contact support for access.');
                } else if (response.status === 503) {
                    throw new Error('Service temporarily unavailable. Please try again later.');
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
            const formData = new FormData();
            formData.append('audio', audioFile);

            const response = await fetch(`${this.baseURL}/transcribe/voice`, {
                method: 'POST',
                body: formData,
                headers: this.apiKey ? { 'X-API-Key': this.apiKey } : {}
            });

            if (!response.ok) {
                throw new Error(`Transcription failed: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Transcription error:', error);
            throw error;
        }
    }

    async summarizeText(text) {
        try {
            return await this.makeRequest('/summarize/text', { text });
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
            return await this.makeRequest('/analyze/journal', { text });
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
