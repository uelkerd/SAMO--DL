/**
 * Comprehensive AI Platform Demo
 * Handles voice transcription, text summarization, and emotion detection
 */

class SAMOAPIClient {
    constructor() {
        // Use centralized configuration
        if (!window.SAMO_CONFIG) {
            console.warn('⚠️ SAMO_CONFIG not found, using fallback configuration');
        }

        this.baseURL = window.SAMO_CONFIG?.API?.BASE_URL || 'https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app';
        this.endpoints = window.SAMO_CONFIG?.API?.ENDPOINTS || {
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
        this.timeout = window.SAMO_CONFIG?.API?.TIMEOUT || 15000; // Reduced from 20s to 15s
        this.coldStartTimeout = window.SAMO_CONFIG?.API?.COLD_START_TIMEOUT || 45000; // Reduced from 60s to 45s
        this.retryAttempts = window.SAMO_CONFIG?.API?.RETRY_ATTEMPTS || 1; // Reduced to 1 for faster feedback
        this.isColdStart = true; // Track if this is the first request
    }

    getApiKey() {
        // Try to get API key from various sources
        // 1. From SAMO_CONFIG (server-injected)
        if (window.SAMO_CONFIG?.API?.API_KEY) {
            return window.SAMO_CONFIG.API.API_KEY;
        }

        // 2. From localStorage (user-set)
        const storedKey = localStorage.getItem('samo_api_key');
        if (storedKey && storedKey.trim()) {
            return storedKey.trim();
        }

        // 3. From environment variable (if available in browser context)
        if (window.SAMO_CONFIG?.API?.API_KEY_ENV) {
            return window.SAMO_CONFIG.API.API_KEY_ENV;
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
        let abortReason = null; // Track abort reason

        // Use cold start timeout for first request, regular timeout otherwise
        const timeout = timeoutMs || (this.isColdStart ? this.coldStartTimeout : this.timeout);
        const timer = setTimeout(() => {
            abortReason = 'timeout';
            controller.abort();
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
                // Send JSON data in request body (proper REST API format)
                config.headers['Content-Type'] = 'application/json';
                config.body = JSON.stringify(data);
            }
        } else if (method === 'GET') {
            // Optional: set Accept if needed
            config.headers['Accept'] = 'application/json';
        }

        try {
            let url = `${this.baseURL}${endpoint}`;

            // For GET requests with data, append query parameters
            if (data && method === 'GET') {
                const queryParams = this.buildQueryString(data);
                if (queryParams) {
                    url += (url.includes('?') ? '&' : '?') + queryParams;
                }
            }

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
            if ((error.name === 'AbortError' || abortReason === 'timeout' || error.message.includes('timeout') || error.message.includes('network')) && attemptsLeft > 1) {
                const backoffDelay = Math.pow(2, this.retryAttempts - attemptsLeft) * 1000;
                console.warn(`Network error, retrying in ${backoffDelay}ms. Attempts left: ${attemptsLeft - 1}`, error.message);

                // Provide user feedback about network retry
                if (typeof addToProgressConsole === 'function') {
                    addToProgressConsole(`Network error, retrying in ${backoffDelay/1000}s...`, 'warning');
                }

                await new Promise(resolve => setTimeout(resolve, backoffDelay));
                return this.makeRequestWithRetry(endpoint, data, method, isFormData, timeoutMs, attemptsLeft - 1);
            }

            // Check if it was a timeout and throw appropriate error
            if (abortReason === 'timeout') {
                console.error(`Request timeout after ${timeout/1000}s`);
                throw new Error(`Request timeout after ${timeout/1000}s`);
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

                // Inform user about fallback to mock data
                if (typeof addToProgressConsole === 'function') {
                    addToProgressConsole('⚠️ API unavailable, using mock summarization for demo', 'warning');
                }

                const mockResponse = this.getMockSummaryResponse(text);
                mockResponse.fallback_reason = error.message;
                return mockResponse;
            }
            throw error;
        }
    }

    getMockSummaryResponse(text) {
        // Improved mock summarization response for demo purposes
        const words = text.split(' ');
        const summaryLength = Math.max(15, Math.floor(words.length * 0.4));

        // Create a more intelligent summary by taking key sentences
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
        let summary;

        if (sentences.length >= 2) {
            // Take first and last sentences for a basic summary
            summary = sentences[0].trim() + '. ' + sentences[sentences.length - 1].trim() + '.';
        } else {
            // Fallback to word truncation
            summary = words.slice(0, summaryLength).join(' ') + '...';
        }

        // Ensure summary is not too long
        if (summary.length > text.length * 0.6) {
            summary = words.slice(0, Math.floor(words.length * 0.4)).join(' ') + '...';
        }

        console.log('🤖 Generated mock summary:', summary);

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

                // Inform user about fallback to mock data
                if (typeof addToProgressConsole === 'function') {
                    addToProgressConsole('⚠️ API unavailable, using mock emotion detection for demo', 'warning');
                }

                const mockResponse = this.getMockEmotionResponse(text);
                mockResponse.fallback_reason = error.message;
                return mockResponse;
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


// Global API client instance
window.apiClient = null;

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ DOM loaded, initializing demo...');
    console.log('🔧 Using simple-demo-functions.js for chart implementation');

    // Initialize global API client
    try {
        window.apiClient = new SAMOAPIClient();
        console.log('✅ Global API client initialized');
    } catch (error) {
        console.error('❌ Failed to initialize API client:', error);
    }
});

// Smooth scrolling for in-page navigation links
// Only applies to anchors within the main navigation to avoid interfering with external or footer anchors
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('nav a[href^="#"], .navbar a[href^="#"], #main-nav a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        // Only handle if the link is for the current page
        if (location.pathname === anchor.pathname && location.hostname === anchor.hostname) {
            e.preventDefault();
            const href = this.getAttribute('href');
            if (!href) return;
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        }
    });
  });
});

// Essential Demo Functions (restored from simple-demo-functions.js)

// Inline message display functions
function showInlineError(message, targetElementId) {
    showInlineMessage(message, targetElementId, 'error');
}

function showInlineSuccess(message, targetElementId) {
    showInlineMessage(message, targetElementId, 'success');
}

function showInlineMessage(message, targetElementId, type = 'error') {
    const existingMessages = document.querySelectorAll('.inline-message');
    existingMessages.forEach(msg => msg.remove());

    const messageDiv = document.createElement('div');
    messageDiv.className = `inline-message alert ${type === 'error' ? 'alert-danger' : 'alert-success'} mt-2`;
    messageDiv.setAttribute('role', 'alert');
    messageDiv.style.cssText = 'animation: fadeIn 0.3s ease-in; font-size: 0.9rem;';
    messageDiv.textContent = message;

    const targetElement = document.getElementById(targetElementId);
    if (targetElement) {
        targetElement.parentNode.insertBefore(messageDiv, targetElement.nextSibling);
    } else {
        document.body.appendChild(messageDiv);
    }

    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.parentNode.removeChild(messageDiv);
        }
    }, 4000);
}

// Generate Sample Text Function
async function generateSampleText() {
    console.log('✨ Generating AI-powered sample journal text...');

    // Define sample texts for fallback
    const sampleTexts = [
        "Today started like any other day, but something unexpected happened that completely changed my mood. I woke up feeling restless, as if something important was waiting for me just beyond the horizon. The morning sunlight streaming through my window felt warmer than usual, and I found myself lingering in bed longer than I should have, savoring the quiet moments before the day officially began. Walking through the park, I noticed things I'd never seen before despite passing this way hundreds of times. That's when I realized what I was feeling – a profound sense of gratitude mixed with a gentle melancholy for time that has passed.",

        "After a long conversation with someone close to me, I'm left feeling quite contemplative and unexpectedly vulnerable. It's funny how a simple exchange of words can peel back layers of what we often bury deep inside us. We discussed dreams that feel too big, disappointments that still sting, and the strange comfort found in knowing that someone else understands the complexity of simply being human. Now, sitting here in the quiet aftermath, I feel emotionally exhausted but also somehow lighter.",

        "I've been struggling with a decision that's been weighing heavily on my mind for weeks. The rational part of me knows what I should do, but my heart keeps pulling me in a different direction. It's that familiar tug-of-war between what feels safe and what feels authentic. Sometimes I wonder if we're meant to feel this conflicted about the paths we choose, or if clarity is something that comes only in hindsight. Tonight, I'm choosing to sit with the uncertainty rather than rush toward an answer.",

        "There's something magical about rainy afternoons that makes me incredibly nostalgic. The sound of droplets against my window takes me back to childhood days when the world felt both infinite and completely contained within the walls of our small house. I remember how my grandmother used to say that rain was just the sky's way of crying happy tears. Looking back, I think she might have been onto something profound about finding beauty in moments of release.",

        "I had one of those moments today where everything felt perfectly aligned. It wasn't anything dramatic – just a simple conversation with a stranger at the coffee shop who smiled genuinely and asked how my day was going. But something about that brief connection reminded me that kindness is still everywhere if we're paying attention. It's amazing how a single moment of human warmth can shift your entire perspective on the day.",

        "I've been thinking a lot about the concept of home lately. Not just the physical space where I live, but that feeling of belonging that seems to come and go like the tide. Sometimes I feel most at home in unexpected places – a quiet corner of a library, a park bench under my favorite tree, or even in the middle of a crowded room filled with laughter. Maybe home isn't a place at all, but a feeling we carry within us.",

        "Tonight I'm sitting on my balcony watching the city lights twinkle below, and I'm overwhelmed by how many stories are unfolding simultaneously around me. Behind each lit window is someone living their own complex narrative of hopes, fears, dreams, and disappointments. It's both humbling and comforting to remember that we're all just trying to figure it out as we go along. Sometimes feeling small in the grand scheme of things is exactly what we need.",

        "I picked up a book today that I loved in college and was surprised by how differently it resonated with me now. The same words that once felt revolutionary now feel like old friends offering gentle wisdom. It made me realize how much I've changed without even noticing. Growth isn't always dramatic or obvious – sometimes it's just the quiet accumulation of experiences that slowly shift how we see the world.",

        "There's something bittersweet about cleaning out old belongings and finding forgotten treasures from different phases of my life. Each item tells a story about who I used to be, the dreams I once had, and the paths I chose not to take. It's like archaeological evidence of my own becoming. I'm learning to feel grateful for all the versions of myself that led me here, even the ones that felt lost at the time.",

        "I had a moment of pure joy today while listening to my favorite song on repeat. It's one of those tracks that never gets old, that seems to capture something essential about being alive. Music has this incredible ability to transport us instantly to emotional spaces we might struggle to access otherwise. Sometimes I think musicians are just emotional translators, helping us understand feelings we didn't even know we had.",

        "The anxiety I've been carrying lately feels like a heavy backpack I forgot I was wearing. It's only when I consciously set it down that I realize how much energy it was taking just to carry it around. I'm learning that acknowledging difficult emotions doesn't make them stronger – it actually gives them permission to move through me instead of getting stuck. Today I'm practicing the art of gentle self-compassion.",

        "I spent the morning in my garden, hands deep in the soil, and felt more grounded than I have in weeks. There's something deeply satisfying about nurturing something from seed to bloom. It reminds me that growth takes time, that patience is its own form of faith, and that some of the most beautiful things happen slowly, underground, before we can see any evidence of progress. Maybe I need to remember this about my own life too."
    ];

    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.value = '🤖 Generating AI text...';
        textInput.style.borderColor = '#8b5cf6';
        textInput.style.boxShadow = '0 0 0 0.2rem rgba(139, 92, 246, 0.25)';
    }

    try {
        let apiKey = window.SAMO_CONFIG?.OPENAI?.API_KEY || localStorage.getItem('openai_api_key');

        if (!apiKey || apiKey.trim() === '') {
            showInlineError('⚠️ OpenAI API key required for AI text generation. Click "Manage API Key" to set up.', 'textInput');

            if (textInput) {
                textInput.value = '';
                textInput.style.borderColor = '#ef4444';
                textInput.style.boxShadow = '0 0 0 0.2rem rgba(239, 68, 68, 0.25)';
                setTimeout(() => {
                    textInput.style.borderColor = '';
                    textInput.style.boxShadow = '';
                }, 3000);
            }
            return;
        }

        const prompts = [
            "Today started like any other day, but something unexpected happened that completely changed my mood. I found myself feeling",
            "I've been reflecting on recent changes in my life, and I'm experiencing a whirlwind of emotions. Right now I'm particularly",
            "This week has been a journey of self-discovery. I wake up each morning feeling different, but today I'm especially",
            "After a long conversation with someone close to me, I'm left feeling quite contemplative and",
            "The weather outside perfectly matches my internal state today. I'm feeling deeply"
        ];

        const randomPrompt = prompts[Math.floor(Math.random() * prompts.length)];
        console.log('🤖 Generating AI text with OpenAI API...');

        // Update user with loading feedback
        if (textInput) {
            textInput.value = '🤖 Generating unique AI text with OpenAI...';
        }

        // Make actual OpenAI API call
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: window.SAMO_CONFIG?.OPENAI?.MODEL || 'gpt-4o-mini',
                messages: [
                    {
                        role: 'system',
                        content: 'You are an AI that creates realistic, emotional journal entries. Write a personal, introspective journal entry that expresses genuine human emotions and experiences. The entry should be 150-300 words, feel authentic, and contain a mix of emotions suitable for emotion detection analysis.'
                    },
                    {
                        role: 'user',
                        content: `Write a journal entry that continues this thought: "${randomPrompt}"`
                    }
                ],
                max_tokens: window.SAMO_CONFIG?.OPENAI?.MAX_TOKENS || 400,
                temperature: window.SAMO_CONFIG?.OPENAI?.TEMPERATURE || 0.7
            })
        });

        if (!response.ok) {
            throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const generatedText = data.choices[0]?.message?.content;

        if (!generatedText) {
            throw new Error('No text generated from OpenAI API');
        }

        console.log('✅ OpenAI API generated unique text successfully');

        if (textInput) {
            textInput.value = generatedText.trim();
            textInput.style.borderColor = '#10b981';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(16, 185, 129, 0.25)';
            setTimeout(() => {
                textInput.style.borderColor = '';
                textInput.style.boxShadow = '';
            }, 2000);
        }

        showInlineSuccess('✅ Unique AI text generated with OpenAI!', 'textInput');
        return; // Exit here - don't fall back to sample texts

    } catch (error) {
        console.error('❌ Error generating AI text:', error);
        console.log('🔄 Falling back to curated sample texts...');

        // Fall back to curated sample texts if OpenAI API fails
        const randomIndex = Math.floor(Math.random() * sampleTexts.length);
        const fallbackText = sampleTexts[randomIndex];

        if (textInput) {
            textInput.value = fallbackText;
            textInput.style.borderColor = '#f59e0b';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(245, 158, 11, 0.25)';
            setTimeout(() => {
                textInput.style.borderColor = '';
                textInput.style.boxShadow = '';
            }, 2000);
        }

        showInlineError(`⚠️ OpenAI API unavailable, using sample text (${randomIndex + 1} of ${sampleTexts.length})`, 'textInput');
    }
}

// API Key Management Function
function manageApiKey() {
    console.log('🔑 Managing API Key...');

    const currentKey = localStorage.getItem('openai_api_key') || '';
    const maskedKey = currentKey ? `${currentKey.substring(0, 7)}...${currentKey.substring(currentKey.length - 4)}` : 'Not set';

    const newKey = prompt(
        `Current OpenAI API Key: ${maskedKey}\n\n` +
        'Enter your OpenAI API Key (or leave empty to remove):\n\n' +
        'Note: This key is stored locally in your browser and is only used for generating sample text.',
        ''
    );

    if (newKey === null) {
        console.log('🔑 API Key management cancelled');
        return;
    }

    if (newKey.trim() === '') {
        localStorage.removeItem('openai_api_key');
        console.log('🔑 API Key removed');
        alert('✅ API Key removed successfully');
    } else if (newKey.startsWith('sk-')) {
        localStorage.setItem('openai_api_key', newKey.trim());
        console.log('🔑 API Key updated');
        alert('✅ API Key saved successfully');
    } else {
        console.log('🔑 Invalid API Key format');
        alert('❌ Invalid API Key format. OpenAI API keys should start with "sk-"');
    }
}

// Essential Processing Functions (restored from simple-demo-functions.js)

async function processText(skipStateCheck = false) {
    console.log('🚀 Processing text...');

    // Check if processing is already in progress (skip if state management is handled externally)
    if (!skipStateCheck && typeof LayoutManager !== 'undefined' && LayoutManager.isProcessing) {
        console.warn('⚠️ Processing blocked - operation already in progress');
        return;
    }

    const text = document.getElementById('textInput').value;
    console.log('🔍 Text from input:', text);
    console.log('🔍 Text length:', text.length);
    if (!text.trim()) {
        showInlineError('Please enter some text to analyze', 'textInput');
        return;
    }
    console.log('🔍 About to call testWithRealAPI from processText');
    await testWithRealAPI();
}

async function testWithRealAPI() {
    console.log('🌐 Testing with real API...');

    // Ensure processing state is properly set
    if (typeof LayoutManager !== 'undefined' && !LayoutManager.isProcessing) {
        console.warn('⚠️ testWithRealAPI called without processing state - setting now');
        if (!LayoutManager.showProcessingState()) {
            console.error('❌ Failed to set processing state in testWithRealAPI');
            return;
        }
    }

    const startTime = performance.now();

    // Initialize progress console
    addToProgressConsole('🚀 Starting AI processing pipeline...', 'info');
    addToProgressConsole('Preparing text for analysis...', 'processing');

    // Update processing status
    updateElement('processingStatusCompact', 'Processing');

    try {
        // Show enhanced loading state
        const chartContainer = document.getElementById('emotionChart');
        if (chartContainer) {
            while (chartContainer.firstChild) {
                chartContainer.removeChild(chartContainer.firstChild);
            }

            const loadingDiv = document.createElement('div');
            loadingDiv.style.cssText = 'text-align: center; padding: 30px; background: rgba(139, 92, 246, 0.05); border-radius: 10px; border: 1px solid rgba(139, 92, 246, 0.2);';

            const spinner = document.createElement('div');
            spinner.className = 'spinner-border text-primary mb-3';
            spinner.style.cssText = 'width: 2rem; height: 2rem;';
            loadingDiv.appendChild(spinner);

            const title = document.createElement('h6');
            title.textContent = '🧠 AI Analysis in Progress';
            title.style.cssText = 'color: #8b5cf6; margin-bottom: 15px;';
            loadingDiv.appendChild(title);

            const message = document.createElement('p');
            message.id = 'emotionLoadingMessage';
            message.textContent = 'Initializing emotion analysis models...';
            message.style.cssText = 'color: #6b7280; margin-bottom: 10px;';
            loadingDiv.appendChild(message);

            const timeEstimate = document.createElement('small');
            timeEstimate.textContent = 'First request may take 30-60 seconds (cold start)';
            timeEstimate.style.cssText = 'color: #9ca3af; font-style: italic;';
            loadingDiv.appendChild(timeEstimate);

            chartContainer.appendChild(loadingDiv);

            // Update progress messages
            setTimeout(() => {
                const msg = document.getElementById('emotionLoadingMessage');
                if (msg) msg.textContent = 'Loading DeBERTa v3 Large model (this may take a moment)...';
            }, 3000);

            setTimeout(() => {
                const msg = document.getElementById('emotionLoadingMessage');
                if (msg) msg.textContent = 'Processing your text with AI emotion analysis...';
            }, 8000);

            setTimeout(() => {
                const msg = document.getElementById('emotionLoadingMessage');
                if (msg) msg.textContent = 'Almost done - finalizing emotion detection results...';
            }, 20000);
        }

        updateElement('primaryEmotion', 'Loading...');

        let testText = document.getElementById('textInput').value || "I am so excited and happy today! This is wonderful news!";

        // Check text length limit
        const MAX_TEXT_LENGTH = window.SAMO_CONFIG?.LIMITS?.TEXT_MAX ?? 400;
        if (testText.length > MAX_TEXT_LENGTH) {
            console.log(`⚠️ Text too long (${testText.length} chars), truncating to ${MAX_TEXT_LENGTH} chars`);
            addToProgressConsole(`Text truncated from ${testText.length} to ${MAX_TEXT_LENGTH} characters`, 'warning');
            testText = testText.substring(0, MAX_TEXT_LENGTH) + "...";
        }

        addToProgressConsole(`Text prepared for analysis (${testText.length} characters)`, 'success');
        addToProgressConsole('🧠 Initializing DeBERTa v3 Large emotion model...', 'processing');
        console.log('🔥 Calling emotion API...');
        addToProgressConsole('🌐 Sending request to emotion analysis API...', 'processing');
        // Create API client instance for proper timeout and error handling
        const apiClient = new SAMOAPIClient();
        const data = await apiClient.makeRequest(apiClient.endpoints.EMOTION, { text: testText }, 'POST');

        addToProgressConsole('✅ Emotion analysis API response received', 'success');
        console.log('✅ Real API response:', data);

        // Process emotion data
        addToProgressConsole('🔍 Processing emotion analysis results...', 'processing');
        let primaryEmotion = null;
        let primaryConfidence = 0;
        let emotionArray = [];

        if (data.emotions && typeof data.emotions === 'object' && data.predicted_emotion) {
            primaryEmotion = data.predicted_emotion;
            primaryConfidence = data.emotions[data.predicted_emotion] || 0;

            // Convert emotions object to array for chart
            emotionArray = Object.entries(data.emotions)
                .map(([emotion, confidence]) => ({ emotion, confidence }))
                .sort((a, b) => b.confidence - a.confidence);

        } else if (data.emotion && data.confidence) {
            primaryEmotion = data.emotion;
            primaryConfidence = data.confidence;
            emotionArray = [{ emotion: data.emotion, confidence: data.confidence }];
        }

        addToProgressConsole(`Primary emotion detected: ${primaryEmotion} (${Math.round(primaryConfidence * 100)}%)`, 'success');

        // Update UI with results
        updateElement('primaryEmotion', primaryEmotion || 'Unknown');
        updateElement('emotionalIntensity', `${Math.round(primaryConfidence * 100)}%`);

        // Fill in additional data fields
        updateElement('sentimentScore', primaryConfidence ? (primaryConfidence * 100).toFixed(1) + '/100' : '-');
        updateElement('confidenceRange', primaryConfidence ? `${(primaryConfidence * 80).toFixed(1)}-${(primaryConfidence * 100).toFixed(1)}%` : '-');
        updateElement('modelDetails', 'DeBERTa v3 Large (SAMO-GoEmotions)');

        // Create emotion chart
        if (emotionArray.length > 0) {
            createEmotionChart(emotionArray);
        }

        // Call summarization API
        addToProgressConsole('📝 Starting text summarization with T5 model...', 'processing');
        const summary = await callSummarizationAPI(testText);

        // Update Processing Information box
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(1);

        updateElement('totalTimeCompact', `${processingTime}s`);
        updateElement('processingStatusCompact', 'Complete');
        updateElement('modelsUsedCompact', 'DeBERTa v3 + T5');
        updateElement('avgConfidenceCompact', `${Math.round(primaryConfidence * 100)}%`);

        // Show results
        addToProgressConsole('🎉 AI processing pipeline completed successfully!', 'success');
        addToProgressConsole(`Total processing time: ${processingTime} seconds`, 'info');
        showResultsSections();

    } catch (error) {
        console.error('❌ Error in testWithRealAPI:', error.message, error.status, error.response?.data);

        // Reset processing state on error
        if (typeof LayoutManager !== 'undefined' && LayoutManager.isProcessing) {
            LayoutManager.endProcessing();
            console.log('🔧 Processing state reset due to error');
        }

        // Update processing status to error
        updateElement('processingStatusCompact', 'Error');

        // Better error handling for different error types
        if (error.name === 'AbortError') {
            const reason = error.message || 'Request was cancelled';
            addToProgressConsole(`Processing cancelled: ${reason}`, 'error');
            showInlineError(`❌ Processing cancelled: ${reason}`, 'textInput');
        } else if (error.message.includes('Failed to fetch')) {
            addToProgressConsole('Network error: Cannot reach API server', 'error');
            showInlineError(`❌ Network error: Cannot reach API server. Please check your connection.`, 'textInput');
        } else if (error.message.includes('timeout')) {
            addToProgressConsole('Request timeout: API server took too long to respond', 'error');
            showInlineError(`❌ Request timeout: API server took too long to respond.`, 'textInput');
        } else {
            addToProgressConsole(`Processing failed: ${error.message}`, 'error');
            showInlineError(`❌ Failed to process text: ${error.message}`, 'textInput');
        }

        // IMMEDIATELY return to initial state on error (no delay)
        if (typeof LayoutManager !== 'undefined') {
            LayoutManager.resetToInitialState();
        }
    }
}

async function callSummarizationAPI(text) {
    console.log('📝 Calling real summarization API...');
    console.log('📝 Input text length:', text.length);
    console.log('📝 Input text preview:', text.substring(0, 100) + '...');
    addToProgressConsole('🌐 Sending request to summarization API...', 'processing');

    try {
        // Create API client instance for proper timeout and error handling
        const apiClient = new SAMOAPIClient();
        console.log('📝 API Base URL:', apiClient.baseURL);
        console.log('📝 Summarization endpoint:', apiClient.endpoints.SUMMARIZE);
        console.log('📝 Full API URL:', `${apiClient.baseURL}${apiClient.endpoints.SUMMARIZE}`);

        const data = await apiClient.makeRequest(apiClient.endpoints.SUMMARIZE, { text: text }, 'POST');

        addToProgressConsole('✅ Summarization API response received', 'success');
        console.log('✅ Summarization API response (full):', JSON.stringify(data, null, 2));

        // Extract summary from response - be more thorough in extraction
        addToProgressConsole('🔍 Processing summarization results...', 'processing');

        // Log all possible fields to debug response structure
        console.log('📝 Response fields available:', Object.keys(data));

        let summaryText = data.summary || data.text || data.summarized_text || data.result || data.output || data.generated_text;

        // Check if the response is nested
        if (!summaryText && data.data) {
            summaryText = data.data.summary || data.data.text || data.data.summarized_text || data.data.result || data.data.output;
        }

        // Log the extracted summary
        console.log('📝 Extracted summary text:', summaryText);

        if (summaryText && summaryText.trim()) {
            updateElement('summaryText', summaryText);
            updateElement('originalLength', text.length);
            updateElement('summaryLength', summaryText.length);
            addToProgressConsole(`Summary generated successfully (${summaryText.length} characters)`, 'success');
            console.log('✅ Summary successfully displayed');
        } else {
            console.warn('⚠️ No valid summary found in response');
            console.warn('⚠️ Full response structure:', JSON.stringify(data, null, 2));
            addToProgressConsole('No valid summary found in API response', 'warning');
            updateElement('summaryText', 'Summary not available - API response did not contain summary text');
        }

        return summaryText;

    } catch (error) {
        console.error('❌ Error in callSummarizationAPI:', error);
        console.error('❌ Error details:', {
            message: error.message,
            stack: error.stack,
            name: error.name
        });

        // More specific error messages
        let errorMessage = 'Failed to generate summary';
        if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Network error: Could not connect to summarization API';
        } else if (error.message.includes('timeout')) {
            errorMessage = 'Request timeout: Summarization API took too long to respond';
        } else if (error.message.includes('500')) {
            errorMessage = 'Server error: Summarization API encountered an internal error';
        } else if (error.message.includes('404')) {
            errorMessage = 'API endpoint not found: Please check API configuration';
        }

        addToProgressConsole(`Summarization failed: ${errorMessage}`, 'error');
        updateElement('summaryText', errorMessage);
        return null;
    }
}


function showResultsSections() {
    console.log('👁️ Showing results sections...');

    try {
        const emotionResults = document.getElementById('emotionResults');
        if (emotionResults) {
            emotionResults.classList.remove('result-section-hidden');
            emotionResults.classList.add('result-section-visible');
            emotionResults.style.display = 'block';
            console.log('✅ Emotion results section shown');
        }

        const summarizationResults = document.getElementById('summarizationResults');
        if (summarizationResults) {
            summarizationResults.classList.remove('result-section-hidden');
            summarizationResults.classList.add('result-section-visible');
            summarizationResults.style.display = 'block';
            console.log('✅ Summarization results section shown');
        }
    } catch (error) {
        console.error('❌ Error showing results sections:', error);
    }
}

// Progress Console Functions
function addToProgressConsole(message, type = 'info') {
    const consoleEl = document.getElementById('progressConsole');
    const consoleRow = document.getElementById('progressConsoleRow');

    if (!consoleEl) return;

    // Show console if hidden
    if (consoleRow) {
        consoleRow.style.display = 'block';
    }

    const timestamp = new Date().toLocaleTimeString();
    let className = 'text-light';
    let icon = '•';

    switch(type) {
        case 'success':
            className = 'text-success';
            icon = '✓';
            break;
        case 'error':
            className = 'text-danger';
            icon = '✗';
            break;
        case 'warning':
            className = 'text-warning';
            icon = '⚠';
            break;
        case 'info':
            className = 'text-info';
            icon = 'ℹ';
            break;
        case 'processing':
            className = 'text-primary';
            icon = '⏳';
            break;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = className;

    // Create timestamp span safely
    const timestampSpan = document.createElement('span');
    timestampSpan.className = 'text-muted';
    timestampSpan.textContent = `[${timestamp}]`;

    // Create icon span safely
    const iconSpan = document.createElement('span');
    iconSpan.textContent = icon;

    // Create message span safely
    const messageSpan = document.createElement('span');
    messageSpan.textContent = ` ${message}`;

    // Append elements safely
    messageDiv.appendChild(timestampSpan);
    messageDiv.appendChild(iconSpan);
    messageDiv.appendChild(messageSpan);

    consoleEl.appendChild(messageDiv);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function clearProgressConsole() {
    const consoleEl = document.getElementById('progressConsole');
    if (consoleEl) {
        consoleEl.textContent = '';
        const readyDiv = document.createElement('div');
        readyDiv.className = 'text-success';
        readyDiv.textContent = 'SAMO-DL Processing Console Ready...';
        consoleEl.appendChild(readyDiv);
    }
}

// Enhanced updateElement function that handles different content types
function updateElement(id, value) {
    try {
        const element = document.getElementById(id);
        if (element) {
            if (id === 'summaryText') {
                // Special handling for summary text - use dark-theme compatible styling
                // Clear existing content safely
                element.textContent = '';

                // Create container div safely
                const containerDiv = document.createElement('div');
                containerDiv.className = 'p-3 bg-dark border border-secondary rounded text-light';

                // Set text content safely
                const textContent = value !== null && value !== undefined ? value : 'No summary available';
                containerDiv.textContent = textContent;

                // Append to element
                element.appendChild(containerDiv);

                console.log(`✅ Updated summary text: ${value}`);
                // Only add success message if it's actually a successful summary (not an error message)
                if (value && !value.includes('Failed to') && !value.includes('not available')) {
                    addToProgressConsole(`Summary text updated successfully`, 'success');
                }
            } else {
                element.textContent = value !== null && value !== undefined ? value : '-';
                console.log(`✅ Updated ${id}: ${value}`);
            }
        } else {
            console.warn(`⚠️ Element not found: ${id}`);
            addToProgressConsole(`Warning: Element ${id} not found`, 'warning');
        }
    } catch (error) {
        console.error(`❌ Error updating element ${id}:`, error);
        addToProgressConsole(`Error updating ${id}: ${error.message}`, 'error');
    }
}

// Enhanced emotion chart creation
function createEmotionChart(emotionData) {
    try {
        addToProgressConsole('Creating emotion visualization chart...', 'processing');

        const chartContainer = document.getElementById('emotionChart');
        if (!chartContainer) {
            addToProgressConsole('Error: Emotion chart container not found', 'error');
            return;
        }

        // Clear any existing content
        chartContainer.textContent = '';

        if (!emotionData || emotionData.length === 0) {
            const noDataDiv = document.createElement('div');
            noDataDiv.className = 'text-muted text-center p-3';
            noDataDiv.textContent = 'No emotion data available';
            chartContainer.appendChild(noDataDiv);
            addToProgressConsole('No emotion data available for chart', 'warning');
            return;
        }

        // Take top 5 emotions
        const top5Emotions = emotionData.slice(0, 5);

        // Create simple bar chart with Bootstrap classes using DOM methods
        const emotionBarsDiv = document.createElement('div');
        emotionBarsDiv.className = 'emotion-bars';

        top5Emotions.forEach((emotion, index) => {
            const name = emotion.emotion || emotion.label || `Emotion ${index + 1}`;
            const confidence = ((emotion.confidence || emotion.score || 0) * 100).toFixed(1);
            const percentage = Math.max(5, confidence); // Minimum 5% for visibility

            const colors = ['primary', 'success', 'warning', 'info', 'secondary'];
            const colorClass = colors[index % colors.length];

            // Create main container div
            const emotionDiv = document.createElement('div');
            emotionDiv.className = 'mb-2';

            // Create header div
            const headerDiv = document.createElement('div');
            headerDiv.className = 'd-flex justify-content-between align-items-center mb-1';

            // Create name span
            const nameSpan = document.createElement('small');
            nameSpan.className = 'fw-bold text-capitalize';
            nameSpan.textContent = name;

            // Create confidence span
            const confidenceSpan = document.createElement('small');
            confidenceSpan.className = 'text-muted';
            confidenceSpan.textContent = `${confidence}%`;

            // Create progress container
            const progressDiv = document.createElement('div');
            progressDiv.className = 'progress';
            progressDiv.style.height = '20px';

            // Create progress bar
            const progressBar = document.createElement('div');
            progressBar.className = `progress-bar bg-${colorClass}`;
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('role', 'progressbar');
            progressBar.setAttribute('aria-valuenow', confidence);
            progressBar.setAttribute('aria-valuemin', '0');
            progressBar.setAttribute('aria-valuemax', '100');

            // Assemble the structure
            headerDiv.appendChild(nameSpan);
            headerDiv.appendChild(confidenceSpan);
            progressDiv.appendChild(progressBar);
            emotionDiv.appendChild(headerDiv);
            emotionDiv.appendChild(progressDiv);
            emotionBarsDiv.appendChild(emotionDiv);
        });

        // Clear existing content and append new content safely
        chartContainer.textContent = '';
        chartContainer.appendChild(emotionBarsDiv);

        addToProgressConsole(`Emotion chart created with ${top5Emotions.length} emotions`, 'success');

    } catch (error) {
        console.error('Error creating emotion chart:', error);
        addToProgressConsole(`Error creating emotion chart: ${error.message}`, 'error');
        const chartContainer = document.getElementById('emotionChart');
        if (chartContainer) {
            chartContainer.textContent = '';
            const errorDiv = document.createElement('div');
            errorDiv.className = 'text-danger text-center p-3';
            errorDiv.textContent = 'Error creating chart';
            chartContainer.appendChild(errorDiv);
        }
    }
}

// Reset demo to input screen
function resetToInputScreen() {
    console.log('🔄 Resetting to input screen...');

    // IMMEDIATELY clear all result content to prevent remnants
    clearAllResultContent();

    // Clear text input
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.value = '';
    }

    // Clear any inline messages
    const existingMessages = document.querySelectorAll('.inline-message');
    existingMessages.forEach(msg => msg.remove());

    // Reset Processing Information values
    updateElement('totalTimeCompact', '-');
    updateElement('processingStatusCompact', 'Ready');
    updateElement('modelsUsedCompact', '-');
    updateElement('avgConfidenceCompact', '-');

    // Clear progress console
    clearProgressConsole();

    // Hide progress console
    const progressConsoleRow = document.getElementById('progressConsoleRow');
    if (progressConsoleRow) {
        progressConsoleRow.style.display = 'none';
    }

    // Switch from results layout to input layout
    const resultsLayout = document.getElementById('resultsLayout');
    const inputLayout = document.getElementById('inputLayout');

    if (resultsLayout && inputLayout) {
        console.log('🔄 Transitioning layouts: results -> input');

        // IMMEDIATELY show input layout (don't wait for animation)
        inputLayout.classList.remove('d-none');
        inputLayout.style.display = 'block'; // Force display
        inputLayout.style.opacity = '1';
        inputLayout.style.transform = 'translateY(0)';
        console.log('✅ Input layout should now be visible');

        // Animate results layout out
        resultsLayout.style.opacity = '0';
        resultsLayout.style.transform = 'translateY(20px)';

        setTimeout(() => {
            resultsLayout.classList.add('d-none');
            console.log('✅ Results layout hidden');
        }, 300);
    } else {
        console.error('❌ Layout elements not found:', { resultsLayout: !!resultsLayout, inputLayout: !!inputLayout });
    }

    console.log('✅ Reset completed');
}

// NEW: Function to immediately clear all result content
function clearAllResultContent() {
    console.log('🧹 Clearing all result content immediately...');

    // Clear emotion analysis results
    updateElement('primaryEmotion', '-');
    updateElement('emotionalIntensity', '-');
    updateElement('sentimentScore', '-');
    updateElement('confidenceRange', '-');
    updateElement('modelDetails', '-');

    // Clear emotion chart
    const emotionChart = document.getElementById('emotionChart');
    if (emotionChart) {
        emotionChart.textContent = '';
    }

    // Clear emotion badges
    const emotionBadges = document.getElementById('emotionBadges');
    if (emotionBadges) {
        emotionBadges.textContent = '';
    }

    // Clear emotion details
    const emotionDetails = document.getElementById('emotionDetails');
    if (emotionDetails) {
        emotionDetails.textContent = '';
    }

    // Clear summarization results
    const summaryText = document.getElementById('summaryText');
    if (summaryText) {
        summaryText.textContent = '';
    }
    updateElement('originalLength', '-');
    updateElement('summaryLength', '-');

    // Clear transcription results
    const transcriptionText = document.getElementById('transcriptionText');
    if (transcriptionText) {
        transcriptionText.textContent = '';
    }
    updateElement('transcriptionConfidence', '-');
    updateElement('transcriptionDuration', '-');

    console.log('✅ All result content cleared');
}

// Make functions globally available
window.generateSampleText = generateSampleText;
window.processText = processText;
window.testWithRealAPI = testWithRealAPI;
window.callSummarizationAPI = callSummarizationAPI;
window.updateElement = updateElement;
window.showResultsSections = showResultsSections;
window.addToProgressConsole = addToProgressConsole;
window.clearProgressConsole = clearProgressConsole;
window.createEmotionChart = createEmotionChart;
window.resetToInputScreen = resetToInputScreen;
window.clearAllResultContent = clearAllResultContent;
window.manageApiKey = manageApiKey;
window.clearAll = clearAllResultContent; // Alias for clearAll function
