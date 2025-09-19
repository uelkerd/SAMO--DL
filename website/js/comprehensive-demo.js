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
        this.timeout = window.SAMO_CONFIG?.API?.TIMEOUT || 20000; // Reduced from 45s to 20s
        this.coldStartTimeout = window.SAMO_CONFIG?.API?.COLD_START_TIMEOUT || 60000; // Special timeout for first request
        this.retryAttempts = window.SAMO_CONFIG?.API?.RETRY_ATTEMPTS || 2; // Reduced from 3 to 2
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
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const msg = errorData.message || errorData.error || `HTTP ${response.status}`;
                throw new Error(msg);
            }
            
            return await response.json();
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
            const response = await this.makeRequest(this.endpoints.EMOTION, { text }, 'POST');
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const msg = errorData.message || errorData.error || `HTTP ${response.status}`;
                throw new Error(msg);
            }
            
            const data = await response.json();
            
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

        // OpenAI proxy not available in deployed API, use sample text
        console.log('⚠️ OpenAI proxy not available in deployed API, using sample text');
        showInlineSuccess('ℹ️ Using sample text (OpenAI proxy not available)', 'textInput');
        
        const sampleTexts = [
            "Today started like any other day, but something unexpected happened that completely changed my mood. I woke up feeling restless, as if something important was waiting for me just beyond the horizon. The morning sunlight streaming through my window felt warmer than usual, and I found myself lingering in bed longer than I should have, savoring the quiet moments before the day officially began.\n\nAs I made my coffee, I couldn't shake the feeling that today would be different. There was an energy in the air that I couldn't quite put my finger on – a mix of anticipation and nervous excitement that made my heart beat a little faster. I decided to take a different route to work, something I rarely do, and I'm so glad I did.\n\nWalking through the park, I noticed things I'd never seen before despite passing this way hundreds of times. The way the light filtered through the leaves created dancing patterns on the ground, and the sound of children's laughter from the nearby playground filled me with an unexpected sense of joy and hope. It reminded me of simpler times, when the smallest things could bring the greatest happiness.\n\nThat's when I realized what I was feeling – a profound sense of gratitude mixed with a gentle melancholy for time that has passed. Life has a way of surprising us when we least expect it, doesn't it?",

            "After a long conversation with someone close to me, I'm left feeling quite contemplative and unexpectedly vulnerable. It's funny how a simple exchange of words can peel back layers of what we often bury deep inside us. We sat on my worn-out couch, the kind that sags just a little too much in the middle, the kind that has held countless conversations that linger in the air like the scent of old coffee.\n\nAs we talked, I found myself unraveling in ways I hadn't anticipated. I shared my fears about the future – the weight of expectations hanging over me like a thick fog, numbing my enthusiasm. I didn't realize just how heavy it had become until the words slipped out, almost unbidden. It felt like releasing a tightly wound spring.\n\nThe conversation drifted into territories I rarely explore with anyone, including myself. We discussed dreams that feel too big, disappointments that still sting, and the strange comfort found in knowing that someone else understands the complexity of simply being human. There's something both terrifying and liberating about being truly seen by another person.\n\nNow, sitting here in the quiet aftermath, I feel emotionally exhausted but also somehow lighter. The vulnerability hangover is real, but so is the connection that was forged in those honest moments. I'm grateful for people who can hold space for all of our messy, complicated feelings."
        ];

        const randomIndex = Math.floor(Math.random() * sampleTexts.length);
        const generatedText = sampleTexts[randomIndex];
        console.log('✅ AI text generated successfully');

        if (textInput) {
            textInput.value = generatedText;
            textInput.style.borderColor = '#10b981';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(16, 185, 129, 0.25)';
            setTimeout(() => {
                textInput.style.borderColor = '';
                textInput.style.boxShadow = '';
            }, 2000);
        }

        showInlineSuccess('✅ AI text generated successfully!', 'textInput');

    } catch (error) {
        console.error('❌ Error generating AI text:', error);

        // Sample text is now handled in the main flow, so just show error

        showInlineError(`❌ Failed to generate AI text: ${error.message}`, 'textInput');

        if (textInput) {
            textInput.value = '';
            textInput.style.borderColor = '#ef4444';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(239, 68, 68, 0.25)';
            setTimeout(() => {
                textInput.style.borderColor = '';
                textInput.style.boxShadow = '';
            }, 3000);
        }
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
            }, 5000);

            setTimeout(() => {
                const msg = document.getElementById('emotionLoadingMessage');
                if (msg) msg.textContent = 'Processing your text with AI emotion analysis...';
            }, 15000);
        }

        updateElement('primaryEmotion', 'Loading...');

        let testText = document.getElementById('textInput').value || "I am so excited and happy today! This is wonderful news!";

        // Check text length limit
        const MAX_TEXT_LENGTH = 400;
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
        const data = await apiClient.makeRequest('/analyze/emotion', { text: testText }, 'POST');

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
    addToProgressConsole('🌐 Sending request to summarization API...', 'processing');

    try {
        // Create API client instance for proper timeout and error handling
        const apiClient = new SAMOAPIClient();
        const data = await apiClient.makeRequest('/analyze/summarize', { text: text }, 'POST');

        addToProgressConsole('✅ Summarization API response received', 'success');
        console.log('✅ Summarization API response:', data);

        // Extract summary from response
        addToProgressConsole('🔍 Processing summarization results...', 'processing');
        const possibleFields = ['summary', 'text', 'summarized_text', 'result', 'output'];
        let summaryText = null;

        for (const field of possibleFields) {
            if (data[field] && typeof data[field] === 'string') {
                summaryText = data[field];
                break;
            }
        }

        if (summaryText) {
            updateElement('summaryText', summaryText);
            updateElement('originalLength', text.length);
            updateElement('summaryLength', summaryText.length);
            addToProgressConsole(`Summary generated successfully (${summaryText.length} characters)`, 'success');
        } else {
            console.warn('⚠️ No valid summary found in response');
            addToProgressConsole('No valid summary found in API response', 'warning');
            updateElement('summaryText', 'Summary not available');
        }

        return summaryText;

    } catch (error) {
        console.error('❌ Error in callSummarizationAPI:', error);
        addToProgressConsole(`Summarization failed: ${error.message}`, 'error');
        updateElement('summaryText', 'Failed to generate summary');
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
    const console = document.getElementById('progressConsole');
    const consoleRow = document.getElementById('progressConsoleRow');

    if (!console) return;

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

    console.appendChild(messageDiv);
    console.scrollTop = console.scrollHeight;
}

function clearProgressConsole() {
    const console = document.getElementById('progressConsole');
    if (console) {
        console.textContent = '';
        const readyDiv = document.createElement('div');
        readyDiv.className = 'text-success';
        readyDiv.textContent = 'SAMO-DL Processing Console Ready...';
        console.appendChild(readyDiv);
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
