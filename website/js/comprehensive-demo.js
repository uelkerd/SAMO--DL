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
            VOICE_JOURNAL: '/analyze/voice-journal'
        };
        this.timeout = window.SAMO_CONFIG?.API?.TIMEOUT || 45000;
        this.retryAttempts = window.SAMO_CONFIG?.API?.RETRY_ATTEMPTS || 3;
    }

    async makeRequest(endpoint, data, method = 'POST', isFormData = false, timeoutMs = null) {
        return this.makeRequestWithRetry(endpoint, data, method, isFormData, timeoutMs, this.retryAttempts);
    }

    async makeRequestWithRetry(endpoint, data, method = 'POST', isFormData = false, timeoutMs = null, attemptsLeft = 3) {
        const config = {
            method,
            headers: {}
        };
        const controller = new AbortController();
        const timeout = timeoutMs || this.timeout;
        const timer = setTimeout(() => controller.abort(new Error('Request timeout')), timeout);
        config.signal = controller.signal;

        // Remove API key requirement for now - using public endpoints
        // if (this.apiKey) {
        //     config.headers['X-API-Key'] = this.apiKey;
        // }

        if (data && method === 'POST') {
            if (isFormData) {
                // For FormData, don't set Content-Type header - let browser set it with boundary
                config.body = data;
            } else {
                config.headers['Content-Type'] = 'application/json';
                config.body = JSON.stringify(data);
            }
        } else if (method === 'GET') {
            config.headers['Content-Type'] = 'application/json';
        }

        try {
            const url = `${this.baseURL}${endpoint}`;
            const response = await fetch(url, config);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const msg = errorData.message || errorData.error || `HTTP ${response.status}`;

                // Handle retryable errors
                if (response.status === 429 || response.status >= 500) {
                    if (attemptsLeft > 1) {
                        const backoffDelay = Math.pow(2, this.retryAttempts - attemptsLeft) * 1000; // Exponential backoff
                        console.warn(`Request failed (${response.status}), retrying in ${backoffDelay}ms. Attempts left: ${attemptsLeft - 1}`);
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

            return await response.json();
        } catch (error) {
            // Handle network errors with retry
            if ((error.name === 'AbortError' || error.message.includes('timeout') || error.message.includes('network')) && attemptsLeft > 1) {
                const backoffDelay = Math.pow(2, this.retryAttempts - attemptsLeft) * 1000;
                console.warn(`Network error, retrying in ${backoffDelay}ms. Attempts left: ${attemptsLeft - 1}`, error.message);
                await new Promise(resolve => setTimeout(resolve, backoffDelay));
                return this.makeRequestWithRetry(endpoint, data, method, isFormData, timeoutMs, attemptsLeft - 1);
            }

            console.error('API request failed:', error);
            throw error;
        } finally {
            clearTimeout(timer);
        }
    }

    async transcribeAudio(audioFile) {
        const formData = new FormData();
        formData.append('audio_file', audioFile);

        try {
            // Use VOICE_JOURNAL endpoint for audio analysis flows (no auth header)
            const config = {
                method: 'POST',
                body: formData
            };
            const response = await fetch(`${this.baseURL}${this.endpoints.VOICE_JOURNAL}`, config);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const msg = errorData.message || errorData.error || `HTTP ${response.status}`;
                throw new Error(msg);
            }

            return await response.json();
        } catch (error) {
            console.error('Transcription error:', error);
            throw error;
        }
    }

    async summarizeText(text) {
        try {
            // Use query parameters instead of JSON body for summarize API
            const url = `${this.baseURL}${this.endpoints.SUMMARIZE}?text=${encodeURIComponent(text)}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Length': '0'
                }
            });

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
            // Use query parameters instead of JSON body for emotion API
            const url = `${this.baseURL}${this.endpoints.EMOTION}?text=${encodeURIComponent(text)}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Length': '0'
                }
            });

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

class ComprehensiveDemo {
    constructor() {
        this.apiClient = new SAMOAPIClient();
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.chart = null;
        this.performanceOptimizer = new PerformanceOptimizer();

        // Add cleanup on page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });

        // Periodic cleanup to prevent memory buildup
        this.cleanupInterval = setInterval(() => {
            this.periodicCleanup();
        }, 30000); // Every 30 seconds

        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        // Input elements
        this.audioFileInput = document.getElementById('audioFile');
        this.textInput = document.getElementById('textInput');
        this.recordBtn = document.getElementById('recordBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.processBtn = document.getElementById('processBtn');
        this.clearBtn = document.getElementById('clearBtn');

        // Visual elements
        this.audioVisualizer = document.getElementById('audioVisualizer');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultSection = document.getElementById('resultSection');

        // Progress steps
        this.steps = {
            step1: document.getElementById('step1'),
            step2: document.getElementById('step2'),
            step3: document.getElementById('step3'),
            step4: document.getElementById('step4')
        };

        // Result containers
        this.transcriptionResults = document.getElementById('transcriptionResults');
        this.summarizationResults = document.getElementById('summarizationResults');
        this.emotionResults = document.getElementById('emotionResults');
    }

    bindEvents() {
        this.processBtn.addEventListener('click', () => this.processInput());
        this.clearBtn.addEventListener('click', () => this.clearAll());
        this.recordBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        this.audioFileInput.addEventListener('change', () => this.handleFileUpload());
    }

    async processInput() {
        const audioFile = this.audioFileInput.files[0];
        const text = this.textInput.value.trim();

        if (!audioFile && !text) {
            this.showError('Please upload an audio file or enter text to process.');
            return;
        }

        this.showLoading();
        this.resetProgressSteps();
        this.hideResults();

        try {
            // Update progress
            this.updateProgressStep('step1', 'completed');
            this.updateLoadingMessage('Processing with AI...');

            const results = await this.apiClient.processCompleteWorkflow(audioFile, text);

            // Update progress steps
            if (results.transcription) {
                this.updateProgressStep('step2', 'completed');
                this.showTranscriptionResults(results.transcription);
            }

            if (results.summary) {
                this.updateProgressStep('step3', 'completed');
                this.showSummarizationResults(results.summary, results);
            }

            if (results.emotions) {
                this.updateProgressStep('step4', 'completed');
                this.showEmotionResults(results.emotions);
            }

            this.updateProcessingInfo(results);
            this.hideLoading();
            this.showResults();

        } catch (error) {
            console.error('Processing failed:', error);
            this.hideLoading();
            this.showError(`Processing failed: ${error.message}`);
        }
    }

    showLoading() {
        this.loadingSection.classList.add('show');
        this.resultSection.classList.remove('show');
        this.loadingSection.setAttribute('aria-busy', 'true');
        this.resultSection.setAttribute('aria-busy', 'false');
    }

    hideLoading() {
        this.loadingSection.classList.remove('show');
    }

    updateLoadingMessage(message) {
        document.getElementById('loadingMessage').textContent = message;
    }

    resetProgressSteps() {
        Object.values(this.steps).forEach(step => {
            step.classList.remove('completed', 'active');
            const icon = step.querySelector('.step-icon');
            if (icon) {
                icon.classList.remove('completed', 'active');
                icon.classList.add('pending');
            }
        });
    }

    updateProgressStep(stepId, status) {
        const step = this.steps[stepId];
        const icon = step.querySelector('.step-icon');

        step.classList.remove('completed', 'active');
        if (icon) {
            icon.classList.remove('completed', 'active', 'pending');

            if (status === 'completed') {
                step.classList.add('completed');
                icon.classList.add('completed');
            } else if (status === 'active') {
                step.classList.add('active');
                icon.classList.add('active');
            } else {
                icon.classList.add('pending');
            }
        }
    }

    showTranscriptionResults(transcription) {
        // Some API responses use 'text', others use 'transcription'. Normalize here for consistency.
        const text = transcription.text || transcription.transcription || 'Transcription not available';
        const confidence = transcription.confidence || 'N/A';
        const duration = transcription.duration || 'N/A';

        document.getElementById('transcriptionText').textContent = text;
        document.getElementById('transcriptionConfidence').textContent =
            typeof confidence === 'number' ? `${Math.round(confidence * 100)}%` : confidence;
        document.getElementById('transcriptionDuration').textContent =
            typeof duration === 'number' ? `${duration.toFixed(2)}s` : duration;

        this.transcriptionResults.style.display = 'block';
    }

    showSummarizationResults(summary, results = null) {
        const summaryText = summary.summary || summary.text || 'Summary not available';
        const summaryLength = summaryText.length;

        // Determine original text length from available sources
        let originalLength = 0;
        if (results) {
            // Try to get original text from various sources in order of preference
            if (results.originalText) {
                originalLength = (results.originalText || '').length;
            } else if (results.transcription) {
                const transcribedText = results.transcription.text || results.transcription.transcription;
                originalLength = transcribedText ? transcribedText.length : 0;
            } else if (results.inputText) {
                originalLength = results.inputText.length;
            }
        }

        document.getElementById('summaryText').textContent = summaryText;
        document.getElementById('originalLength').textContent = originalLength;
        document.getElementById('summaryLength').textContent = summaryLength;

        this.summarizationResults.style.display = 'block';
    }

    showEmotionResults(emotions) {
        // Handle different response formats
        let emotionData = [];
        if (Array.isArray(emotions)) {
            emotionData = emotions;
        } else if (emotions.emotions) {
            emotionData = emotions.emotions;
        } else if (emotions.predictions) {
            emotionData = emotions.predictions;
        } else if (emotions.probabilities) {
            // Handle probabilities object format: {probabilities: {label: prob}}
            emotionData = Object.entries(emotions.probabilities).map(([label, prob]) => ({
                emotion: label,
                confidence: prob
            }));
        }

        // Use performance optimizer to normalize emotion data
        const normalizedEmotions = this.performanceOptimizer.optimizeEmotionData(emotionData);
        console.log('🔍 Normalized emotions for chart:', normalizedEmotions);
        console.log('🔍 Normalized emotions length:', normalizedEmotions.length);

        // Create emotion badges (only show top 5)
        const badgesContainer = document.getElementById('emotionBadges');
        badgesContainer.textContent = '';

        // Only show top 5 emotions as badges
        const top5Emotions = normalizedEmotions.slice(0, 5);
        top5Emotions.forEach(emotion => {
            const confidence = Math.max(0, Math.min(1, emotion.confidence)) * 100; // Clamp between 0-100
            const emotionName = emotion.emotion || 'Unknown';

            const badge = document.createElement('span');
            badge.className = 'emotion-badge';
            badge.style.backgroundColor = this.getEmotionColor(emotionName);
            badge.textContent = `${emotionName}: ${confidence.toFixed(1)}%`;
            badgesContainer.appendChild(badge);
        });

        // Create emotion chart (only top 5 emotions)
        const chartData = normalizedEmotions.slice(0, 5);
        console.log('🔍 Creating chart with data:', chartData);
        this.createEmotionChart(chartData);

        // Show emotion details (only top 5)
        this.showEmotionDetails(chartData);

        this.emotionResults.style.display = 'block';
    }

    createEmotionChart(emotionData) {
        const ctx = document.getElementById('emotionChart');
        if (!ctx) {
            console.error('Emotion chart canvas not found');
            return;
        }

        // Destroy existing chart properly
        if (this.chart) {
            try {
                this.chart.destroy();
                this.chart = null;
            } catch (error) {
                console.warn('Error destroying chart:', error);
                this.chart = null;
            }
        }

        // Use the basic chart directly since we have Chart.js
        this.createBasicChart(ctx, emotionData);
    }

    createBasicChart(ctx, emotionData) {
        // Fallback chart creation if performance optimizer fails
        console.log('🔍 createBasicChart called with:', emotionData);
        console.log('🔍 emotionData type:', typeof emotionData);
        console.log('🔍 emotionData length:', emotionData?.length);

        // Check if Chart.js is loaded
        if (typeof Chart === 'undefined') {
            console.error('❌ Chart.js not loaded!');
            this.showChartError('Chart.js library not loaded. Please refresh the page.');
            return;
        }

        if (!Array.isArray(emotionData) || emotionData.length === 0) {
            console.error('❌ Invalid emotion data for chart:', emotionData);
            return;
        }

        const labels = emotionData.map(e => e.emotion || e.label);
        const data = emotionData.map(e => (e.confidence || e.score) * 100);
        const colors = labels.map(label => this.getEmotionColor(label));

        console.log('🔍 Chart labels:', labels);
        console.log('🔍 Chart data:', data);
        console.log('🔍 Chart colors:', colors);

        try {
            this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence (%)',
                    data: data,
                    backgroundColor: colors,
                    borderColor: colors.map((c) =>
                        c.startsWith('rgba(')
                          ? c.replace(/rgba\((\d+\s*,\s*\d+\s*,\s*\d+),\s*[\d.]+\)/, 'rgba($1, 1)')
                          : c
                    ),
                    borderWidth: 2,
                    borderRadius: 8,
                    borderSkipped: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(139, 92, 246, 0.1)',
                            borderColor: 'rgba(139, 92, 246, 0.2)'
                        },
                        ticks: {
                            color: '#cbd5e1',
                            maxRotation: 45
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(139, 92, 246, 0.1)',
                            borderColor: 'rgba(139, 92, 246, 0.2)'
                        },
                        ticks: {
                            color: '#cbd5e1',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 15, 35, 0.9)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#e2e8f0',
                        borderColor: 'rgba(139, 92, 246, 0.5)',
                        borderWidth: 1
                    }
                }
            }
        });

        } catch (error) {
            console.error('❌ Error creating chart:', error);
            this.showChartError('Failed to create chart: ' + error.message);
        }
    }

    /**
     * Show chart error message
     */
    showChartError(message) {
        const chartContainer = document.getElementById('emotionChart');
        if (chartContainer) {
            const parent = chartContainer.parentElement;
            if (parent) {
                // Clear existing content safely
                parent.textContent = '';

                // Create alert container
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-warning';
                alertDiv.setAttribute('role', 'alert');

                // Create heading
                const heading = document.createElement('h6');
                heading.className = 'alert-heading';

                const warningIcon = document.createElement('span');
                warningIcon.className = 'material-icons me-2';
                warningIcon.textContent = 'warning';

                heading.appendChild(warningIcon);
                heading.appendChild(document.createTextNode('Chart Error'));

                // Create message paragraph
                const messagePara = document.createElement('p');
                messagePara.className = 'mb-0';
                messagePara.textContent = message; // Safe text content

                // Create separator
                const hr = document.createElement('hr');

                // Create instruction paragraph
                const instructionPara = document.createElement('p');
                instructionPara.className = 'mb-0 small';
                instructionPara.textContent = 'Please refresh the page and try again.';

                // Assemble the alert
                alertDiv.appendChild(heading);
                alertDiv.appendChild(messagePara);
                alertDiv.appendChild(hr);
                alertDiv.appendChild(instructionPara);

                parent.appendChild(alertDiv);
            }
        }
    }

    showEmotionDetails(emotionData) {
        const detailsContainer = document.getElementById('emotionDetails');
        if (!detailsContainer) {
            console.error('❌ emotionDetails container not found');
            return;
        }
        const title = document.createElement('h6');
        title.className = 'fw-bold mb-3';
        title.textContent = 'Top Emotions';
        detailsContainer.textContent = '';
        detailsContainer.appendChild(title);

        // Sort by confidence and show top 5
        const sortedEmotions = emotionData
            .sort((a, b) => (b.confidence || b.score) - (a.confidence || a.score))
            .slice(0, 5);

        sortedEmotions.forEach((emotion, index) => {
            const confidence = (emotion.confidence || emotion.score) * 100;
            const emotionName = emotion.emotion || emotion.label;

            const detailItem = document.createElement('div');
            detailItem.className = 'mb-3';

            const headerDiv = document.createElement('div');
            headerDiv.className = 'd-flex justify-content-between align-items-center mb-1';

            const emotionLabel = document.createElement('span');
            emotionLabel.className = 'fw-bold';
            emotionLabel.textContent = `${index + 1}. ${emotionName}`;

            const badge = document.createElement('span');
            badge.className = 'badge';
            badge.style.backgroundColor = this.getEmotionColor(emotionName);
            badge.textContent = `${Math.round(confidence)}%`;

            headerDiv.appendChild(emotionLabel);
            headerDiv.appendChild(badge);

            const progressDiv = document.createElement('div');
            progressDiv.className = 'progress';
            progressDiv.style.height = '8px';

            const progressBar = document.createElement('div');
            progressBar.className = 'progress-bar';
            progressBar.style.width = `${confidence}%`;
            progressBar.style.backgroundColor = this.getEmotionColor(emotionName);

            progressDiv.appendChild(progressBar);

            detailItem.appendChild(headerDiv);
            detailItem.appendChild(progressDiv);
            detailsContainer.appendChild(detailItem);
        });
    }

    getEmotionColor(emotion) {
        const colors = {
            'joy': 'rgba(34, 197, 94, 0.8)',
            'happiness': 'rgba(34, 197, 94, 0.8)',
            'excitement': 'rgba(34, 197, 94, 0.8)',
            'sadness': 'rgba(59, 130, 246, 0.8)',
            'grief': 'rgba(59, 130, 246, 0.8)',
            'anger': 'rgba(239, 68, 68, 0.8)',
            'annoyance': 'rgba(239, 68, 68, 0.8)',
            'fear': 'rgba(245, 158, 11, 0.8)',
            'nervousness': 'rgba(245, 158, 11, 0.8)',
            'surprise': 'rgba(139, 92, 246, 0.8)',
            'love': 'rgba(244, 63, 94, 0.8)',
            'caring': 'rgba(244, 63, 94, 0.8)',
            'gratitude': 'rgba(16, 185, 129, 0.8)',
            'pride': 'rgba(16, 185, 129, 0.8)',
            'optimism': 'rgba(16, 185, 129, 0.8)',
            'disgust': 'rgba(107, 114, 128, 0.8)',
            'confusion': 'rgba(107, 114, 128, 0.8)',
            'neutral': 'rgba(107, 114, 128, 0.8)'
        };
        return colors[emotion] || 'rgba(139, 92, 246, 0.8)';
    }

    updateProcessingInfo(results) {
        // Format processing time for better readability
        const formatProcessingTime = (ms) => {
            if (ms >= 1000) {
                return `${(ms / 1000).toFixed(2)}s`;
            }
            return `${ms}ms`;
        };
        document.getElementById('totalTime').textContent = formatProcessingTime(results.processingTime);
        document.getElementById('processingStatus').textContent = 'Success';
        document.getElementById('processingStatus').className = 'text-success';
        document.getElementById('modelsUsed').textContent = results.modelsUsed.join(', ');

        // Calculate average confidence - handle different response formats
        const em = results.emotions;
        if (em) {
            let avg = null;
            if (Array.isArray(em)) {
                avg = em.reduce((s, e) => s + (e.confidence || e.score || 0), 0) / Math.max(em.length, 1);
            } else if (em.probabilities && typeof em.probabilities === 'object') {
                const vals = Object.values(em.probabilities);
                avg = vals.reduce((s, v) => s + (Number(v) || 0), 0) / Math.max(vals.length, 1);
            }
            if (avg != null) {
                document.getElementById('avgConfidence').textContent = `${Math.round(avg * 100)}%`;
            } else {
                document.getElementById('avgConfidence').textContent = 'N/A';
            }
        } else {
            document.getElementById('avgConfidence').textContent = 'N/A';
        }
    }

    showResults() {
        this.resultSection.classList.add('show');
    }

    hideResults() {
        this.resultSection.classList.remove('show');
        this.resultSection.setAttribute('aria-busy', 'false');
        this.transcriptionResults.style.display = 'none';
        this.summarizationResults.style.display = 'none';
        this.emotionResults.style.display = 'none';
    }

    clearAll() {
        this.audioFileInput.value = '';
        this.textInput.value = '';
        this.hideResults();
        this.resetProgressSteps();
        this.stopRecording();
    }

    async startRecording() {
        try {
            if (typeof window.MediaRecorder === 'undefined') {
                this.showError('Recording not supported in this browser.');
                return;
            }
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                // Use the actual MediaRecorder MIME type instead of hardcoded 'audio/wav'
                const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
                const fileExtension = mimeType.includes('webm') ? 'webm' :
                                    mimeType.includes('mp4') ? 'mp4' :
                                    mimeType.includes('ogg') ? 'ogg' : 'wav';

                const audioBlob = new Blob(this.audioChunks, { type: mimeType });
                const audioFile = new File([audioBlob], `recording.${fileExtension}`, { type: mimeType });

                // Create a new FileList-like object
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(audioFile);
                this.audioFileInput.files = dataTransfer.files;

                // Hide visualizer
                this.audioVisualizer.style.display = 'none';
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.recordBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.audioVisualizer.style.display = 'flex';

        } catch (error) {
            console.error('Error starting recording:', error);
            this.showError('Could not start recording. Please check microphone permissions.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            this.recordBtn.disabled = false;
            this.stopBtn.disabled = true;
        }
    }

    handleFileUpload() {
        if (this.audioFileInput.files[0]) {
            // Clear text input when audio is uploaded
            this.textInput.value = '';
        }
    }

    showError(message) {
        if (!this.errorMsgEl) {
            // Create error message element if it doesn't exist
            this.errorMsgEl = document.createElement('div');
            this.errorMsgEl.className = 'error-message';
            this.errorMsgEl.setAttribute('role', 'alert');
            this.errorMsgEl.setAttribute('aria-live', 'assertive');
            this.textInput.parentNode.insertBefore(this.errorMsgEl, this.textInput.nextSibling);
        }
        this.errorMsgEl.textContent = message;
        this.errorMsgEl.classList.add('show');
    }

    clearError() {
        if (this.errorMsgEl) {
            this.errorMsgEl.textContent = '';
            this.errorMsgEl.classList.remove('show');
        }
    }

    /**
     * Clean up resources to prevent memory leaks
     */
    cleanup() {
        console.log('🧹 Cleaning up resources...');

        // Destroy chart
        if (this.chart) {
            try {
                this.chart.destroy();
                this.chart = null;
            } catch (error) {
                console.warn('Error destroying chart during cleanup:', error);
            }
        }

        // Clean up performance optimizer
        if (this.performanceOptimizer && typeof this.performanceOptimizer.destroy === 'function') {
            this.performanceOptimizer.destroy();
        }

        // Stop media recording if active
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            try {
                this.mediaRecorder.stop();
            } catch (error) {
                console.warn('Error stopping media recorder:', error);
            }
        }

        // Clear audio chunks
        this.audioChunks = [];

        // Clear cleanup interval
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
            this.cleanupInterval = null;
        }

        console.log('✅ Cleanup completed');
    }

    /**
     * Periodic cleanup to prevent memory buildup
     */
    periodicCleanup() {
        // Only run if performance optimizer is available
        if (this.performanceOptimizer && typeof this.performanceOptimizer.cleanupMemory === 'function') {
            this.performanceOptimizer.cleanupMemory();
        }

        // Clear any old audio chunks
        if (this.audioChunks.length > 10) {
            this.audioChunks = this.audioChunks.slice(-5);
        }
    }
}

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ DOM loaded, initializing demo...');
    // DISABLED: ComprehensiveDemo class conflicts with simple-demo-functions.js
    // Using simple-demo-functions.js instead for better stability
    // new ComprehensiveDemo();
    console.log('🔧 Using simple-demo-functions.js for chart implementation');
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

        const openaiConfig = window.SAMO_CONFIG.OPENAI;
        const response = await fetch(openaiConfig.API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey.trim()}`
            },
            body: JSON.stringify({
                model: openaiConfig.MODEL,
                messages: [
                    {
                        role: 'system',
                        content: 'You are a creative writing assistant that generates authentic, emotionally rich personal journal entries. Write in first person, include specific details and genuine emotions.'
                    },
                    {
                        role: 'user',
                        content: `Write a personal journal entry that continues this thought: "${randomPrompt}" - Make it authentic and emotionally detailed.`
                    }
                ],
                max_tokens: openaiConfig.MAX_TOKENS,
                temperature: openaiConfig.TEMPERATURE + 0.1
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(`OpenAI API error: ${response.status} ${errorData.error?.message || ''}`);
        }

        const data = await response.json();
        if (!data.choices?.[0]?.message) {
            throw new Error('Invalid response format from OpenAI API');
        }

        const generatedText = data.choices[0].message.content.trim();
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

// Essential Processing Functions (restored from simple-demo-functions.js)

async function processText() {
    console.log('🚀 Processing text...');
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
        const apiUrl = `https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app/analyze/emotion?text=${encodeURIComponent(testText)}`;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort('Request timeout after 90 seconds - API may be experiencing cold start delays'), 90000); // Increased for cold starts

        addToProgressConsole('🌐 Sending request to emotion analysis API...', 'processing');
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Length': '0',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            addToProgressConsole(`API call failed: ${response.status} ${response.statusText}`, 'error');
            throw new Error(`API call failed: ${response.status} ${response.statusText}`);
        }

        addToProgressConsole('✅ Emotion analysis API response received', 'success');
        const data = await response.json();
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
        console.error('❌ Error in testWithRealAPI:', error);

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
    }
}

async function callSummarizationAPI(text) {
    console.log('📝 Calling real summarization API...');
    addToProgressConsole('🌐 Sending request to summarization API...', 'processing');

    try {
        const params = new URLSearchParams({
            text: text
        });

        const apiUrl = `${window.SAMO_CONFIG.API.BASE_URL}${window.SAMO_CONFIG.API.ENDPOINTS.SUMMARIZE}?${params.toString()}`;

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 45000);

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Length': '0',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            addToProgressConsole(`Summarization API failed: ${response.status} ${response.statusText}`, 'error');
            throw new Error(`Summarization API failed: ${response.status} ${response.statusText}`);
        }

        addToProgressConsole('✅ Summarization API response received', 'success');
        const data = await response.json();
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

function updateElement(id, value) {
    try {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value !== null && value !== undefined ? value : '-';
            console.log(`✅ Updated ${id}: ${value}`);
        } else {
            console.warn(`⚠️ Element not found: ${id}`);
        }
    } catch (error) {
        console.error(`❌ Error updating element ${id}:`, error);
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
    messageDiv.innerHTML = `<span class="text-muted">[${timestamp}]</span> ${icon} ${message}`;

    console.appendChild(messageDiv);
    console.scrollTop = console.scrollHeight;
}

function clearProgressConsole() {
    const console = document.getElementById('progressConsole');
    if (console) {
        console.innerHTML = '<div class="text-success">SAMO-DL Processing Console Ready...</div>';
    }
}

// Enhanced updateElement function that handles different content types
function updateElement(id, value) {
    try {
        const element = document.getElementById(id);
        if (element) {
            if (id === 'summaryText') {
                // Special handling for summary text - use dark-theme compatible styling
                element.innerHTML = `<div class="p-3 bg-dark border border-secondary rounded text-light">${value !== null && value !== undefined ? value : 'No summary available'}</div>`;
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
        chartContainer.innerHTML = '';

        if (!emotionData || emotionData.length === 0) {
            chartContainer.innerHTML = '<div class="text-muted text-center p-3">No emotion data available</div>';
            addToProgressConsole('No emotion data available for chart', 'warning');
            return;
        }

        // Take top 5 emotions
        const top5Emotions = emotionData.slice(0, 5);

        // Create simple bar chart with Bootstrap classes
        let chartHTML = '<div class="emotion-bars">';

        top5Emotions.forEach((emotion, index) => {
            const name = emotion.emotion || emotion.label || `Emotion ${index + 1}`;
            const confidence = ((emotion.confidence || emotion.score || 0) * 100).toFixed(1);
            const percentage = Math.max(5, confidence); // Minimum 5% for visibility

            const colors = ['primary', 'success', 'warning', 'info', 'secondary'];
            const colorClass = colors[index % colors.length];

            chartHTML += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between align-items-center mb-1">
                        <small class="fw-bold text-capitalize">${name}</small>
                        <small class="text-muted">${confidence}%</small>
                    </div>
                    <div class="progress" style="height: 20px;">
                        <div class="progress-bar bg-${colorClass}"
                             style="width: ${percentage}%;"
                             role="progressbar"
                             aria-valuenow="${confidence}"
                             aria-valuemin="0"
                             aria-valuemax="100">
                        </div>
                    </div>
                </div>
            `;
        });

        chartHTML += '</div>';
        chartContainer.innerHTML = chartHTML;

        addToProgressConsole(`Emotion chart created with ${top5Emotions.length} emotions`, 'success');

    } catch (error) {
        console.error('Error creating emotion chart:', error);
        addToProgressConsole(`Error creating emotion chart: ${error.message}`, 'error');
        const chartContainer = document.getElementById('emotionChart');
        if (chartContainer) {
            chartContainer.innerHTML = '<div class="text-danger text-center p-3">Error creating chart</div>';
        }
    }
}

// Reset demo to input screen
function resetToInputScreen() {
    console.log('🔄 Resetting to input screen...');

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
        // Animate transition back to input
        resultsLayout.style.opacity = '0';
        resultsLayout.style.transform = 'translateY(20px)';

        setTimeout(() => {
            resultsLayout.classList.add('d-none');
            inputLayout.classList.remove('d-none');

            setTimeout(() => {
                inputLayout.style.opacity = '1';
                inputLayout.style.transform = 'translateY(0)';
            }, 50);
        }, 300);
    }

    console.log('✅ Reset completed');
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
