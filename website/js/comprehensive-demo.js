/**
 * Comprehensive AI Platform Demo
 * Handles voice transcription, text summarization, and emotion detection
 */

class SAMOAPIClient {
    constructor() {
        // Use configuration from config.js if available, otherwise fallback to demo mode
        this.baseURL = (typeof SAMO_CONFIG !== 'undefined') ? SAMO_CONFIG.baseURL : 'https://samo-unified-api-71517823771.us-central1.run.app';
        this.apiKey = (typeof SAMO_CONFIG !== 'undefined') ? SAMO_CONFIG.apiKey : 'demo-key-123';
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
                    throw new Error('Rate limit exceeded. Please try again in a moment.');
                } else if (response.status === 401) {
                    throw new Error('API key required. Please contact support for access.');
                } else if (response.status === 503) {
                    throw new Error('Service temporarily unavailable. Please try again later.');
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async transcribeAudio(audioFile) {
        const formData = new FormData();
        formData.append('audio_file', audioFile);
        
        try {
            const response = await fetch(`${this.baseURL}/transcribe/voice`, {
                method: 'POST',
                body: formData
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
            if (error.message.includes('Rate limit') || error.message.includes('API key') || error.message.includes('Service temporarily')) {
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
            return await this.makeRequest('/predict', { text });
        } catch (error) {
            // If API is not available, return mock data for demo purposes
            if (error.message.includes('Rate limit') || error.message.includes('API key') || error.message.includes('Service temporarily')) {
                console.warn('API not available, using mock data for demo:', error.message);
                return this.getMockEmotionResponse(text);
            }
            throw error;
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
                results.transcription = await this.transcribeAudio(audioFile);
                currentText = results.transcription.text || results.transcription.transcription;
                results.modelsUsed.push('SAMO Whisper');
            } catch (error) {
                console.error('Transcription failed:', error);
                throw new Error('Voice transcription failed. Please try again.');
            }
        }

        // Step 2: Summarize text
        if (currentText) {
            try {
                results.summary = await this.summarizeText(currentText);
                results.modelsUsed.push('SAMO T5');
            } catch (error) {
                console.error('Summarization failed:', error);
                // Continue without summary
            }
        }

        // Step 3: Detect emotions
        if (currentText) {
            try {
                results.emotions = await this.detectEmotions(currentText);
                results.modelsUsed.push('DeBERTa v3 Large');
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
            alert('Please upload an audio file or enter text to process.');
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
                this.showSummarizationResults(results.summary, text);
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
            alert(`Processing failed: ${error.message}`);
        }
    }

    showLoading() {
        this.loadingSection.classList.add('show');
        this.resultSection.classList.remove('show');
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
            icon.classList.remove('completed', 'active');
            icon.classList.add('pending');
        });
    }

    updateProgressStep(stepId, status) {
        const step = this.steps[stepId];
        const icon = step.querySelector('.step-icon');
        
        step.classList.remove('completed', 'active');
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

    showTranscriptionResults(transcription) {
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

    showSummarizationResults(summary, originalText) {
        const summaryText = summary.summary || summary.text || 'Summary not available';
        const originalLength = originalText.length;
        const summaryLength = summaryText.length;
        
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
        }

        // Create emotion badges
        const badgesContainer = document.getElementById('emotionBadges');
        badgesContainer.innerHTML = '';
        
        emotionData.forEach(emotion => {
            const badge = document.createElement('span');
            badge.className = 'emotion-badge';
            badge.style.backgroundColor = this.getEmotionColor(emotion.emotion || emotion.label);
            badge.textContent = `${emotion.emotion || emotion.label}: ${Math.round((emotion.confidence || emotion.score) * 100)}%`;
            badgesContainer.appendChild(badge);
        });

        // Create emotion chart
        this.createEmotionChart(emotionData);
        
        // Show emotion details
        this.showEmotionDetails(emotionData);
        
        this.emotionResults.style.display = 'block';
    }

    createEmotionChart(emotionData) {
        const ctx = document.getElementById('emotionChart').getContext('2d');
        
        // Destroy existing chart
        if (this.chart) {
            this.chart.destroy();
        }
        
        const labels = emotionData.map(e => e.emotion || e.label);
        const data = emotionData.map(e => (e.confidence || e.score) * 100);
        const colors = labels.map(label => this.getEmotionColor(label));
        
        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence (%)',
                    data: data,
                    backgroundColor: colors,
                    borderColor: colors.map(color => color.replace('0.8', '1')),
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
    }

    showEmotionDetails(emotionData) {
        const detailsContainer = document.getElementById('emotionDetails');
        detailsContainer.innerHTML = '<h6 class="fw-bold mb-3">Top Emotions</h6>';
        
        // Sort by confidence and show top 5
        const sortedEmotions = emotionData
            .sort((a, b) => (b.confidence || b.score) - (a.confidence || a.score))
            .slice(0, 5);
        
        sortedEmotions.forEach((emotion, index) => {
            const confidence = (emotion.confidence || emotion.score) * 100;
            const emotionName = emotion.emotion || emotion.label;
            
            const detailItem = document.createElement('div');
            detailItem.className = 'mb-3';
            detailItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="fw-bold">${index + 1}. ${emotionName}</span>
                    <span class="badge" style="background-color: ${this.getEmotionColor(emotionName)}">
                        ${Math.round(confidence)}%
                    </span>
                </div>
                <div class="progress" style="height: 8px;">
                    <div class="progress-bar" 
                         style="width: ${confidence}%; background-color: ${this.getEmotionColor(emotionName)}">
                    </div>
                </div>
            `;
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
        document.getElementById('totalTime').textContent = `${results.processingTime}ms`;
        document.getElementById('processingStatus').textContent = 'Success';
        document.getElementById('processingStatus').className = 'text-success';
        document.getElementById('modelsUsed').textContent = results.modelsUsed.join(', ');
        
        // Calculate average confidence
        if (results.emotions && Array.isArray(results.emotions)) {
            const avgConfidence = results.emotions.reduce((sum, e) => 
                sum + (e.confidence || e.score || 0), 0) / results.emotions.length;
            document.getElementById('avgConfidence').textContent = 
                `${Math.round(avgConfidence * 100)}%`;
        }
    }

    showResults() {
        this.resultSection.classList.add('show');
    }

    hideResults() {
        this.resultSection.classList.remove('show');
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
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
                
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
            alert('Could not start recording. Please check microphone permissions.');
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
}

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', function() {
    new ComprehensiveDemo();
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
