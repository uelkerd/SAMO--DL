/**
 * UI Controller Module
 * Handles all UI interactions and updates for the SAMO Demo
 */
class UIController {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.errorMsgEl = null;
    }

    initializeElements() {
        // Input elements
        this.audioFileInput = document.getElementById('audioFile');
        this.textInput = document.getElementById('textInput');
        this.processBtn = document.getElementById('processBtn');
        
        // Progress elements
        this.loadingSection = document.getElementById('loadingSection');
        this.progressSteps = document.querySelectorAll('.progress-step');
        
        // Result elements
        this.resultSection = document.getElementById('resultSection');
        this.transcriptionResult = document.getElementById('transcriptionResult');
        this.summaryResult = document.getElementById('summaryResult');
        this.emotionResult = document.getElementById('emotionResult');
    }

    setupEventListeners() {
        this.processBtn.addEventListener('click', () => this.handleProcessClick());
        this.audioFileInput.addEventListener('change', () => this.clearError());
        this.textInput.addEventListener('input', () => this.clearError());
    }

    handleProcessClick() {
        const audioFile = this.audioFileInput.files[0];
        const text = this.textInput.value.trim();

        // Clear previous error message
        this.clearError();

        if (!audioFile && !text) {
            this.showError('Please upload an audio file or enter text to process.');
            return;
        }

        // Trigger the main processing workflow
        if (window.demo) {
            window.demo.processCompleteWorkflow(audioFile, text);
        }
    }

    showError(message) {
        if (!this.errorMsgEl) {
            // Create error message element if it doesn't exist
            this.errorMsgEl = document.createElement('div');
            this.errorMsgEl.className = 'error-message';
            this.errorMsgEl.style.color = '#dc3545';
            this.errorMsgEl.style.background = '#f8d7da';
            this.errorMsgEl.style.border = '1px solid #f5c6cb';
            this.errorMsgEl.style.borderRadius = '8px';
            this.errorMsgEl.style.padding = '0.75rem';
            this.errorMsgEl.style.marginTop = '0.5rem';
            this.textInput.parentNode.insertBefore(this.errorMsgEl, this.textInput.nextSibling);
        }
        this.errorMsgEl.textContent = message;
        this.errorMsgEl.style.display = 'block';
    }

    clearError() {
        if (this.errorMsgEl) {
            this.errorMsgEl.textContent = '';
            this.errorMsgEl.style.display = 'none';
        }
    }

    showLoading() {
        this.loadingSection.classList.add('show');
        this.resultSection.classList.remove('show');
    }

    hideLoading() {
        this.loadingSection.classList.remove('show');
    }

    showResults() {
        this.resultSection.classList.add('show');
    }

    updateProgressStep(stepId, status) {
        const step = document.getElementById(stepId);
        if (step) {
            step.className = `progress-step ${status}`;
        }
    }

    updateProcessingInfo(results) {
        document.getElementById('totalTime').textContent = this.formatTime(results.processingTime);
        document.getElementById('processingStatus').textContent = 'Success';
        document.getElementById('processingStatus').className = 'text-success';
        document.getElementById('modelsUsed').textContent = results.modelsUsed.join(', ');
        
        // Calculate average confidence
        if (results.emotions && Array.isArray(results.emotions)) {
            const avgConfidence = results.emotions.reduce((sum, e) => 
                sum + (e.confidence || e.score || 0), 0) / results.emotions.length;
            document.getElementById('avgConfidence').textContent = `${(avgConfidence * 100).toFixed(1)}%`;
        }
    }

    formatTime(milliseconds) {
        if (milliseconds < 1000) {
            return `${milliseconds}ms`;
        } else {
            return `${(milliseconds / 1000).toFixed(2)}s`;
        }
    }

    showTranscriptionResults(transcription) {
        const content = document.createElement('div');
        content.className = 'result-content';
        
        const title = document.createElement('p');
        title.innerHTML = '<strong>Transcribed Text:</strong>';
        content.appendChild(title);
        
        const text = document.createElement('p');
        text.textContent = transcription.text;
        content.appendChild(text);
        
        const stats = document.createElement('div');
        stats.className = 'transcription-stats';
        stats.innerHTML = `
            <small class="text-muted">
                Duration: ${transcription.duration || 'N/A'} | 
                Confidence: ${((transcription.confidence || 0) * 100).toFixed(1)}% |
                Language: ${transcription.language || 'en'}
            </small>
        `;
        content.appendChild(stats);
        
        this.transcriptionResult.innerHTML = '';
        this.transcriptionResult.appendChild(content);
    }

    showSummaryResults(summary) {
        const content = document.createElement('div');
        content.className = 'result-content';
        
        const title = document.createElement('p');
        title.innerHTML = '<strong>Summary:</strong>';
        content.appendChild(title);
        
        const summaryContent = document.createElement('div');
        summaryContent.className = 'summary-content';
        const summaryText = document.createElement('p');
        summaryText.textContent = summary.summary;
        summaryContent.appendChild(summaryText);
        content.appendChild(summaryContent);
        
        const stats = document.createElement('div');
        stats.className = 'summary-stats';
        
        const statsData = [
            { value: summary.original_length, label: 'Original Length' },
            { value: summary.summary_length, label: 'Summary Length' },
            { value: summary.compression_ratio, label: 'Compression Ratio' }
        ];
        
        statsData.forEach(stat => {
            const statItem = document.createElement('div');
            statItem.className = 'stat-item';
            
            const statValue = document.createElement('div');
            statValue.className = 'stat-value';
            statValue.textContent = stat.value;
            statItem.appendChild(statValue);
            
            const statLabel = document.createElement('div');
            statLabel.className = 'stat-label';
            statLabel.textContent = stat.label;
            statItem.appendChild(statLabel);
            
            stats.appendChild(statItem);
        });
        
        content.appendChild(stats);
        
        this.summaryResult.innerHTML = '';
        this.summaryResult.appendChild(content);
    }

    showEmotionResults(emotions) {
        const content = document.createElement('div');
        content.className = 'result-content';
        
        const title = document.createElement('p');
        title.innerHTML = '<strong>Detected Emotions:</strong>';
        content.appendChild(title);
        
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
        
        // Normalize emotion data to ensure consistent key names
        const normalizedEmotions = emotionData.map(emotion => ({
            emotion: emotion.emotion || emotion.label || 'Unknown',
            confidence: emotion.confidence || emotion.score || 0
        }));
        
        normalizedEmotions.forEach(emotion => {
            const confidence = Math.max(0, Math.min(1, emotion.confidence)) * 100; // Clamp between 0-100
            const emotionName = emotion.emotion || 'Unknown';
            
            const emotionItem = document.createElement('div');
            emotionItem.className = 'emotion-item';
            
            const emotionNameSpan = document.createElement('span');
            emotionNameSpan.className = 'emotion-name';
            emotionNameSpan.textContent = emotionName;
            emotionItem.appendChild(emotionNameSpan);
            
            const emotionConfidence = document.createElement('span');
            emotionConfidence.className = 'emotion-confidence';
            emotionConfidence.textContent = `${confidence.toFixed(1)}%`;
            emotionItem.appendChild(emotionConfidence);
            
            content.appendChild(emotionItem);
        });
        
        this.emotionResult.innerHTML = '';
        this.emotionResult.appendChild(content);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
