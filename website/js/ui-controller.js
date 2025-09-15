/**
 * UI Controller Module
 * Handles all UI interactions and updates for the SAMO Demo
 */
class UIController {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.errorMsgEl = null;
        this.chartUtils = new ChartUtils();
    }

    initializeElements() {
        // Input elements
        this.audioFileInput = document.getElementById('audioFile');
        this.textInput = document.getElementById('textInput');
        this.processBtn = document.getElementById('processBtn');

        // Log if critical elements are missing
        if (!this.processBtn) {
            console.error('Critical element missing: processBtn');
        }
        if (!this.textInput) {
            console.error('Critical element missing: textInput');
        }

        // Progress elements
        this.loadingSection = document.getElementById('loadingSection');
        this.progressSteps = document.querySelectorAll('.progress-step');

        // Result elements
        this.resultSection = document.getElementById('resultSection');
        this.transcriptionResults = document.getElementById('transcriptionResults');
        this.summarizationResults = document.getElementById('summarizationResults');
        this.emotionResults = document.getElementById('emotionResults');

        // Individual result containers
        this.transcriptionText = document.getElementById('transcriptionText');
        this.summaryText = document.getElementById('summaryText');
        this.emotionBadges = document.getElementById('emotionBadges');
        this.emotionDetails = document.getElementById('emotionDetails');
    }

    setupEventListeners() {
        // Only set up event listeners if elements exist
        if (this.processBtn) {
            this.processBtn.addEventListener('click', () => this.handleProcessClick());
            console.log('Process button event listener attached');
        } else {
            console.error('Cannot attach event listener: processBtn not found');
        }

        if (this.audioFileInput) {
            this.audioFileInput.addEventListener('change', () => this.clearError());
        }

        if (this.textInput) {
            this.textInput.addEventListener('input', () => this.clearError());
        }
    }

    handleProcessClick() {
        console.log('Process button clicked!');

        const audioFile = this.audioFileInput ? this.audioFileInput.files[0] : null;
        const text = this.textInput ? this.textInput.value.trim() : '';

        console.log('Audio file:', audioFile);
        console.log('Text input:', text);

        // Clear previous error message
        this.clearError();

        if (!audioFile && !text) {
            console.log('No input provided, showing error');
            this.showError('Please upload an audio file or enter text to process.');
            return;
        }

        // Trigger the main processing workflow
        if (window.demo) {
            if (typeof window.demo.processCompleteWorkflow === 'function') {
                console.log('Calling demo.processCompleteWorkflow with:', { audioFile, text });
                window.demo.processCompleteWorkflow(audioFile, text);
            } else {
                this.showError('Demo not properly initialized. Please refresh the page.');
            }
        } else {
            console.error('window.demo not available');
            this.showError('Demo system not initialized. Please refresh the page.');
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
            // Update the step container class
            if (step.classList.contains('progress-step-vertical')) {
                // New vertical design
                step.className = `progress-step-vertical ${status}`;

                // Also update the icon
                const icon = step.querySelector('.step-icon-small');
                if (icon) {
                    icon.className = `step-icon-small ${status}`;
                }
            } else {
                // Legacy vertical design
                step.className = `progress-step ${status}`;
            }

            console.log(`Updated step ${stepId} to status: ${status}`);
        } else {
            console.warn(`Step element not found: ${stepId}`);
        }
    }

    updateProcessingInfo(results) {
        document.getElementById('totalTime').textContent = this.formatTime(results.processingTime);
        document.getElementById('processingStatus').textContent = 'Success';
        document.getElementById('processingStatus').className = 'text-success';
        document.getElementById('modelsUsed').textContent = results.modelsUsed.join(', ');
        
        // Calculate average confidence
        let avgConfidence = 0;
        if (results.emotions) {
            if (Array.isArray(results.emotions)) {
                // Handle array format
                avgConfidence = results.emotions.reduce((sum, e) => 
                    sum + (e.confidence || e.score || 0), 0) / results.emotions.length;
            } else if (results.emotions.emotions && Array.isArray(results.emotions.emotions)) {
                // Handle object with emotions array
                avgConfidence = results.emotions.emotions.reduce((sum, e) => 
                    sum + (e.confidence || e.score || 0), 0) / results.emotions.emotions.length;
            } else if (results.emotions.confidence) {
                // Handle object format with confidence property
                avgConfidence = results.emotions.confidence;
            } else if (results.emotions.emotion_analysis && results.emotions.emotion_analysis.confidence) {
                // Handle nested emotion_analysis format
                avgConfidence = results.emotions.emotion_analysis.confidence;
            }
        }
        
        if (avgConfidence > 0) {
            document.getElementById('avgConfidence').textContent = `${(avgConfidence * 100).toFixed(1)}%`;
        } else {
            document.getElementById('avgConfidence').textContent = 'N/A';
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
        // Show the transcription results section by replacing class
        this.transcriptionResults.className = 'row mb-4 result-section-visible';

        // Update the transcription text
        this.transcriptionText.textContent = transcription.text || 'No transcription available';
        this.transcriptionText.style.color = '#e2e8f0';

        // Update confidence and duration
        const confidence = ((transcription.confidence || 0) * 100).toFixed(1);
        const duration = transcription.duration || 'N/A';

        document.getElementById('transcriptionConfidence').textContent = `${confidence}%`;
        document.getElementById('transcriptionDuration').textContent = duration;
    }

    showSummaryResults(summary) {
        console.log('showSummaryResults called with:', summary);
        // Show the summarization results section by replacing class
        this.summarizationResults.className = 'row mb-4 result-section-visible';

        // Update the summary text
        const summaryText = summary.summary || summary.text || 'No summary available';
        this.summaryText.textContent = summaryText;
        this.summaryText.style.color = '#e2e8f0';

        // Update length statistics
        document.getElementById('originalLength').textContent = summary.original_length || '0';
        document.getElementById('summaryLength').textContent = summary.summary_length || summaryText.length || '0';
    }

    showEmotionResults(emotions) {
        console.log('showEmotionResults called with:', emotions);
        // Show the emotion results section by replacing class
        this.emotionResults.className = 'row mb-4 result-section-visible';
        
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
        
        // Sort by confidence (highest first) and take top 5
        const topEmotions = normalizedEmotions
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 5);
        
        // Clear previous content
        this.emotionBadges.innerHTML = '';
        this.emotionDetails.innerHTML = '';
        
        // Create emotion badges
        topEmotions.forEach(emotion => {
            const confidence = Math.max(0, Math.min(1, emotion.confidence)) * 100;
            const emotionName = emotion.emotion || 'Unknown';
            
            const badge = document.createElement('span');
            badge.className = 'badge bg-primary me-2 mb-2';
            badge.style.fontSize = '0.9rem';
            badge.textContent = `${emotionName} `;
            const small = document.createElement('small');
            small.textContent = `(${confidence.toFixed(1)}%)`;
            badge.appendChild(small);
            
            this.emotionBadges.appendChild(badge);
        });
        
        // Create detailed emotion list
        const detailsList = document.createElement('div');
        detailsList.className = 'emotion-details-list';
        
        topEmotions.forEach((emotion, index) => {
            const confidence = Math.max(0, Math.min(1, emotion.confidence)) * 100;
            const emotionName = emotion.emotion || 'Unknown';
            
            const detailItem = document.createElement('div');
            detailItem.className = 'd-flex justify-content-between align-items-center mb-2';
            
            const emotionLabel = document.createElement('span');
            emotionLabel.textContent = emotionName;
            emotionLabel.className = 'fw-medium';
            
            const confidenceBar = document.createElement('div');
            confidenceBar.className = 'progress flex-grow-1 mx-3';
            confidenceBar.style.height = '8px';
            
            const progressBar = document.createElement('div');
            progressBar.className = 'progress-bar bg-primary';
            progressBar.style.width = `${confidence}%`;
            progressBar.setAttribute('role', 'progressbar');
            progressBar.setAttribute('aria-valuenow', confidence);
            progressBar.setAttribute('aria-valuemin', '0');
            progressBar.setAttribute('aria-valuemax', '100');
            
            confidenceBar.appendChild(progressBar);
            
            const confidenceText = document.createElement('small');
            confidenceText.className = 'text-muted';
            confidenceText.textContent = `${confidence.toFixed(1)}%`;
            
            detailItem.appendChild(emotionLabel);
            detailItem.appendChild(confidenceBar);
            detailItem.appendChild(confidenceText);
            
            detailsList.appendChild(detailItem);
        });
        
        this.emotionDetails.appendChild(detailsList);
        
        // Update the emotion chart if available
        console.log('Creating emotion chart with data:', topEmotions);
        let chartCreated = false;
        if (this.chartUtils && this.chartUtils.createEmotionChart) {
            try {
                const chartCanvas = document.getElementById('emotionChart');
                if (chartCanvas && chartCanvas.tagName === 'CANVAS') {
                    chartCreated = this.chartUtils.createEmotionChart('emotionChart', topEmotions);
                    if (chartCreated) {
                        console.log('Chart created successfully');
                    } else {
                        console.log('Chart creation failed, using fallback');
                    }
                } else {
                    console.error('Chart canvas element not found or is not a canvas');
                }
            } catch (error) {
                console.error('Chart creation failed:', error);
                chartCreated = false;
            }
        }

        if (!chartCreated) {
            console.log('Using fallback chart display');
            this.showEmotionChartFallback(topEmotions);
        }
    }

    showEmotionChartFallback(emotions) {
        const chartContainer = document.getElementById('emotionChart');
        if (!chartContainer) {
            return;
        }
        
        // Create a simple visual representation
        chartContainer.innerHTML = '';
        
        const chartDiv = document.createElement('div');
        chartDiv.className = 'emotion-chart-fallback';
        chartDiv.style.cssText = `
            padding: 20px;
            background: rgba(139, 92, 246, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(139, 92, 246, 0.3);
        `;
        
        const title = document.createElement('h6');
        title.textContent = 'Emotion Confidence Levels';
        title.style.cssText = 'color: #e2e8f0; margin-bottom: 15px; font-weight: bold;';
        chartDiv.appendChild(title);
        
        emotions.forEach((emotion, index) => {
            const confidence = Math.max(0, Math.min(1, emotion.confidence)) * 100;
            const emotionName = emotion.emotion || 'Unknown';
            
            const emotionBar = document.createElement('div');
            emotionBar.style.cssText = 'margin-bottom: 10px;';
            
            const label = document.createElement('div');
            label.style.cssText = 'display: flex; justify-content: space-between; margin-bottom: 5px; color: #e2e8f0; font-size: 0.9rem;';
            const nameSpan = document.createElement('span');
            nameSpan.textContent = emotionName;
            const confidenceSpan = document.createElement('span');
            confidenceSpan.textContent = `${confidence.toFixed(1)}%`;
            label.appendChild(nameSpan);
            label.appendChild(confidenceSpan);
            
            const barContainer = document.createElement('div');
            barContainer.style.cssText = 'background: rgba(255, 255, 255, 0.1); height: 8px; border-radius: 4px; overflow: hidden;';
            
            const bar = document.createElement('div');
            bar.style.cssText = `
                height: 100%;
                width: ${confidence}%;
                background: linear-gradient(90deg, #8b5cf6, #a855f7);
                border-radius: 4px;
                transition: width 0.5s ease;
            `;
            
            barContainer.appendChild(bar);
            emotionBar.appendChild(label);
            emotionBar.appendChild(barContainer);
            chartDiv.appendChild(emotionBar);
        });
        
        chartContainer.appendChild(chartDiv);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.textContent;
    }
}
