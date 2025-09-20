/**
 * Voice Recording Module for SAMO Demo
 * Handles microphone access, audio recording, and integration with the demo interface
 */

class VoiceRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;
        this.recordingStartTime = null;
        this.recordingTimer = null;

        // UI Elements
        this.recordBtn = null;
        this.stopBtn = null;
        this.recordingIndicator = null;
        this.recordingTime = null;

        // Bind methods
        this.startRecording = this.startRecording.bind(this);
        this.stopRecording = this.stopRecording.bind(this);
        this.onDataAvailable = this.onDataAvailable.bind(this);
        this.onRecordingStop = this.onRecordingStop.bind(this);
    }

    async init() {
        try {
            // Get UI elements
            this.recordBtn = document.getElementById('recordBtn');
            this.stopBtn = document.getElementById('stopBtn');
            this.recordingIndicator = document.querySelector('.recording-indicator');
            this.recordingTime = document.getElementById('recordingTime');

            if (!this.recordBtn || !this.stopBtn) {
                console.warn('Voice recording UI elements not found');
                return false;
            }

            // Add event listeners
            this.recordBtn.addEventListener('click', this.startRecording);
            this.stopBtn.addEventListener('click', this.stopRecording);

            // Check for MediaRecorder support
            if (!navigator.mediaDevices || !window.MediaRecorder) {
                console.error('MediaRecorder not supported');
                this.disableRecording('Voice recording not supported in this browser');
                return false;
            }

            // Enable recording UI
            this.recordBtn.disabled = false;
            console.log('✅ Voice recorder initialized successfully');
            return true;

        } catch (error) {
            console.error('Failed to initialize voice recorder:', error);
            this.disableRecording('Failed to initialize voice recording');
            return false;
        }
    }

    async startRecording() {
        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 44100
                }
            });

            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: this.getSupportedMimeType()
            });

            // Set up event handlers
            this.mediaRecorder.ondataavailable = this.onDataAvailable;
            this.mediaRecorder.onstop = this.onRecordingStop;

            // Reset audio chunks
            this.audioChunks = [];

            // Start recording
            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            this.recordingStartTime = Date.now();

            // Update UI
            this.updateRecordingUI(true);
            this.startRecordingTimer();

            console.log('🎙️ Recording started');

        } catch (error) {
            console.error('Failed to start recording:', error);
            this.handleRecordingError(error);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;

            // Stop all tracks
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
                this.stream = null;
            }

            // Update UI
            this.updateRecordingUI(false);
            this.stopRecordingTimer();

            console.log('🛑 Recording stopped');
        }
    }

    onDataAvailable(event) {
        if (event.data.size > 0) {
            this.audioChunks.push(event.data);
        }
    }

    async onRecordingStop() {
        try {
            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, {
                type: this.getSupportedMimeType()
            });

            console.log(`📄 Audio blob created: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

            // Process the recorded audio
            await this.processRecordedAudio(audioBlob);

        } catch (error) {
            console.error('Failed to process recorded audio:', error);
            this.showError('Failed to process recorded audio');
        }
    }

    async processRecordedAudio(audioBlob) {
        try {
            // Show processing state
            this.showProcessingState();

            // Create a File object from the blob
            const audioFile = new File([audioBlob], 'recording.webm', {
                type: audioBlob.type
            });

            // Get or create API client
            let apiClient = window.apiClient;
            if (!apiClient) {
                console.log('⚠️ Global API client not available, creating new instance...');
                try {
                    // Try to create a new SAMOAPIClient instance
                    if (typeof SAMOAPIClient !== 'undefined') {
                        apiClient = new SAMOAPIClient();
                        console.log('✅ Created new API client instance');
                    } else {
                        throw new Error('SAMOAPIClient class not available');
                    }
                } catch (createError) {
                    throw new Error(`Unable to create API client: ${createError.message}`);
                }
            }

            // Use API client to transcribe
            if (apiClient && typeof apiClient.transcribeAudio === 'function') {
                console.log('🔄 Sending audio for transcription...');
                const response = await apiClient.transcribeAudio(audioFile);

                if (response.ok) {
                    const result = await response.json();
                    console.log('✅ Transcription successful:', result);

                    // Display results in the UI
                    this.displayTranscriptionResults(result);
                } else {
                    const errorText = await response.text().catch(() => 'Unknown error');
                    throw new Error(`API request failed (${response.status}): ${errorText}`);
                }
            } else {
                throw new Error('API client transcribeAudio method not available');
            }

        } catch (error) {
            console.error('Failed to transcribe audio:', error);

            // Provide specific error messages based on error type
            let userMessage = 'Transcription failed';
            if (error.message.includes('API client not available')) {
                userMessage = 'Voice service unavailable. Please refresh the page and try again.';
            } else if (error.message.includes('Failed to fetch') || error.message.includes('Network')) {
                userMessage = 'Network error. Please check your connection and try again.';
            } else if (error.message.includes('timeout')) {
                userMessage = 'Request timeout. Please try with a shorter recording.';
            } else if (error.message.includes('400')) {
                userMessage = 'Invalid audio format. Please try recording again.';
            } else if (error.message.includes('500')) {
                userMessage = 'Server error. Please try again in a moment.';
            } else {
                userMessage = `Transcription failed: ${error.message}`;
            }

            this.showError(userMessage);

            // Reset processing state on error
            if (window.LayoutManager && window.LayoutManager.isProcessing) {
                window.LayoutManager.endProcessing();
                console.log('🔧 Processing state reset due to transcription error');
            }
        } finally {
            this.hideProcessingState();
        }
    }

    displayTranscriptionResults(result) {
        try {
            // Update text input with transcribed text
            const textInput = document.getElementById('textInput');
            if (textInput) {
                let tx = '';
                if (typeof result === 'string') {
                    tx = result;
                } else if (result?.text) {
                    tx = result.text;
                } else if (result?.transcription && typeof result.transcription === 'string') {
                    tx = result.transcription;
                } else if (result?.transcription?.text) {
                    tx = result.transcription.text;
                }
                if (tx) {
                    textInput.value = tx;
                    console.log('📝 Transcribed text inserted into input');
                }
            }

            // If we have complete analysis results, display them
            if (result.emotion_analysis || result.summary) {
                // Trigger the processing to show results
                if (typeof processTextWithStateManagement === 'function') {
                    processTextWithStateManagement();
                } else if (typeof processText === 'function') {
                    processText();
                }
            }

            // Show success message
            this.showSuccess('Voice successfully transcribed!');

        } catch (error) {
            console.error('Failed to display transcription results:', error);
            this.showError('Failed to display results');
        }
    }

    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/wav'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }

        return 'audio/webm'; // fallback
    }

    updateRecordingUI(isRecording) {
        if (this.recordBtn) {
            this.recordBtn.disabled = isRecording;
            this.recordBtn.innerHTML = isRecording
                ? '<i class="fas fa-microphone me-2"></i>Recording...'
                : '<i class="fas fa-microphone me-2"></i>Record';
        }

        if (this.stopBtn) {
            this.stopBtn.disabled = !isRecording;
        }

        if (this.recordingIndicator) {
            this.recordingIndicator.style.display = isRecording ? 'block' : 'none';
        }

        // Show/hide recording timer
        if (this.recordingTime) {
            this.recordingTime.style.display = isRecording ? 'inline' : 'none';
        }
    }

    startRecordingTimer() {
        this.recordingTimer = setInterval(() => {
            if (this.recordingStartTime && this.recordingTime) {
                const elapsed = Math.floor((Date.now() - this.recordingStartTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                this.recordingTime.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }

    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
        if (this.recordingTime) {
            this.recordingTime.textContent = '0:00';
        }
    }

    showProcessingState() {
        // Use existing layout manager if available
        if (window.LayoutManager && typeof window.LayoutManager.showProcessingState === 'function') {
            // Check if processing is allowed first
            if (!window.LayoutManager.canStartProcessing()) {
                console.warn('⚠️ Cannot show processing state - operation already in progress');
                return false;
            }
            return window.LayoutManager.showProcessingState();
        }
        return true;
    }

    hideProcessingState() {
        // Use existing layout manager if available
        if (window.LayoutManager && typeof window.LayoutManager.showResultsState === 'function') {
            window.LayoutManager.showResultsState();
        }
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showMessage(message, type = 'info') {
        // Create a simple toast notification
        const toast = document.createElement('div');
        toast.className = `toast-notification toast-${type}`;
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 6px;
            color: white;
            font-weight: 500;
            z-index: 10000;
            opacity: 0;
            transition: opacity 0.3s ease;
        `;

        // Set background color based on type
        const colors = {
            success: '#28a745',
            error: '#dc3545',
            info: '#17a2b8'
        };
        toast.style.backgroundColor = colors[type] || colors.info;

        document.body.appendChild(toast);

        // Animate in
        setTimeout(() => toast.style.opacity = '1', 100);

        // Remove after delay
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 3000);
    }

    handleRecordingError(error) {
        let errorMessage = 'Recording failed';
        let helpText = '';

        if (error.name === 'NotAllowedError') {
            errorMessage = 'Microphone access denied';
            helpText = 'Please click the microphone icon in your browser\'s address bar and allow access, then try again.';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'No microphone detected';
            helpText = 'Please connect a microphone to your device and refresh the page.';
        } else if (error.name === 'NotSupportedError') {
            errorMessage = 'Audio recording not supported';
            helpText = 'Please try using a modern browser like Chrome, Firefox, or Safari.';
        } else if (error.name === 'SecurityError') {
            errorMessage = 'Security error - HTTPS required';
            helpText = 'Voice recording requires a secure connection. Please access this page via HTTPS.';
        }

        console.error('Recording error:', error);
        this.showError(`${errorMessage}. ${helpText}`);
        this.updateRecordingUI(false);

        // Reset processing state if error occurs
        if (window.LayoutManager && window.LayoutManager.isProcessing) {
            window.LayoutManager.endProcessing();
        }
    }

    disableRecording(reason) {
        if (this.recordBtn) {
            this.recordBtn.disabled = true;
            this.recordBtn.innerHTML = '<i class="fas fa-microphone-slash me-2"></i>Unavailable';
            this.recordBtn.title = reason;
        }
        if (this.stopBtn) {
            this.stopBtn.disabled = true;
        }
    }
}

// Global voice recorder instance
window.voiceRecorder = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async function() {
    console.log('🎙️ Initializing voice recorder...');
    window.voiceRecorder = new VoiceRecorder();
    await window.voiceRecorder.init();
});