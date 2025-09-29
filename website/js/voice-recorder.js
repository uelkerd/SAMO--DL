/**
 * Voice Recording Module for SAMO Demo
 * Handles microphone access, audio recording, and integration with the demo interface
 *
 * @typedef {Object} TranscriptionResult
 * @property {string} [text] - The transcribed text content
 * @property {number} [confidence] - Confidence score (0-1) for the transcription
 * @property {any} [error] - Error information if transcription failed
 * @property {Object} [transcription] - Alternative transcription object format
 * @property {string} [transcription.text] - Transcribed text in nested format
 *
 * @typedef {Object} ApiClientInterface
 * @property {function(File): Promise<TranscriptionResult>} transcribeAudio - Transcribes audio file to text
 *
 * @param {ApiClientInterface|null} apiClient - API client implementing the transcription interface
 */
class VoiceRecorder {
    constructor(apiClient = null) {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.stream = null;
        this.recordingStartTime = null;
        this.recordingTimer = null;

        // Dependencies (injected)
        this.apiClient = apiClient;

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
        // Ignore if a recording is already in progress
        if (this.isRecording) {
            console.warn('Recording already in progress');
            return;
        }

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
            // Determine MIME type with proper fallback handling and validation
            let chosenType;
            let usedFallback = false;

            if (this.mediaRecorder && typeof this.mediaRecorder.mimeType === 'string') {
                chosenType = this.mediaRecorder.mimeType;
            } else {
                chosenType = this.getSupportedMimeType();
                usedFallback = true;
                console.warn('⚠️ MediaRecorder.mimeType not available, falling back to getSupportedMimeType()');
            }

            // Validate that the chosen type is actually supported
            if (!this.isValidMimeType(chosenType)) {
                console.warn(`⚠️ Chosen MIME type '${chosenType}' is not supported, using safe default`);
                chosenType = 'audio/webm'; // Safe default
                usedFallback = true;
            }

            // Create audio blob with validated MIME type
            const audioBlob = new Blob(this.audioChunks, { type: chosenType });

            // Enhanced logging with MIME type decision details
            const fallbackInfo = usedFallback ? ' (fallback used)' : ' (from MediaRecorder)';
            console.log(`📄 Audio blob created: ${audioBlob.size} bytes, type: ${audioBlob.type}${fallbackInfo}`);

            // Process the recorded audio
            await this.processRecordedAudio(audioBlob);

        } catch (error) {
            console.error('Failed to process recorded audio:', error);
            this.showError('Failed to process recorded audio');
        }
    }

    /**
     * Helper to get file extension from MIME type
     */
    getExtensionFromMimeType(mimeType) {
        // Strip codec parameters to get base MIME type (e.g., "audio/webm;codecs=opus" -> "audio/webm")
        const baseType = String(mimeType || '').split(';', 1)[0].toLowerCase();
        const mimeToExt = {
            'audio/webm': 'webm',
            'audio/mp4': 'mp4',
            'audio/wav': 'wav',
            'audio/mpeg': 'mp3',
            'audio/ogg': 'ogg'
        };
        return mimeToExt[baseType] || 'audio';
    }

    async processRecordedAudio(audioBlob) {
        let processingShown = false;
        try {
            // Show processing state
            processingShown = this.showProcessingState() === true;
            if (!processingShown) {
                return;
            }

            // Create a File object from the blob with correct extension
            const extension = this.getExtensionFromMimeType(audioBlob.type);
            const audioFile = new File([audioBlob], `recording.${extension}`, {
                type: audioBlob.type
            });

            // Use injected API client (dependency injection)
            const { apiClient } = this;
            if (!apiClient) {
                throw new Error('API client not available for transcription. Please ensure apiClient is properly initialized.');
            }

            // Validate API client method exists
            if (typeof apiClient.transcribeAudio !== 'function') {
                throw new Error('API client transcribeAudio method not available');
            }

            // Validate File argument
            if (!(audioFile instanceof File)) {
                throw new Error(`Invalid audio file argument: expected File instance, got ${typeof audioFile}`);
            }

            console.log('🔄 Sending audio for transcription...');
            const result = await apiClient.transcribeAudio(audioFile);

            console.log('✅ Transcription successful:', result);

            // Validate result shape before processing
            this.validateTranscriptionResult(result);

            // Display results in the UI
            this.displayTranscriptionResults(result);

        } catch (error) {
            console.error('Failed to transcribe audio:', error);

            // Provide specific error messages based on error type
            let userMessage = 'Transcription failed';
            if (/API client/i.test(error.message)) {
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

            // Re-throw the error so tests can catch it
            throw error;
        } finally {
            if (processingShown) {
                this.hideProcessingState();
            }
        }
    }

    /**
     * Validate transcription result shape and handle malformed responses
     * @param {any} result - The result from transcribeAudio API call
     * @throws {Error} If result is malformed or indicates failure
     */
    validateTranscriptionResult(result) {
        // Check if result is null, undefined, or not an object
        if (result === null || result === undefined) {
            console.error('❌ Transcription result is null or undefined');
            throw new Error('Transcription service returned no result');
        }

        if (typeof result !== 'object') {
            console.error('❌ Transcription result is not an object:', typeof result, result);
            throw new Error(`Transcription service returned invalid result type: ${typeof result}`);
        }

        // Check for explicit error indicators
        if (result.error) {
            console.error('❌ Transcription service returned error:', result.error);
            throw new Error(`Transcription failed: ${result.error}`);
        }

        // Check for success field (if present)
        if (result.success === false) {
            console.error('❌ Transcription service indicated failure:', result);
            throw new Error('Transcription service reported failure');
        }

        // Validate that we have some form of transcription text
        const hasText = result.text ||
                       (result.transcription && (typeof result.transcription === 'string' || result.transcription.text)) ||
                       (typeof result === 'string');

        if (!hasText) {
            console.error('❌ Transcription result missing text content:', result);
            console.error('Raw response for debugging:', JSON.stringify(result, null, 2));
            throw new Error('Transcription completed but no text was returned');
        }

        // Log successful validation
        console.log('✅ Transcription result validation passed');
    }

    /**
     * Extract transcription text from API response
     * Handles multiple possible response formats for backward compatibility
     */
    extractTranscriptionText(result) {
        // Standardize transcription text extraction with clear priority
        if (typeof result === 'string') {
            return result;
        }

        // Preferred format: result.transcription.text (standardized API response)
        if (result?.transcription?.text) {
            return result.transcription.text;
        }

        // Alternative formats for backward compatibility
        if (result?.text) {
            console.warn('⚠️ Using deprecated result.text format - consider updating API to use result.transcription.text');
            return result.text;
        }

        if (result?.transcription && typeof result.transcription === 'string') {
            console.warn('⚠️ Using deprecated result.transcription format - consider updating API to use result.transcription.text');
            return result.transcription;
        }

        console.error('❌ Unable to extract transcription text from response:', result);
        return null;
    }

    displayTranscriptionResults(result) {
        try {
            // Update text input with transcribed text
            const textInput = document.getElementById('textInput');
            if (textInput) {
                const transcriptionText = this.extractTranscriptionText(result);
                if (transcriptionText) {
                    textInput.value = transcriptionText;
                    console.log('📝 Transcribed text inserted into input');
                } else {
                    console.error('❌ No transcription text found in result');
                    this.showError('Failed to extract transcription text');
                }
            }

            // If we have complete analysis results, display them directly
            if (result.emotion_analysis || result.summary) {
                // Use existing results to update the UI, avoid redundant processing
                if (typeof displayAnalysisResults === 'function') {
                    displayAnalysisResults(result.emotion_analysis, result.summary);
                } else {
                    // Fallback: directly update UI elements if displayAnalysisResults is not defined
                    if (result.emotion_analysis && document.getElementById('emotionAnalysis')) {
                        // Pretty-print JSON for better readability
                        document.getElementById('emotionAnalysis').textContent = JSON.stringify(result.emotion_analysis, null, 2);
                    }
                    if (result.summary && document.getElementById('summary')) {
                        document.getElementById('summary').textContent = result.summary;
                    }
                }
            }

            // Show success message
            this.showSuccess('Voice successfully transcribed!');

        } catch (error) {
            console.error('Failed to display transcription results:', error);
            this.showError('Failed to display results');
        }
    }

    /**
     * Validate that a MIME type is supported and not null/undefined
     * @param {string} mimeType - The MIME type to validate
     * @returns {boolean} True if the MIME type is valid and supported
     */
    isValidMimeType(mimeType) {
        return mimeType &&
               typeof mimeType === 'string' &&
               mimeType.trim().length > 0 &&
               MediaRecorder.isTypeSupported(mimeType);
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
                console.log(`✅ Found supported MIME type: ${type}`);
                return type;
            }
        }

        console.warn('⚠️ No supported MIME types found, using fallback');
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
            // Check if processing is allowed first (if available)
            if (typeof window.LayoutManager.canStartProcessing === 'function' &&
                !window.LayoutManager.canStartProcessing()) {
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
        if (window.NotificationManager) {
            window.NotificationManager.success(message);
        } else {
            console.warn('⚠️ NotificationManager not available, falling back to console');
            console.log('✅', message);
        }
    }

    showError(message) {
        if (window.NotificationManager) {
            window.NotificationManager.error(message);
        } else {
            console.error('❌', message);
        }
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

// Export the VoiceRecorder class for testing
window.VoiceRecorder = VoiceRecorder;

// Global voice recorder instance
window.voiceRecorder = null;

/**
 * Initialize voice recorder with dependency injection support
 * @param {Object} options - Initialization options
 * @param {Object} options.apiClient - Optional API client instance (for testing)
 * @param {Object} options.apiClientManager - Optional API client manager (for testing)
 * @param {Object} options.config - Optional configuration overrides
 */
async function initializeVoiceRecorder(options = {}) {
    const {
        apiClient = null,
        apiClientManager = window.ApiClientManager,
        config = window.SAMO_CONFIG
    } = options;

    console.log('🎙️ Initializing voice recorder...');

    try {
        let resolvedApiClient = apiClient;

        // If no API client provided, try to get one
        if (!resolvedApiClient) {
            if (!apiClientManager) {
                throw new Error('API client manager not available');
            }

            // Use hybrid approach: try event-based first, fallback to polling
            const initOptions = {
                timeoutMs: config?.API?.TIMEOUTS?.API_CLIENT_INIT || 5000,
                useEventBased: true
            };

            resolvedApiClient = await apiClientManager.waitForApiClient(initOptions);
        }

        // Create voice recorder with resolved API client
        window.voiceRecorder = new VoiceRecorder(resolvedApiClient);
        await window.voiceRecorder.init();
        console.log('✅ Voice recorder initialized with API client');

        // Notify that voice recorder is ready
        const event = new CustomEvent('voiceRecorderReady', {
            detail: { voiceRecorder: window.voiceRecorder }
        });
        window.dispatchEvent(event);

    } catch (error) {
        console.error('❌ Failed to initialize voice recorder:', error);

        // Fallback: create without API client (will show clear error if transcription attempted)
        window.voiceRecorder = new VoiceRecorder(null);
        await window.voiceRecorder.init();
        console.warn('⚠️ Voice recorder initialized without API client - transcription will fail');

        // Notify that voice recorder is ready (but with limited functionality)
        const event = new CustomEvent('voiceRecorderReady', {
            detail: {
                voiceRecorder: window.voiceRecorder,
                limitedFunctionality: true,
                error: error.message
            }
        });
        window.dispatchEvent(event);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async function() {
    // Check if API client utilities are available
    if (!window.ApiClientManager) {
        console.warn('⚠️ ApiClientManager not available, using fallback initialization');
        await initializeVoiceRecorder({ apiClientManager: null });
        return;
    }

    // Use the new initialization function
    await initializeVoiceRecorder();
});
