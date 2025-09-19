/**
 * LayoutManager Module for Testing
 * Extracts the LayoutManager object from layout-manager.js for testing
 */

// LayoutManager object extracted from layout-manager.js
const LayoutManager = {
    currentState: 'initial', // initial, processing, results
    isProcessing: false, // Processing guard to prevent concurrent operations
    activeRequests: new Set(), // Track active API requests
    processingStartTime: null, // Track when processing started
    maxProcessingTime: 120000, // Maximum processing time (2 minutes) before auto-reset

    // Safety reset to ensure clean state on page load
    resetProcessingState() {
        console.log('🔄 Safety reset: clearing processing state...');
        this.isProcessing = false;
        this.activeRequests.clear();
        this.currentState = 'initial';
    },

    // Emergency reset if processing gets stuck (with timeout)
    emergencyReset() {
        console.warn('🚨 Emergency reset: processing state appears stuck, forcing reset...');
        this.isProcessing = false;
        this.activeRequests.clear();
        this.currentState = 'initial';
        // Also clear any UI elements that might be stuck
        if (typeof clearAllResultContent === 'function') {
            clearAllResultContent();
        }
    },

    // Check if processing is allowed (prevents concurrent operations)
    canStartProcessing() {
        return !this.isProcessing;
    },

    // Start processing (sets guard)
    startProcessing() {
        if (this.isProcessing) {
            // Check if processing has been stuck for too long
            const timeElapsed = Date.now() - this.processingStartTime;
            if (timeElapsed > this.maxProcessingTime) {
                console.warn(`⚠️ Processing stuck for ${timeElapsed/1000}s, forcing reset...`);
                this.forceResetProcessing();
            } else {
                console.warn('⚠️ Processing already in progress, ignoring request');
                console.warn('⚠️ Current state:', this.currentState);
                console.warn('⚠️ Active requests:', this.activeRequests.size);
                console.warn(`⚠️ Time elapsed: ${timeElapsed/1000}s`);
                return false;
            }
        }
        this.isProcessing = true;
        this.processingStartTime = Date.now();
        this.activeRequests.clear();
        console.log('🚀 Processing started - locked for concurrent operations');
        return true;
    },

    // End processing (removes guard)
    endProcessing() {
        this.isProcessing = false;
        this.processingStartTime = null;
        this.activeRequests.clear();
        console.log('✅ Processing completed - ready for new operations');
    },

    // Cancel all active requests
    cancelActiveRequests() {
        console.log(`🚫 Cancelling ${this.activeRequests.size} active requests...`);
        for (const controller of this.activeRequests) {
            if (controller && typeof controller.abort === 'function') {
                controller.abort();
            }
        }
        this.activeRequests.clear();
    },

    // Add request controller for tracking
    addActiveRequest(controller) {
        if (controller) {
            this.activeRequests.add(controller);
            console.log(`📡 Added request to tracking (${this.activeRequests.size} active)`);
        }
    },

    // Remove request controller
    removeActiveRequest(controller) {
        if (this.activeRequests.delete(controller)) {
            console.log(`📡 Removed request from tracking (${this.activeRequests.size} remaining)`);
        }
    },

    // Force cancel all active requests immediately
    forceResetProcessing() {
        console.warn('🚨 Force resetting processing state and cancelling all requests...');
        this.cancelActiveRequests();
        this.isProcessing = false;
        this.processingStartTime = null;
        this.currentState = 'initial';
        console.log('✅ Processing force reset completed');
    },

    // Show processing state with proper transitions
    showProcessingState() {
        console.log('📺 Transitioning to processing state...');

        if (!this.startProcessing()) {
            console.error('❌ Cannot start processing - already in progress');
            return false;
        }

        this.currentState = 'processing';

        // Hide input layout
        const inputLayout = document.getElementById('inputLayout');
        if (inputLayout) {
            inputLayout.style.display = 'none';
        }

        // Show results layout with loading state
        const resultsLayout = document.getElementById('resultsLayout');
        if (resultsLayout) {
            resultsLayout.classList.remove('d-none');
            resultsLayout.style.display = 'block';

            // Ensure loading section is visible
            const loadingSection = document.getElementById('loadingSection');
            if (loadingSection) {
                loadingSection.style.display = 'block';
            }
        }

        console.log('✅ Processing state transition complete');
        return true;
    },

    // Show results and hide loading
    showResults() {
        console.log('📊 Showing results...');

        this.currentState = 'results';

        // Hide loading section
        const loadingSection = document.getElementById('loadingSection');
        if (loadingSection) {
            loadingSection.style.display = 'none';
        }

        // Show result sections
        const emotionResults = document.getElementById('emotionResults');
        const summarizationResults = document.getElementById('summarizationResults');

        if (emotionResults) {
            emotionResults.classList.remove('result-section-hidden');
            emotionResults.classList.add('result-section-visible');
        }

        if (summarizationResults) {
            summarizationResults.classList.remove('result-section-hidden');
            summarizationResults.classList.add('result-section-visible');
        }

        console.log('✅ Results display complete');
    },

    // Reset to initial state
    resetToInitialState() {
        console.log('🔄 Resetting to initial state...');

        // Cancel any active requests first
        this.forceResetProcessing();

        this.currentState = 'initial';

        // Show input layout
        const inputLayout = document.getElementById('inputLayout');
        if (inputLayout) {
            inputLayout.style.display = 'block';
            inputLayout.classList.remove('d-none');
        }

        // Hide results layout
        const resultsLayout = document.getElementById('resultsLayout');
        if (resultsLayout) {
            resultsLayout.classList.add('d-none');
            resultsLayout.style.display = 'none';
        }

        // Hide result sections
        const emotionResults = document.getElementById('emotionResults');
        const summarizationResults = document.getElementById('summarizationResults');

        if (emotionResults) {
            emotionResults.classList.add('result-section-hidden');
            emotionResults.classList.remove('result-section-visible');
        }

        if (summarizationResults) {
            summarizationResults.classList.add('result-section-hidden');
            summarizationResults.classList.remove('result-section-visible');
        }

        // Clear text input
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = '';
        }

        console.log('✅ Reset to initial state complete');
    },

    // Toggle debug section visibility
    toggleDebugSection() {
        const debugSection = document.getElementById('debugTestSection');
        if (debugSection) {
            const isHidden = debugSection.classList.contains('d-none');
            debugSection.classList.toggle('d-none', !isHidden);

            const button = document.getElementById('debugToggleBtn');
            if (button) {
                button.textContent = isHidden ? 'Hide Debug' : 'Show Debug';
            }
        }
    }
};

module.exports = LayoutManager;