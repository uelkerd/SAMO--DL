/**
 * Layout State Management Functions
 * Handles transitions between different UI states and progress tracking
 */

// Layout State Management Functions
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
        console.log('✅ Force reset completed');
    },

    // Transition to processing state
    showProcessingState() {
        console.log('🔄 Transitioning to processing state...');

        // Check if processing is allowed
        if (!this.startProcessing()) {
            console.warn('⚠️ Cannot start processing - operation already in progress');
            // Try emergency reset and retry once
            console.warn('🔄 Attempting emergency reset and retry...');
            this.emergencyReset();
            if (!this.startProcessing()) {
                console.error('❌ Emergency reset failed - processing still blocked');
                return false;
            }
            console.log('✅ Emergency reset successful - processing can proceed');
        }

        this.currentState = 'processing';

        // IMMEDIATELY clear all result content to prevent remnants during processing
        if (typeof clearAllResultContent === 'function') {
            clearAllResultContent();
        }

        // Hide input layout with smooth transition
        const inputLayout = document.getElementById('inputLayout');
        if (inputLayout) {
            inputLayout.style.opacity = '0';
            inputLayout.style.transform = 'translateY(-20px)';

            setTimeout(() => {
                inputLayout.classList.add('d-none');
            }, 300);
        }

        // Show loading in results area
        this.showLoadingState();
    },

    // Transition to results state
    showResultsState() {
        console.log('✅ Transitioning to results state...');
        this.currentState = 'results';

        // End processing since we've reached results
        this.endProcessing();

        // Hide loading
        this.hideLoadingState();

        // Show results layout with smooth transition
        const resultsLayout = document.getElementById('resultsLayout');
        if (resultsLayout) {
            resultsLayout.classList.remove('d-none');
            resultsLayout.style.opacity = '0';
            resultsLayout.style.transform = 'translateY(20px)';

            // Animate in
            setTimeout(() => {
                resultsLayout.style.opacity = '1';
                resultsLayout.style.transform = 'translateY(0)';
            }, 100);
        }

        // Sync processing info data
        this.syncProcessingInfo();
    },

    // Return to initial state
    resetToInitialState() {
        console.log('🔄 Resetting to initial state...');

        // Cancel any active requests first
        this.cancelActiveRequests();

        // Force end processing to remove lock (no matter what state we're in)
        this.isProcessing = false;
        this.processingStartTime = null;
        this.activeRequests.clear();
        this.currentState = 'initial';
        console.log('🔧 Processing state forcibly reset');

        // IMMEDIATELY clear all result content to prevent remnants
        if (typeof clearAllResultContent === 'function') {
            clearAllResultContent();
        }

        // Clear text input
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = '';
        }

        // Clear any inline messages
        const existingMessages = document.querySelectorAll('.inline-message');
        existingMessages.forEach(msg => msg.remove());

        // Reset Processing Information values
        if (typeof updateElement === 'function') {
            updateElement('totalTimeCompact', '-');
            updateElement('processingStatusCompact', 'Ready');
            updateElement('modelsUsedCompact', '-');
            updateElement('avgConfidenceCompact', '-');
        }

        // Hide results layout
        const resultsLayout = document.getElementById('resultsLayout');
        if (resultsLayout) {
            resultsLayout.style.opacity = '0';
            resultsLayout.style.transform = 'translateY(20px)';

            setTimeout(() => {
                resultsLayout.classList.add('d-none');
            }, 300);
        }

        // Show input layout immediately
        const inputLayout = document.getElementById('inputLayout');
        if (inputLayout) {
            console.log('🔄 LayoutManager: Showing input layout');
            inputLayout.classList.remove('d-none');
            inputLayout.style.display = 'block'; // Force display
            inputLayout.style.opacity = '1';
            inputLayout.style.transform = 'translateY(0)';
            console.log('✅ LayoutManager: Input layout should be visible');
        } else {
            console.error('❌ LayoutManager: inputLayout element not found');
        }

        // Hide loading
        this.hideLoadingState();

        // Reset progress steps
        this.resetProgressSteps();
    },

    // Show loading state in results area
    showLoadingState() {
        const resultsLayout = document.getElementById('resultsLayout');
        if (resultsLayout) {
            resultsLayout.classList.remove('d-none');
            resultsLayout.style.opacity = '1';

            // Show only loading spinner initially
            const loadingSection = document.getElementById('loadingSection');
            if (loadingSection) {
                loadingSection.style.display = 'block';
            }
        }
    },

    // Hide loading state
    hideLoadingState() {
        const loadingSection = document.getElementById('loadingSection');
        if (loadingSection) {
            loadingSection.style.display = 'none';
        }
    },

    // Sync processing info between original and compact versions
    syncProcessingInfo() {
        const mappings = [
            ['totalTime', 'totalTimeCompact'],
            ['processingStatus', 'processingStatusCompact'],
            ['modelsUsed', 'modelsUsedCompact'],
            ['avgConfidence', 'avgConfidenceCompact']
        ];

        mappings.forEach(([original, compact]) => {
            const originalEl = document.getElementById(original);
            const compactEl = document.getElementById(compact);

            if (originalEl && compactEl) {
                compactEl.textContent = originalEl.textContent;
            }
        });
    },

    // Update progress steps
    updateProgressStep(stepNumber, state) {
        // Update both horizontal and vertical progress indicators
        const stepElement = document.getElementById(`step${stepNumber}`);
        const stepIcon = document.getElementById(`step${stepNumber}-icon`);

        if (stepElement && stepIcon) {
            // Remove existing state classes
            stepElement.classList.remove('active', 'completed', 'error');
            stepIcon.classList.remove('pending', 'active', 'completed', 'error');

            // Add new state
            stepElement.classList.add(state);
            stepIcon.classList.add(state);
        }
    },

    // Reset progress steps to initial state
    resetProgressSteps() {
        for (let i = 1; i <= 4; i++) {
            this.updateProgressStep(i, 'pending');
        }
    },

    // Toggle debug section visibility
    toggleDebugSection(show = null) {
        const debugSection = document.getElementById('debugTestSection');
        const toggleBtn = document.getElementById('debugToggleBtn');

        if (debugSection) {
            let isVisible;

            if (show === null) {
                // Toggle current state
                isVisible = !debugSection.classList.contains('d-none');
                if (isVisible) {
                    debugSection.classList.add('d-none');
                } else {
                    debugSection.classList.remove('d-none');
                }
                isVisible = !isVisible;
            } else if (show) {
                debugSection.classList.remove('d-none');
                isVisible = true;
            } else {
                debugSection.classList.add('d-none');
                isVisible = false;
            }

            // Update toggle button text
            if (toggleBtn) {
                const icon = toggleBtn.querySelector('.material-icons');
                const textNode = toggleBtn.lastChild;

                if (isVisible) {
                    textNode.textContent = ' Hide Debug';
                    icon.textContent = 'bug_report';
                    toggleBtn.classList.remove('btn-outline-secondary');
                    toggleBtn.classList.add('btn-warning');
                } else {
                    textNode.textContent = ' Show Debug';
                    icon.textContent = 'bug_report';
                    toggleBtn.classList.remove('btn-warning');
                    toggleBtn.classList.add('btn-outline-secondary');
                }
            }
        }
    }
};

// Enhanced processing function with state management
window.processTextWithStateManagement = function() {
    console.log('🚀 Processing with enhanced state management...');

    // Check if processing is allowed
    if (!LayoutManager.canStartProcessing()) {
        console.warn('⚠️ Processing blocked - operation already in progress');
        return;
    }

    // Start processing (sets guard) - don't call showProcessingState() here
    if (!LayoutManager.startProcessing()) {
        console.error('❌ Failed to start processing - operation already in progress');
        return;
    }

    // Set processing state and update UI
    LayoutManager.currentState = 'processing';
    
    // IMMEDIATELY clear all result content to prevent remnants during processing
    if (typeof clearAllResultContent === 'function') {
        clearAllResultContent();
    }

    // Hide input layout with smooth transition
    const inputLayout = document.getElementById('inputLayout');
    if (inputLayout) {
        inputLayout.style.opacity = '0';
        setTimeout(() => {
            inputLayout.style.display = 'none';
        }, 300);
    }

    // Show processing layout
    const processingLayout = document.getElementById('processingLayout');
    if (processingLayout) {
        processingLayout.style.display = 'block';
        processingLayout.style.opacity = '0';
        setTimeout(() => {
            processingLayout.style.opacity = '1';
        }, 50);
    }

    // Update progress steps
    LayoutManager.updateProgressStep(1, 'active');

    // Call the original processing function
    if (typeof processText === 'function') {
        // Set up a promise to handle the transition to results
        const originalFunc = processText;
        processText(true).then(() => {  // Skip state check since we handle it here
            // After processing completes, show results state
            setTimeout(() => {
                LayoutManager.showResultsState();
                LayoutManager.updateProgressStep(4, 'completed');
            }, 1000);
        }).catch((error) => {
            console.error('Processing error:', error);
            LayoutManager.resetToInitialState();
        });
    }
};

// Enhanced clear function with state management
window.clearAllWithStateManagement = function() {
    console.log('🧹 Clearing with enhanced state management...');

    // Reset to initial state using LayoutManager (this should handle everything)
    LayoutManager.resetToInitialState();

    // Call original clear function if available
    if (typeof clearAll === 'function') {
        clearAll();
    }
};

// Make LayoutManager globally available
window.LayoutManager = LayoutManager;
