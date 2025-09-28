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

    // Dependencies (injected for testability and loose coupling)
    dependencies: {
        clearAllResultContent: null, // Function to clear result content
        updateElement: null, // Function to update element content
        textInput: null, // DOM element reference
        resultsLayout: null, // DOM element reference
        inputLayout: null, // DOM element reference
        inlineMessages: null // Selector for inline messages
    },

    // Initialize with dependencies (dependency injection)
    init(dependencies = {}) {
        console.log('🔧 LayoutManager: Initializing with dependencies...');

        // Set up dependencies with fallbacks to global scope for backward compatibility
        this.dependencies.clearAllResultContent = dependencies.clearAllResultContent || (typeof clearAllResultContent === 'function' ? clearAllResultContent : null);
        this.dependencies.updateElement = dependencies.updateElement || (typeof updateElement === 'function' ? updateElement : null);
        this.dependencies.textInput = dependencies.textInput || document.getElementById('textInput');
        this.dependencies.resultsLayout = dependencies.resultsLayout || document.getElementById('resultsLayout');
        this.dependencies.inputLayout = dependencies.inputLayout || document.getElementById('inputLayout');
        this.dependencies.inlineMessages = dependencies.inlineMessages || '.inline-message';

        console.log('✅ LayoutManager: Dependencies initialized');
    },

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
        if (this.dependencies.clearAllResultContent) {
            this.dependencies.clearAllResultContent();
        }
    },

    // Check if processing is allowed (prevents concurrent operations)
    canStartProcessing() {
        return !this.isProcessing;
    },

    // Start processing (sets guard)
    startProcessing() {
        if (this.isProcessing) {
            // Guard against null processingStartTime to prevent NaN calculations
            if (this.processingStartTime == null) {
                console.warn('⚠️ Missing processingStartTime; forcing reset...');
                this.forceResetProcessing();
                return false;
            }
            // Check if processing has been stuck for too long
            const timeElapsed = Date.now() - this.processingStartTime;
            if (timeElapsed > this.maxProcessingTime) {
                console.warn(`⚠️ Processing stuck for ${timeElapsed/1000}s, forcing reset...`);
                this.forceResetProcessing();
                return false;
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
            console.error('❌ Cannot start processing - operation already in progress');
            console.error('💡 Suggestion: Check if previous processing completed or call resetToInitialState()');
            console.error('🔍 Current state:', this.currentState);
            console.error('🔍 Active requests:', this.activeRequests.size);
            return false; // Fail fast to expose underlying issues
        }

        this.currentState = 'processing';

        // IMMEDIATELY clear all result content to prevent remnants during processing
        if (this.dependencies.clearAllResultContent) {
            this.dependencies.clearAllResultContent();
        }

        // Hide input layout with smooth transition
        if (this.dependencies.inputLayout) {
            this.dependencies.inputLayout.style.opacity = '0';
            this.dependencies.inputLayout.style.transform = 'translateY(-20px)';

            setTimeout(() => {
                this.dependencies.inputLayout.classList.add('d-none');
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
        if (this.dependencies.clearAllResultContent) {
            this.dependencies.clearAllResultContent();
        }

        // Clear text input (using injected dependency)
        if (this.dependencies.textInput) {
            this.dependencies.textInput.value = '';
        }

        // Clear any inline messages (using injected dependency)
        const existingMessages = document.querySelectorAll(this.dependencies.inlineMessages);
        existingMessages.forEach(msg => msg.remove());

        // Reset Processing Information values (using injected dependency)
        if (this.dependencies.updateElement) {
            this.dependencies.updateElement('totalTimeCompact', '-');
            this.dependencies.updateElement('processingStatusCompact', 'Ready');
            this.dependencies.updateElement('modelsUsedCompact', '-');
            this.dependencies.updateElement('avgConfidenceCompact', '-');
        }

        // Hide results layout (using injected dependency)
        if (this.dependencies.resultsLayout) {
            this.dependencies.resultsLayout.style.opacity = '0';
            this.dependencies.resultsLayout.style.transform = 'translateY(20px)';

            setTimeout(() => {
                this.dependencies.resultsLayout.classList.add('d-none');
            }, 300);
        }

        // Show input layout immediately (using injected dependency)
        if (this.dependencies.inputLayout) {
            console.log('🔄 LayoutManager: Showing input layout');
            this.dependencies.inputLayout.classList.remove('d-none'); // Remove Bootstrap's hide class
            this.dependencies.inputLayout.style.opacity = '1';
            this.dependencies.inputLayout.style.transform = 'translateY(0)';
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
        if (this.dependencies.resultsLayout) {
            this.dependencies.resultsLayout.classList.remove('d-none');
            this.dependencies.resultsLayout.style.opacity = '1';

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
            stepElement.classList.remove('pending', 'active', 'completed', 'error');
            stepIcon.classList.remove('pending', 'active', 'completed', 'error');

            // Add new state
            stepElement.classList.add(state);
            stepIcon.classList.add(state);
        } else {
            // Add warning for missing elements to improve debuggability
            if (!stepElement) {
                console.warn(`Progress step element #step${stepNumber} not found`);
            }
            if (!stepIcon) {
                console.warn(`Progress step icon #step${stepNumber}-icon not found`);
            }
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
                inputLayout.classList.add('d-none'); // Use CSS class instead of direct style manipulation
            }, 300);
        }

    // Show processing layout
    const processingLayout = document.getElementById('processingLayout');
    if (processingLayout) {
        processingLayout.classList.remove('d-none'); // Use CSS class instead of direct style manipulation
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
        const maybe = processText(true);  // Skip state check since we handle it here
        const onDone = () => {
            // Minimal delay to ensure smooth UI transition after processing completes
            setTimeout(() => {
                LayoutManager.showResultsState();
                LayoutManager.updateProgressStep(4, 'completed');
            }, 50); // Reduced from 1000ms to 50ms for better perceived performance
        };
        if (maybe && typeof maybe.then === 'function') {
            maybe.then(onDone).catch((error) => {
                console.error('Processing error:', error);
                LayoutManager.resetToInitialState();
            });
        } else {
            onDone();
        }
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

// Initialize LayoutManager when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🏗️ Initializing LayoutManager...');
    LayoutManager.init(); // Initialize with default dependencies (falls back to global scope)
    console.log('✅ LayoutManager initialized');
});

// Make LayoutManager globally available
window.LayoutManager = LayoutManager;
