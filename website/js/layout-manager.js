/**
 * Layout State Management Functions
 * Handles transitions between different UI states and progress tracking
 */

// Layout State Management Functions
const LayoutManager = {
    currentState: 'initial', // initial, processing, results

    // Transition to processing state
    showProcessingState() {
        console.log('🔄 Transitioning to processing state...');
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
        this.currentState = 'initial';

        // IMMEDIATELY clear all result content to prevent remnants
        if (typeof clearAllResultContent === 'function') {
            clearAllResultContent();
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

        // Show input layout
        const inputLayout = document.getElementById('inputLayout');
        if (inputLayout) {
            setTimeout(() => {
                inputLayout.classList.remove('d-none');
                inputLayout.style.opacity = '1';
                inputLayout.style.transform = 'translateY(0)';
            }, 350);
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

    // Transition to processing state
    LayoutManager.showProcessingState();

    // Update progress steps
    LayoutManager.updateProgressStep(1, 'active');

    // Call the original processing function
    if (typeof processText === 'function') {
        // Set up a promise to handle the transition to results
        const originalFunc = processText;
        processText().then(() => {
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

    // Reset to initial state
    LayoutManager.resetToInitialState();

    // Call original clear function
    if (typeof clearAll === 'function') {
        clearAll();
    }
};

// Make LayoutManager globally available
window.LayoutManager = LayoutManager;
