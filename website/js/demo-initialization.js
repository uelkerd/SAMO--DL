/**
 * Demo Initialization Script
 * Handles DOM ready events and button event listeners
 */

// Debug: Check if everything is loaded correctly
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Main demo loaded');
    console.log('processText available:', typeof window.processText === 'function');
    console.log('textInput element found:', !!document.getElementById('textInput'));
    console.log('processBtn element found:', !!document.getElementById('processBtn'));

    // Test if the button click works
    const processBtn = document.getElementById('processBtn');
    if (processBtn) {
        console.log('✅ Process button found, adding click listener');
        processBtn.addEventListener('click', function() {
            console.log('🔘 Process button clicked!');
            // Use enhanced state management processing
            if (typeof processTextWithStateManagement === 'function') {
                processTextWithStateManagement();
            } else if (typeof window.processText === 'function') {
                // Fallback to original function
                if (window.LayoutManager?.showProcessingState) {
                    window.LayoutManager.showProcessingState();
                }
                window.processText(true);  // Skip state check since showProcessingState() handles it
            } else {
                console.error('❌ processText function not available');
            }
        });
    } else {
        console.error('❌ Process button not found');
    }

    // Add click listener for Generate button
    const generateBtn = document.getElementById('generateBtn');
    if (generateBtn) {
        console.log('✅ Generate button found, adding click listener');
        generateBtn.addEventListener('click', function() {
            console.log('🔘 Generate button clicked!');
            console.log('🔍 generateSampleText type:', typeof window.generateSampleText);
            console.log('🔍 generateSampleText function:', window.generateSampleText);

            if (typeof window.generateSampleText === 'function') {
                console.log('✅ Calling generateSampleText...');
                window.generateSampleText();
            } else {
                console.error('❌ generateSampleText function not available');
                console.log('🔍 Available functions:', Object.keys(window).filter(key => key.includes('generate')));
            }
        });
    } else {
        console.error('❌ Generate button not found');
    }

    // Add click listener for API Key button
    const apiKeyBtn = document.getElementById('apiKeyBtn');
    if (apiKeyBtn) {
        console.log('✅ API Key button found, adding click listener');
        apiKeyBtn.addEventListener('click', function() {
            console.log('🔘 API Key button clicked!');
            if (typeof window.manageApiKey === 'function') {
                window.manageApiKey();
            } else {
                console.error('❌ manageApiKey function not available');
            }
        });

        // Update button status on page load
        if (typeof updateApiKeyButtonStatus === 'function') {
            updateApiKeyButtonStatus();
        }
    } else {
        console.error('❌ API Key button not found');
    }

    // Add click listener for Clear button
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        console.log('✅ Clear button found, adding click listener');
        clearBtn.addEventListener('click', function() {
            console.log('🔘 Clear button clicked!');
            // Use enhanced state management clearing
            if (typeof clearAllWithStateManagement === 'function') {
                clearAllWithStateManagement();
            } else if (typeof window.clearAll === 'function') {
                // Fallback to original function
                if (window.LayoutManager?.resetToInitialState) {
                    window.LayoutManager.resetToInitialState();
                }
                window.clearAll();
            } else {
                console.error('❌ clearAll function not available');
            }
        });
    } else {
        console.error('❌ Clear button not found');
    }

    // Initialize debug section (hidden by default, can be shown by adding ?showDebug=1 to URL)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('showDebug') === '1') {
        LayoutManager.toggleDebugSection(true);
    }

    // Safety reset to ensure clean processing state
    if (window.LayoutManager?.resetProcessingState) {
        window.LayoutManager.resetProcessingState();
    }

    // Initialize layout to initial state
    if (window.LayoutManager?.resetToInitialState) {
        window.LayoutManager.resetToInitialState();
    }

    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    console.log('✅ Enhanced layout manager initialized');
    console.log('✅ Bootstrap tooltips initialized:', tooltipList.length);
});
