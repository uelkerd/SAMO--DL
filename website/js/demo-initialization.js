/**
 * Demo Initialization Script
 * Handles DOM ready events and button event listeners
 */

// Debug: Check if everything is loaded correctly
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Main demo loaded');
    console.log('SamoDemo available:', typeof window.SamoDemo === 'object');
    console.log('processText available:', typeof window.SamoDemo?.processText === 'function');
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
            } else if (typeof window.SamoDemo?.processText === 'function') {
                // Fallback to original function
                if (window.LayoutManager?.showProcessingState) {
                    window.LayoutManager.showProcessingState();
                }
                window.SamoDemo.processText(true);  // Skip state check since showProcessingState() handles it
            } else {
                console.error('❌ processText function not available');
                alert('Error: Core demo functionality is missing. Please reload the page.');
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
            console.log('🔍 generateSampleText type:', typeof window.SamoDemo?.generateSampleText);
            console.log('🔍 generateSampleText function:', window.SamoDemo?.generateSampleText);

            if (typeof window.SamoDemo?.generateSampleText === 'function') {
                console.log('✅ Calling generateSampleText...');
                window.SamoDemo.generateSampleText();
            } else {
                console.error('❌ generateSampleText function not available');
                console.log('🔍 Available functions:', Object.keys(window.SamoDemo || {}).filter(key => key.includes('generate')));
                alert('Error: Sample text generation is not available. Please reload the page.');
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
            if (typeof window.SamoDemo?.manageApiKey === 'function') {
                window.SamoDemo.manageApiKey();
            } else {
                console.error('❌ manageApiKey function not available');
                alert('Error: API key management is not available. Please reload the page.');
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
            } else if (typeof window.SamoDemo?.clearAll === 'function') {
                // Fallback to original function
                if (window.LayoutManager?.resetToInitialState) {
                    window.LayoutManager.resetToInitialState();
                }
                window.SamoDemo.clearAll();
            } else {
                console.error('❌ clearAll function not available');
                alert('Error: Clear functionality is not available. Please reload the page.');
            }
        });
    } else {
        console.error('❌ Clear button not found');
    }

    // Initialize debug section (hidden by default, can be shown by adding ?showDebug=1 to URL)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('showDebug') === '1') {
        window.LayoutManager?.toggleDebugSection?.(true);
    }

    // Safety reset to ensure clean processing state
    if (window.LayoutManager?.resetProcessingState) {
        window.LayoutManager.resetProcessingState();
    }

    // Initialize layout to initial state
    if (window.LayoutManager?.resetToInitialState) {
        window.LayoutManager.resetToInitialState();
    }

    // Initialize Bootstrap tooltips if Bootstrap is loaded
    let tooltipList = [];
    if (window.bootstrap) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
        console.log('✅ Bootstrap tooltips initialized:', tooltipList.length);
    } else {
        console.warn('⚠️ Bootstrap is not loaded. Tooltips not initialized.');
    }

    console.log('✅ Enhanced layout manager initialized');
});
