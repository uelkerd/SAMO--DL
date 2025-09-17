/**
 * Comprehensive Demo for SAMO Deep Learning Platform
 * Demonstrates Whisper (transcription), T5 (summarization), and SAMO DeBERTa v3 Large (emotion detection)
 */

class ComprehensiveDemo {
    constructor() {
        this.apiClient = new SAMOAPIClient();
        this.uiController = new UIController();
        this.chartUtils = new ChartUtils();
    }

    async processCompleteWorkflow(audioFile, text) {
        const startTime = Date.now();
        const results = {
            transcription: null,
            summary: null,
            emotions: null,
            modelsUsed: [],
            processingTime: 0
        };

        try {
            this.uiController.showLoading();
            this.uiController.updateProgressStep('step1', 'active');

            // Step 1: Transcribe audio if provided
            if (audioFile) {
                try {
                    this.uiController.updateProgressStep('step2', 'active');
                    results.processingStartTime = Date.now();
                    results.transcription = await this.apiClient.transcribe(audioFile);
                    results.modelsUsed.push('SAMO Whisper');
                    this.uiController.updateProgressStep('step2', 'completed');
                    this.uiController.showTranscriptionResults(results.transcription);
                } catch (error) {
                    console.error('Transcription failed:', error);
                    this.uiController.updateProgressStep('step2', 'error');
                    // Continue without transcription - voice processing is currently unavailable
                }
            } else {
                // No audio provided; mark step as completed for text-only flow
                this.uiController.updateProgressStep('step1', 'completed');
            }

            // Step 2 & 3: Analyze text (both summarization and emotion detection in one call)
            let currentText = text;
            if (results.transcription && results.transcription.text) {
                currentText = results.transcription.text;
            }
            
            // Validate that we have text to process
            if (!audioFile && (!currentText || !currentText.trim())) {
                throw new Error('Please provide audio or text to process.');
            }

            if (currentText) {
                try {
                    this.uiController.updateProgressStep('step3', 'active');
                    this.uiController.updateProgressStep('step4', 'active');
                    
                    // Call the unified analysis endpoint that returns both summary and emotions
                    results.processingStartTime = Date.now();
                    const analysisResponse = await this.apiClient.analyzeJournal(currentText);
                    
                    // Extract summary results
                    if (analysisResponse.summary) {
                        results.summary = analysisResponse.summary;
                        results.modelsUsed.push('SAMO T5');
                        this.uiController.updateProgressStep('step3', 'completed');
                        this.uiController.showSummaryResults(results.summary);
                    }
                    
                    // Extract emotion results
                    if (analysisResponse.emotions) {
                        results.emotions = analysisResponse.emotions;
                        results.modelsUsed.push('SAMO DeBERTa v3 Large');
                        this.uiController.updateProgressStep('step4', 'completed');
                        this.uiController.showEmotionResults(results.emotions);
                        
                        // Display detailed model analysis - WORKING SOLUTION FROM SIMPLE DEMO
                        console.log('🔍 About to update detailed model analysis with:', {
                            emotions: results.emotions,
                            summary: results.summary
                        });
                        
                        // Use the EXACT same working function from simple demo
                        updateDetailedAnalysisFromSimpleDemo(results.emotions, results.summary);
                    }
                } catch (error) {
                    console.error('Text analysis failed:', error);
                    this.uiController.updateProgressStep('step3', 'error');
                    this.uiController.updateProgressStep('step4', 'error');
                    // Continue without analysis - don't throw error
                }
            }

            // Complete input step
            this.uiController.updateProgressStep('step1', 'completed');

            // Finalize results
            results.processingTime = Date.now() - startTime;
            this.uiController.updateProcessingInfo(results);
            this.uiController.hideLoading();
            this.uiController.showResults();

        } catch (error) {
            console.error('Processing failed:', error);
            this.uiController.hideLoading();
            this.uiController.showError(`Processing failed: ${error.message}`);
        }
    }
}

// Inline error display function to replace alert dialogs
function showInlineError(message, targetElementId) {
    // Remove any existing error messages
    const existingErrors = document.querySelectorAll('.inline-error-message');
    existingErrors.forEach(error => error.remove());

    // Create error message element
    const errorDiv = document.createElement('div');
    errorDiv.className = 'inline-error-message alert alert-danger mt-2';
    errorDiv.setAttribute('role', 'alert');
    errorDiv.style.cssText = 'animation: fadeIn 0.3s ease-in;';

    // Create strong element for emphasis
    const strongEl = document.createElement('strong');
    strongEl.textContent = 'Error: ';

    // Add content safely
    errorDiv.appendChild(strongEl);
    errorDiv.appendChild(document.createTextNode(message));

    // Insert error message after target element
    const targetElement = document.getElementById(targetElementId);
    if (targetElement) {
        targetElement.parentNode.insertBefore(errorDiv, targetElement.nextSibling);
    } else {
        // Fallback: append to body
        document.body.appendChild(errorDiv);
    }

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.remove();
        }
    }, 5000);
}

// Clear All functionality
function clearAll() {
    console.log('🧹 Clearing all inputs and results...');
    
    // Clear text input
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.value = '';
    }
    
    // Clear audio file input
    const audioFileInput = document.getElementById('audioFile');
    if (audioFileInput) {
        audioFileInput.value = '';
    }
    
    // Hide results
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'none';
    }
    
    // Reset progress steps
    const steps = ['step1', 'step2', 'step3', 'step4'];
    steps.forEach(stepId => {
        const step = document.getElementById(stepId);
        if (step) {
            step.className = 'progress-step-vertical';
            const icon = document.getElementById(stepId + '-icon');
            if (icon) {
                icon.className = 'step-icon-small pending';
            }
        }
    });
    
    console.log('✅ All inputs and results cleared');
}

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', function() {
    try {
        console.log('🚀 Initializing SAMO Demo...');
        
        // Debug Material Icons loading
        console.log('🔍 Checking Material Icons...');
        const testIcon = document.createElement('span');
        testIcon.className = 'material-icons';
        testIcon.textContent = 'check';
        document.body.appendChild(testIcon);
        const computedStyle = window.getComputedStyle(testIcon);
        console.log('Material Icons computed style:', computedStyle.fontFamily);
        if (computedStyle.fontFamily.includes('Material Icons')) {
            console.log('✅ Material Icons are loaded correctly');
        } else {
            console.warn('⚠️ Material Icons may not be loaded properly');
        }
        document.body.removeChild(testIcon);
        
        // Pure HTML/CSS charts - no external dependencies needed!
        console.log('✅ Pure HTML/CSS charts ready - no external dependencies!');

        // Bind clear button
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', clearAll);
            console.log('✅ Clear button bound');
        }

        // Bind process button
        const processBtn = document.getElementById('processBtn');
        if (processBtn) {
            processBtn.addEventListener('click', async function() {
                console.log('🚀 Processing input...');
                
                const textInput = document.getElementById('textInput');
                const audioFileInput = document.getElementById('audioFile');
                
                const text = textInput ? textInput.value.trim() : '';
                const audioFile = audioFileInput ? audioFileInput.files[0] : null;
                
                if (!text && !audioFile) {
                    showInlineError('Please enter text or upload an audio file', 'processBtn');
                    return;
                }
                
                try {
                    const demo = new ComprehensiveDemo();
                    await demo.processCompleteWorkflow(audioFile, text);
                } catch (error) {
                    console.error('Processing failed:', error);
                    showInlineError('Processing failed: ' + error.message, 'processBtn');
                }
            });
            console.log('✅ Process button bound');
        }

        // Check dependencies first
        const deps = {
            SAMOAPIClient: typeof SAMOAPIClient !== 'undefined',
            UIController: typeof UIController !== 'undefined',
            ChartUtils: typeof ChartUtils !== 'undefined'
        };

        console.log('📋 Dependencies check:', deps);

        const missingDeps = Object.entries(deps).filter(([name, loaded]) => !loaded);
        if (missingDeps.length > 0) {
            console.error('❌ Missing dependencies:', missingDeps.map(([name]) => name));
            throw new Error(`Missing dependencies: ${missingDeps.map(([name]) => name).join(', ')}`);
        }

        window.demo = new ComprehensiveDemo();
        console.log('✅ Demo initialized successfully:', window.demo);

        // Verify processCompleteWorkflow exists
        if (typeof window.demo.processCompleteWorkflow === 'function') {
            console.log('✅ processCompleteWorkflow method available');
        } else {
            console.error('❌ processCompleteWorkflow method missing');
        }

    } catch (error) {
        console.error('❌ Demo initialization failed:', error);

        // Show user-friendly error
        setTimeout(() => {
            const errorDiv = document.createElement('div');
            errorDiv.style.cssText = `
                position: fixed; top: 20px; right: 20px; z-index: 9999;
                background: #ef4444; color: white; padding: 15px; border-radius: 8px;
                font-family: Arial, sans-serif; font-size: 14px; max-width: 300px;
            `;
            // Clear existing content safely
            errorDiv.textContent = '';

            // Create content safely
            const strongEl = document.createElement('strong');
            strongEl.textContent = 'Demo Initialization Failed';

            const br1 = document.createElement('br');
            const br2 = document.createElement('br');

            const smallEl = document.createElement('small');
            smallEl.textContent = 'Please refresh the page';

            // Add content safely
            errorDiv.appendChild(strongEl);
            errorDiv.appendChild(br1);
            errorDiv.appendChild(document.createTextNode(error.message));
            errorDiv.appendChild(br2);
            errorDiv.appendChild(smallEl);
            document.body.appendChild(errorDiv);
        }, 100);
    }
    
    // Smooth scrolling for in-page navigation links
    document.querySelectorAll('nav a[href^="#"], .navbar a[href^="#"], #main-nav a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            // Only handle if the link is for the current page
            if (location.pathname === anchor.pathname && location.hostname === anchor.hostname) {
                e.preventDefault();
                
                // Validate href before using it
                const href = this.getAttribute('href');
                if (!href || typeof href !== 'string' || !href.startsWith('#')) {
                    console.warn('Invalid href for smooth scrolling:', href);
                    return;
                }
                
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                } else {
                    console.warn('Target element not found for href:', href);
                }
            }
        });
    });
});

// Display detailed model analysis
function displayDetailedModelAnalysis(emotionData, summaryData) {
    console.log('Displaying detailed model analysis...', emotionData, summaryData);
    
    // Normalize emotion data (same logic as UI controller)
    let normalizedEmotions = [];
    if (Array.isArray(emotionData)) {
        normalizedEmotions = emotionData.map(emotion => ({
            emotion: emotion.emotion || emotion.label || 'Unknown',
            confidence: emotion.confidence || emotion.score || 0
        }));
    } else if (emotionData.emotions && Array.isArray(emotionData.emotions)) {
        normalizedEmotions = emotionData.emotions.map(emotion => ({
            emotion: emotion.emotion || emotion.label || 'Unknown',
            confidence: emotion.confidence || emotion.score || 0
        }));
    } else if (emotionData.probabilities) {
        normalizedEmotions = Object.entries(emotionData.probabilities).map(([label, prob]) => ({
            emotion: label,
            confidence: prob
        }));
    }
    
    // Sort by confidence (highest first)
    normalizedEmotions = normalizedEmotions.sort((a, b) => b.confidence - a.confidence);
    
    // Calculate primary emotion
    const primaryEmotion = normalizedEmotions.length > 0 ? normalizedEmotions[0] : null;
    const primaryEmotionName = primaryEmotion ? primaryEmotion.emotion : 'Unknown';
    const primaryEmotionConfidence = primaryEmotion ? Math.round(primaryEmotion.confidence * 100) : 0;
    
    // Calculate emotional intensity (average confidence)
    const avgConfidence = normalizedEmotions.length > 0 ? 
        normalizedEmotions.reduce((sum, e) => sum + e.confidence, 0) / normalizedEmotions.length : 0;
    const intensity = avgConfidence > 0.7 ? 'High' : avgConfidence > 0.4 ? 'Medium' : 'Low';
    
    // Calculate sentiment score (weighted average)
    const sentimentWeights = {
        'joy': 1, 'happiness': 1, 'excitement': 1, 'optimism': 0.8, 'gratitude': 0.9,
        'sadness': -1, 'anger': -1, 'fear': -0.8, 'anxiety': -0.7, 'frustration': -0.9,
        'neutral': 0, 'calm': 0.2
    };
    
    const sentimentScore = normalizedEmotions.length > 0 ? 
        normalizedEmotions.reduce((sum, e) => sum + (e.confidence * (sentimentWeights[e.emotion] || 0)), 0) : 0;
    const sentimentLabel = sentimentScore > 0.3 ? 'Positive' : sentimentScore < -0.3 ? 'Negative' : 'Neutral';
    
    // Calculate confidence range (focus on top 3 emotions for more reasonable range)
    const topEmotions = normalizedEmotions.slice(0, 3);
    const confidences = topEmotions.map(e => e.confidence);
    const minConf = confidences.length > 0 ? Math.min(...confidences) : 0;
    const maxConf = confidences.length > 0 ? Math.max(...confidences) : 0;
    const confidenceRange = `${Math.round(minConf * 100)}% - ${Math.round(maxConf * 100)}%`;
    
    // Model processing details
    const modelDetails = `Processed ${normalizedEmotions.length} emotions using SAMO DeBERTa v3 Large. ` +
        `Model confidence: ${Math.round(avgConfidence * 100)}%. ` +
        `Processing time: ${Date.now() - window.processingStartTime || 0}ms. ` +
        `Text length: ${summaryData?.original_length || 0} characters.`;
    
    // Update the UI - with validation
    console.log('🔍 Updating UI elements...');
    
    const primaryEmotionEl = document.getElementById('primaryEmotion');
    const emotionalIntensityEl = document.getElementById('emotionalIntensity');
    const sentimentScoreEl = document.getElementById('sentimentScore');
    const confidenceRangeEl = document.getElementById('confidenceRange');
    const modelDetailsEl = document.getElementById('modelDetails');
    
    console.log('🔍 DOM elements found:', {
        primaryEmotion: !!primaryEmotionEl,
        emotionalIntensity: !!emotionalIntensityEl,
        sentimentScore: !!sentimentScoreEl,
        confidenceRange: !!confidenceRangeEl,
        modelDetails: !!modelDetailsEl
    });
    
    if (primaryEmotionEl) primaryEmotionEl.textContent = `${primaryEmotionName} (${primaryEmotionConfidence}%)`;
    if (emotionalIntensityEl) emotionalIntensityEl.textContent = intensity;
    if (sentimentScoreEl) sentimentScoreEl.textContent = `${sentimentLabel} (${sentimentScore.toFixed(2)})`;
    if (confidenceRangeEl) confidenceRangeEl.textContent = confidenceRange;
    if (modelDetailsEl) modelDetailsEl.textContent = modelDetails;
    
    console.log('✅ UI update completed');
}

// EXACT COPY OF WORKING FUNCTION FROM SIMPLE DEMO
function updateDetailedAnalysisFromSimpleDemo(emotions, summary) {
    console.log('🔧 Using EXACT working function from Simple Demo...', emotions, summary);
    
    // Calculate values - EXACT same logic as Simple Demo
    const primaryEmotion = emotions[0];
    const primaryEmotionName = primaryEmotion ? primaryEmotion.emotion : 'Unknown';
    const primaryEmotionConfidence = primaryEmotion ? Math.round(primaryEmotion.confidence * 100) : 0;
    
    const avgConfidence = emotions.length > 0 ? 
        emotions.reduce((sum, e) => sum + e.confidence, 0) / emotions.length : 0;
    const intensity = avgConfidence > 0.7 ? 'High' : avgConfidence > 0.4 ? 'Medium' : 'Low';
    
    // Sentiment calculation - EXACT same as Simple Demo
    const sentimentWeights = {
        'joy': 1, 'happiness': 1, 'excitement': 1, 'optimism': 0.8, 'gratitude': 0.9,
        'sadness': -1, 'anger': -1, 'fear': -0.8, 'anxiety': -0.7, 'frustration': -0.9,
        'neutral': 0, 'calm': 0.2
    };
    
    const sentimentScore = emotions.length > 0 ? 
        emotions.reduce((sum, e) => sum + (e.confidence * (sentimentWeights[e.emotion] || 0)), 0) : 0;
    const sentimentLabel = sentimentScore > 0.3 ? 'Positive' : sentimentScore < -0.3 ? 'Negative' : 'Neutral';
    
    // Confidence range (top 3 emotions) - EXACT same as Simple Demo
    const top3 = emotions.slice(0, 3);
    const confidences = top3.map(e => e.confidence);
    const minConf = confidences.length > 0 ? Math.min(...confidences) : 0;
    const maxConf = confidences.length > 0 ? Math.max(...confidences) : 0;
    const confidenceRange = `${Math.round(minConf * 100)}% - ${Math.round(maxConf * 100)}%`;
    
    // Model details - EXACT same as Simple Demo
    const modelDetails = `Processed ${emotions.length} emotions using SAMO DeBERTa v3 Large. Model confidence: ${Math.round(avgConfidence * 100)}%. Text length: ${summary?.original_length || 0} characters.`;
    
    // Update DOM elements - EXACT same as Simple Demo
    updateElement('primaryEmotion', `${primaryEmotionName} (${primaryEmotionConfidence}%)`);
    updateElement('emotionalIntensity', intensity);
    updateElement('sentimentScore', `${sentimentLabel} (${sentimentScore.toFixed(2)})`);
    updateElement('confidenceRange', confidenceRange);
    updateElement('modelDetails', modelDetails);
    
    console.log('✅ Simple Demo function completed successfully');
}

// Helper function - EXACT copy from Simple Demo
function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
        console.log(`✅ Updated ${id}: ${value}`);
    } else {
        console.error(`❌ Element not found: ${id}`);
    }
}

// Make the functions globally available
window.displayDetailedModelAnalysis = displayDetailedModelAnalysis;
window.updateDetailedAnalysisFromSimpleDemo = updateDetailedAnalysisFromSimpleDemo;
window.updateElement = updateElement;

// Debug test functions
function testDetailedAnalysis() {
    console.log('🧪 Testing detailed analysis with mock data...');
    
    const mockEmotions = [
        { emotion: 'joy', confidence: 0.85 },
        { emotion: 'excitement', confidence: 0.72 },
        { emotion: 'optimism', confidence: 0.68 },
        { emotion: 'gratitude', confidence: 0.45 },
        { emotion: 'neutral', confidence: 0.15 }
    ];
    
    const mockSummary = {
        original_length: 266,
        summary_length: 93
    };
    
    // Use the EXACT same working function
    updateDetailedAnalysisFromSimpleDemo(mockEmotions, mockSummary);
}

function clearDetailedAnalysis() {
    console.log('🧹 Clearing detailed analysis...');
    updateElement('primaryEmotion', '-');
    updateElement('emotionalIntensity', '-');
    updateElement('sentimentScore', '-');
    updateElement('confidenceRange', '-');
    updateElement('modelDetails', '-');
}

// Also make demo and debug functions available immediately for testing
window.ComprehensiveDemo = ComprehensiveDemo;
window.testDetailedAnalysis = testDetailedAnalysis;
window.clearDetailedAnalysis = clearDetailedAnalysis;
