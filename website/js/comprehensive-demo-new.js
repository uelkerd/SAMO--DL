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
                    results.transcription = await this.apiClient.transcribeAudio(audioFile);
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
                    const analysisResponse = await this.apiClient.analyzeText(currentText);
                    
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
        
        // Debug Chart.js loading
        console.log('🔍 Checking Chart.js...');
        if (typeof Chart !== 'undefined') {
            console.log('✅ Chart.js is loaded correctly');
        } else {
            console.error('❌ Chart.js is not loaded');
        }

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
                    alert('Please enter text or upload an audio file');
                    return;
                }
                
                try {
                    const demo = new ComprehensiveDemo();
                    await demo.processCompleteWorkflow(audioFile, text);
                } catch (error) {
                    console.error('Processing failed:', error);
                    alert('Processing failed: ' + error.message);
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
            errorDiv.innerHTML = `
                <strong>Demo Initialization Failed</strong><br>
                ${error.message}<br>
                <small>Please refresh the page</small>
            `;
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

// Also make demo available immediately for testing
window.ComprehensiveDemo = ComprehensiveDemo;
