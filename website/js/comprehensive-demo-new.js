/**
 * Comprehensive Demo for SAMO Deep Learning Platform
 * Demonstrates Whisper (transcription), T5 (summarization), and DeBERTa v3 Large (emotion detection)
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
                    results.transcription = await this.apiClient.transcribeAudio(audioFile);
                    results.modelsUsed.push('Whisper');
                    this.uiController.updateProgressStep('step1', 'completed');
                    this.uiController.showTranscriptionResults(results.transcription);
                } catch (error) {
                    console.error('Transcription failed:', error);
                    this.uiController.updateProgressStep('step1', 'error');
                }
            }

            // Step 2: Summarize text
            let currentText = text;
            if (results.transcription && results.transcription.text) {
                currentText = results.transcription.text;
            }

            if (currentText) {
                try {
                    this.uiController.updateProgressStep('step2', 'active');
                    results.summary = await this.apiClient.summarizeText(currentText);
                    results.modelsUsed.push('T5');
                    this.uiController.updateProgressStep('step2', 'completed');
                    this.uiController.showSummaryResults(results.summary);
                } catch (error) {
                    console.error('Summarization failed:', error);
                    this.uiController.updateProgressStep('step2', 'error');
                    // Continue without summary
                }
            }

            // Step 3: Detect emotions
            if (currentText) {
                try {
                    this.uiController.updateProgressStep('step3', 'active');
                    results.emotions = await this.apiClient.detectEmotions(currentText);
                    results.modelsUsed.push('DeBERTa v3 Large');
                    this.uiController.updateProgressStep('step3', 'completed');
                    this.uiController.showEmotionResults(results.emotions);
                } catch (error) {
                    console.error('Emotion detection failed:', error);
                    this.uiController.updateProgressStep('step3', 'error');
                    // Continue without emotion detection - don't throw error
                }
            }

            // Step 4: Complete
            this.uiController.updateProgressStep('step4', 'completed');
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

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', function() {
    window.demo = new ComprehensiveDemo();
    console.log('Demo initialized:', window.demo);
    
    // Smooth scrolling for in-page navigation links
    document.querySelectorAll('nav a[href^="#"], .navbar a[href^="#"], #main-nav a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            // Only handle if the link is for the current page
            if (location.pathname === anchor.pathname && location.hostname === anchor.hostname) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
});

// Also make demo available immediately for testing
window.ComprehensiveDemo = ComprehensiveDemo;
