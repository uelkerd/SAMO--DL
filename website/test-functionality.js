/**
 * Test script for SAMO Demo functionality
 * This script can be run in the browser console to test the demo
 */

// Test function to verify emotion detection display
function testEmotionDisplay() {
    console.log('Testing emotion display...');
    
    // Create mock emotion data
    const mockEmotions = {
        emotions: [
            { emotion: 'frustration', confidence: 0.85 },
            { emotion: 'anger', confidence: 0.72 },
            { emotion: 'annoyance', confidence: 0.68 },
            { emotion: 'sadness', confidence: 0.45 },
            { emotion: 'neutral', confidence: 0.15 }
        ],
        confidence: 0.75,
        mock: true
    };
    
    // Test UI controller
    const uiController = new UIController();
    uiController.showEmotionResults(mockEmotions);
    
    console.log('Emotion display test completed. Check the page for results.');
}

// Test function to verify text summarization display
function testSummaryDisplay() {
    console.log('Testing summary display...');
    
    // Create mock summary data
    const mockSummary = {
        summary: "User is frustrated with system crashes and IT support issues, affecting work progress and deadlines.",
        original_length: 500,
        summary_length: 85,
        compression_ratio: "0.17",
        mock: true
    };
    
    // Test UI controller
    const uiController = new UIController();
    uiController.showSummaryResults(mockSummary);
    
    console.log('Summary display test completed. Check the page for results.');
}

// Test function to verify confidence calculation
function testConfidenceCalculation() {
    console.log('Testing confidence calculation...');
    
    const mockResults = {
        emotions: {
            emotions: [
                { emotion: 'frustration', confidence: 0.85 },
                { emotion: 'anger', confidence: 0.72 },
                { emotion: 'annoyance', confidence: 0.68 }
            ]
        },
        processingTime: 1500,
        modelsUsed: ['SAMO DeBERTa v3 Large']
    };
    
    const uiController = new UIController();
    uiController.updateProcessingInfo(mockResults);
    
    console.log('Confidence calculation test completed. Check the processing info section.');
}

// Test function to run all tests
function runAllTests() {
    console.log('Running all SAMO Demo tests...');
    
    testEmotionDisplay();
    setTimeout(() => testSummaryDisplay(), 1000);
    setTimeout(() => testConfidenceCalculation(), 2000);
    
    console.log('All tests completed. Check the page for results.');
}

// Make functions available globally
window.testEmotionDisplay = testEmotionDisplay;
window.testSummaryDisplay = testSummaryDisplay;
window.testConfidenceCalculation = testConfidenceCalculation;
window.runAllTests = runAllTests;

console.log('SAMO Demo test functions loaded. Run runAllTests() to test all functionality.');
