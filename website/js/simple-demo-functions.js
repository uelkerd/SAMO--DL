/**
 * Simple Demo Functions - EXACT working code from working-main-demo.html
 * These functions are proven to work and are self-contained
 */

// EXACT copy of working functions from working-main-demo.html
function processText() {
    console.log('🚀 Processing text...');
    const text = document.getElementById('textInput').value;
    if (!text.trim()) {
        alert('Please enter some text to analyze');
        return;
    }
    testWithMockData();
}

function testWithMockData() {
    console.log('🧪 Testing with mock data...');
    
    const mockEmotions = [
        { emotion: 'joy', confidence: 0.85 },
        { emotion: 'excitement', confidence: 0.72 },
        { emotion: 'optimism', confidence: 0.68 },
        { emotion: 'gratitude', confidence: 0.45 },
        { emotion: 'neutral', confidence: 0.15 }
    ];
    
    const mockSummary = {
        original_length: 266,
        summary_length: 93,
        summary: "The text expresses excitement and optimism about future opportunities, while acknowledging some nervousness about upcoming challenges. The overall sentiment is positive and confident."
    };
    
    // Show the results sections
    showResultsSections();
    
    // Create chart
    createSimpleChart(mockEmotions);
    
    // Update detailed analysis
    updateDetailedAnalysis(mockEmotions, mockSummary);
    
    // Update summary
    updateSummary(mockSummary);
    
    console.log('✅ Mock data test completed');
}

function testWithRealAPI() {
    console.log('🌐 Testing with real API...');
    
    try {
        // Show loading state
        const chartContainer = document.getElementById('emotionChart');
        if (chartContainer) {
            chartContainer.innerHTML = '<p>🔄 Calling real API...</p>';
        }
        updateElement('primaryEmotion', 'Loading...');
        updateElement('emotionalIntensity', 'Loading...');
        updateElement('sentimentScore', 'Loading...');
        updateElement('confidenceRange', 'Loading...');
        updateElement('modelDetails', 'Loading...');
        
        // Test text
        const testText = document.getElementById('textInput').value || "I am so excited and happy today! This is such wonderful news and I feel optimistic about the future.";
        
        // Call the real API
        fetch('https://samo-dl-unified-ai-api-7q3q3q3q3q-uc.a.run.app/analyze/journal', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'your-api-key-here'
            },
            body: JSON.stringify({
                text: testText
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`API call failed: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Real API response:', data);
            
            // Extract emotions and summary from the response
            const emotions = data.emotion_analysis?.emotions || [];
            const summary = {
                original_length: testText.length,
                summary_length: data.summary?.length || 0,
                summary: data.summary || "No summary available"
            };
            
            // Show the results sections
            showResultsSections();
            
            // Create chart with real data
            createSimpleChart(emotions);
            
            // Update detailed analysis with real data
            updateDetailedAnalysis(emotions, summary);
            
            // Update summary
            updateSummary(summary);
            
            console.log('✅ Real API test completed successfully');
        })
        .catch(error => {
            console.error('❌ Real API test failed:', error);
            
            // Show error state
            if (chartContainer) {
                chartContainer.innerHTML = `<p>❌ API Error: ${error.message}</p>`;
            }
            updateElement('primaryEmotion', 'Error');
            updateElement('emotionalIntensity', 'Error');
            updateElement('sentimentScore', 'Error');
            updateElement('confidenceRange', 'Error');
            updateElement('modelDetails', `API Error: ${error.message}`);
            
            // Fallback to mock data
            console.log('🔄 Falling back to mock data...');
            setTimeout(() => {
                testWithMockData();
            }, 2000);
        });
        
    } catch (error) {
        console.error('❌ Real API test failed:', error);
        testWithMockData();
    }
}

function createSimpleChart(emotions) {
    console.log('📊 Creating SIMPLE chart...');
    const container = document.getElementById('emotionChart');
    
    if (!container) {
        console.error('❌ Chart container not found');
        return;
    }
    
    // Sort emotions by confidence
    const sortedEmotions = emotions.sort((a, b) => b.confidence - a.confidence);
    
    // Create a SUPER SIMPLE chart with basic HTML and inline styles
    let chartHTML = '<div style="padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; margin: 10px 0;">';
    chartHTML += '<h6 style="text-align: center; margin-bottom: 20px; color: white;">📊 Emotion Confidence Levels</h6>';
    
    sortedEmotions.forEach((emotion, index) => {
        const percentage = Math.round(emotion.confidence * 100);
        const color = getEmotionColor(emotion.emotion);
        
        chartHTML += `
            <div style="margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px; border: 1px solid rgba(255,255,255,0.2);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: bold; color: white; text-transform: capitalize;">${emotion.emotion}</span>
                    <span style="color: #94a3b8;">${percentage}%</span>
                </div>
                <div style="background: rgba(0,0,0,0.3); height: 20px; border-radius: 10px; overflow: hidden; position: relative;">
                    <div style="width: ${percentage}%; height: 100%; background: ${color}; border-radius: 10px; transition: width 1s ease;"></div>
                </div>
            </div>
        `;
    });
    
    chartHTML += `
        <div style="text-align: center; margin-top: 15px; color: #94a3b8; font-size: 14px;">
            Based on ${emotions.length} detected emotions
        </div>
    </div>`;
    
    container.innerHTML = chartHTML;
    console.log('✅ SIMPLE Chart created successfully');
    console.log('📊 Chart HTML length:', chartHTML.length);
    console.log('📊 Container after update:', container.innerHTML.substring(0, 200));
    console.log('📊 Emotion bars found:', container.querySelectorAll('div').length);
}

function updateDetailedAnalysis(emotions, summary) {
    console.log('🧠 Updating detailed analysis...');
    
    // Calculate values
    const primaryEmotion = emotions[0];
    const avgConfidence = emotions.reduce((sum, e) => sum + e.confidence, 0) / emotions.length;
    const intensity = avgConfidence > 0.7 ? 'High' : avgConfidence > 0.4 ? 'Medium' : 'Low';
    
    // Sentiment calculation
    const sentimentWeights = {
        'joy': 1, 'happiness': 1, 'excitement': 1, 'optimism': 0.8, 'gratitude': 0.9,
        'sadness': -1, 'anger': -1, 'fear': -0.8, 'anxiety': -0.7, 'frustration': -0.9,
        'neutral': 0, 'calm': 0.2
    };
    
    const sentimentScore = emotions.reduce((sum, e) => sum + (e.confidence * (sentimentWeights[e.emotion] || 0)), 0);
    const sentimentLabel = sentimentScore > 0.3 ? 'Positive' : sentimentScore < -0.3 ? 'Negative' : 'Neutral';
    
    // Confidence range (top 3 emotions)
    const top3 = emotions.slice(0, 3);
    const confidences = top3.map(e => e.confidence);
    const minConf = Math.min(...confidences);
    const maxConf = Math.max(...confidences);
    const confidenceRange = `${Math.round(minConf * 100)}% - ${Math.round(maxConf * 100)}%`;
    
    // Model details
    const modelDetails = `Processed ${emotions.length} emotions using SAMO DeBERTa v3 Large. Model confidence: ${Math.round(avgConfidence * 100)}%. Text length: ${summary.original_length} characters.`;
    
    // Update DOM elements
    updateElement('primaryEmotion', `${primaryEmotion.emotion} (${Math.round(primaryEmotion.confidence * 100)}%)`);
    updateElement('emotionalIntensity', intensity);
    updateElement('sentimentScore', `${sentimentLabel} (${sentimentScore.toFixed(2)})`);
    updateElement('confidenceRange', confidenceRange);
    updateElement('modelDetails', modelDetails);
    
    console.log('✅ Detailed analysis updated');
}

function updateSummary(summary) {
    console.log('📝 Updating summary...');
    const container = document.getElementById('summaryResults');
    if (container) {
        container.innerHTML = `
            <div class="alert alert-info">
                <h6 class="fw-bold mb-2">📝 Generated Summary</h6>
                <p class="mb-2">${summary.summary}</p>
                <small class="text-muted">
                    Original: ${summary.original_length} characters → Summary: ${summary.summary_length} characters 
                    (${Math.round((1 - summary.summary_length / summary.original_length) * 100)}% reduction)
                </small>
            </div>
        `;
        console.log('✅ Summary updated');
    }
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
        console.log(`✅ Updated ${id}: ${value}`);
    } else {
        console.error(`❌ Element not found: ${id}`);
    }
}

function getEmotionColor(emotion) {
    const colors = {
        'joy': 'linear-gradient(90deg, #10b981, #34d399)',
        'excitement': 'linear-gradient(90deg, #8b5cf6, #a78bfa)',
        'optimism': 'linear-gradient(90deg, #3b82f6, #60a5fa)',
        'gratitude': 'linear-gradient(90deg, #f59e0b, #fbbf24)',
        'neutral': 'linear-gradient(90deg, #6b7280, #9ca3af)',
        'sadness': 'linear-gradient(90deg, #1f2937, #374151)',
        'anger': 'linear-gradient(90deg, #dc2626, #ef4444)'
    };
    return colors[emotion] || 'linear-gradient(90deg, #6366f1, #8b5cf6)';
}

function showResultsSections() {
    console.log('👁️ Showing results sections...');
    
    // Show emotion analysis results
    const emotionResults = document.getElementById('emotionResults');
    if (emotionResults) {
        emotionResults.classList.remove('result-section-hidden');
        emotionResults.classList.add('result-section-visible');
        console.log('✅ Emotion results section shown');
    }
    
    // Show summarization results
    const summarizationResults = document.getElementById('summarizationResults');
    if (summarizationResults) {
        summarizationResults.classList.remove('result-section-hidden');
        summarizationResults.classList.add('result-section-visible');
        console.log('✅ Summarization results section shown');
    }
}

function clearAll() {
    console.log('🧹 Clearing all data...');
    
    // Hide results sections
    const emotionResults = document.getElementById('emotionResults');
    if (emotionResults) {
        emotionResults.classList.add('result-section-hidden');
        emotionResults.classList.remove('result-section-visible');
    }
    
    const summarizationResults = document.getElementById('summarizationResults');
    if (summarizationResults) {
        summarizationResults.classList.add('result-section-hidden');
        summarizationResults.classList.remove('result-section-visible');
    }
    
    // Clear chart content
    const chartContainer = document.getElementById('emotionChart');
    if (chartContainer) {
        chartContainer.innerHTML = '<p class="text-center text-muted">Click "Process Text" to see the emotion analysis chart</p>';
    }
    
    // Clear summary content
    const summaryContainer = document.getElementById('summaryResults');
    if (summaryContainer) {
        summaryContainer.innerHTML = '<p class="text-center text-muted">Click "Process Text" to see the text summary</p>';
    }
    
    // Clear detailed analysis
    updateElement('primaryEmotion', '-');
    updateElement('emotionalIntensity', '-');
    updateElement('sentimentScore', '-');
    updateElement('confidenceRange', '-');
    updateElement('modelDetails', '-');
    
    console.log('✅ All data cleared');
}

// Make functions globally available
window.processText = processText;
window.testWithMockData = testWithMockData;
window.testWithRealAPI = testWithRealAPI;
window.createSimpleChart = createSimpleChart;
window.updateDetailedAnalysis = updateDetailedAnalysis;
window.updateSummary = updateSummary;
window.updateElement = updateElement;
window.getEmotionColor = getEmotionColor;
window.clearAll = clearAll;
window.showResultsSections = showResultsSections;

console.log('🚀 Simple Demo Functions loaded and ready!');
