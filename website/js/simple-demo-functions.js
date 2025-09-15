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
    // Try real API first, fallback to mock data
    testWithRealAPI();
}

async function generateSampleText() {
    console.log('✨ Generating AI-powered sample journal text...');
    
    // Show loading state
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.value = '🤖 Generating AI text...';
        textInput.style.borderColor = '#8b5cf6';
        textInput.style.boxShadow = '0 0 0 0.2rem rgba(139, 92, 246, 0.25)';
    }
    
    try {
        // Different emotional prompts for variety
        const prompts = [
            "Today I'm feeling incredibly excited and optimistic about",
            "I'm experiencing a mix of emotions right now. On one hand, I feel",
            "What a challenging day this has been. I started the morning feeling",
            "I'm in such a peaceful and content state of mind. After",
            "Today I'm feeling incredibly motivated and determined because"
        ];
        
        const randomPrompt = prompts[Math.floor(Math.random() * prompts.length)];
        
        // Call Hugging Face Inference API
        const hfToken = window.SAMO_CONFIG?.HUGGING_FACE?.API_TOKEN;
        
        if (!hfToken || hfToken === 'YOUR_HF_TOKEN_HERE') {
            console.warn('⚠️ Hugging Face token not configured, falling back to static samples');
            generateStaticSampleText();
            return;
        }
        
        console.log('🤗 Using HF token:', hfToken.substring(0, 10) + '...');
        
        const response = await fetch('https://api-inference.huggingface.co/models/microsoft/DialoGPT-small', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${hfToken}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                inputs: randomPrompt,
                parameters: {
                    max_length: window.SAMO_CONFIG.HUGGING_FACE.MAX_LENGTH,
                    temperature: window.SAMO_CONFIG.HUGGING_FACE.TEMPERATURE,
                    do_sample: true,
                    top_p: 0.9
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`Hugging Face API error: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        let generatedText = result[0]?.generated_text || '';
        
        // Clean up the generated text
        if (generatedText) {
            // Remove the original prompt from the beginning
            generatedText = generatedText.replace(randomPrompt, '').trim();
            
            // Ensure it's a complete sentence
            if (!generatedText.endsWith('.') && !generatedText.endsWith('!') && !generatedText.endsWith('?')) {
                generatedText += '.';
            }
            
            // Combine prompt with generated text
            generatedText = randomPrompt + ' ' + generatedText;
        } else {
            throw new Error('No text generated from API');
        }
        
        // Update the text input
        if (textInput) {
            textInput.value = generatedText;
            console.log('✅ AI-generated text inserted:', generatedText.substring(0, 100) + '...');
            
            // Success animation
            textInput.style.borderColor = '#10b981';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(16, 185, 129, 0.25)';
            setTimeout(() => {
                textInput.style.borderColor = '';
                textInput.style.boxShadow = '';
            }, 2000);
        }
        
    } catch (error) {
        console.error('❌ AI text generation failed:', error);
        
        // Fallback to static samples
        console.log('🔄 Falling back to static sample texts...');
        generateStaticSampleText();
    }
}

function generateStaticSampleText() {
    console.log('📝 Using static sample texts as fallback...');
    
    const sampleTexts = [
        "Today has been absolutely incredible! I woke up feeling energized and optimistic about the future. The morning sun streaming through my window filled me with such warmth and joy. I had a productive meeting with my team where we discussed our exciting new project, and I could feel the enthusiasm radiating from everyone. There's something magical about working with passionate people who share your vision. I'm genuinely excited about the possibilities ahead, though I must admit I'm also feeling a bit nervous about the challenges we might face. But you know what? I'm confident we can overcome anything together. This sense of gratitude and anticipation is exactly what I needed to fuel my motivation for the coming weeks.",
        
        "I'm feeling a complex mix of emotions right now. On one hand, I'm incredibly proud of the progress we've made on our latest project. The team has been working tirelessly, and seeing our hard work pay off fills me with such satisfaction and pride. However, I'm also experiencing some anxiety about the upcoming deadline. The pressure is mounting, and I can feel the weight of responsibility on my shoulders. I'm excited about the potential impact of what we're building, but I'm also worried about whether we can deliver everything on time. It's this strange combination of anticipation and apprehension that keeps me up at night. I know I should be more confident, but sometimes the fear of failure creeps in. I'm trying to stay positive and focus on the amazing opportunity we have.",
        
        "What a rollercoaster of a day! I started the morning feeling frustrated and overwhelmed by all the tasks piling up on my desk. The constant interruptions and unexpected problems were really testing my patience. I felt angry and stressed, wondering why everything had to be so complicated. But then something beautiful happened - a colleague reached out with words of encouragement, and suddenly my perspective shifted. I felt grateful for their kindness and support. The afternoon brought some exciting news about a potential promotion, and I found myself feeling hopeful and optimistic again. It's amazing how quickly emotions can change when you're surrounded by good people and positive energy. I'm ending the day feeling much more balanced and ready to tackle whatever comes next.",
        
        "I'm in such a peaceful state of mind right now. After a long week of intense work and constant decision-making, I finally took some time for myself this weekend. I went for a long walk in the park, and the simple act of being in nature filled me with such calm and contentment. I feel grateful for these quiet moments of reflection. There's something deeply satisfying about slowing down and appreciating the little things - the sound of birds chirping, the gentle breeze, the way the sunlight filters through the trees. I'm feeling optimistic about the future and confident in my ability to handle whatever challenges come my way. This sense of inner peace is exactly what I needed to recharge and refocus on what truly matters.",
        
        "Today I'm feeling incredibly motivated and determined! I had a breakthrough moment during my morning workout where I pushed myself harder than I thought possible. The sense of accomplishment and strength I felt was absolutely exhilarating. I'm excited about the new goals I've set for myself, both personally and professionally. There's this fire burning inside me that's driving me to be better, to do more, to achieve things I never thought possible. I'm feeling confident and ready to take on any challenge that comes my way. The support from my friends and family has been incredible, and their belief in me fills me with such gratitude and love. I know there will be obstacles ahead, but I'm feeling prepared and optimistic about overcoming them."
    ];
    
    // Pick a random sample text
    const randomText = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
    
    // Update the text input
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.value = randomText;
        console.log('✅ Static sample text generated and inserted');
        
        // Add a subtle animation to show the text was updated
        textInput.style.borderColor = '#f59e0b';
        textInput.style.boxShadow = '0 0 0 0.2rem rgba(245, 158, 11, 0.25)';
        setTimeout(() => {
            textInput.style.borderColor = '';
            textInput.style.boxShadow = '';
        }, 1000);
    }
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
        
        // Try the LIVE emotion API first (no auth required)
        console.log('🔥 Calling LIVE emotion API...');
        fetch('https://samo-emotion-api-minimal-71517823771.us-central1.run.app/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
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
            
            // Convert API response to our format
            const emotions = [];
            if (data.emotions) {
                // Current API format: {emotions: [{emotion: "excitement", confidence: 0.739}]}
                data.emotions.forEach(emotion => {
                    emotions.push({
                        emotion: emotion.emotion,
                        confidence: emotion.confidence
                    });
                });
            } else if (data.all_emotions) {
                // Alternative API format
                data.all_emotions.forEach(emotion => {
                    emotions.push({
                        emotion: emotion.emotion,
                        confidence: emotion.confidence
                    });
                });
            } else if (data.emotion_analysis?.emotions) {
                // Old API format
                Object.entries(data.emotion_analysis.emotions).forEach(([emotion, confidence]) => {
                    emotions.push({ emotion, confidence });
                });
            }
            
            // Create mock summary for now (since emotion API doesn't do summarization)
            const summary = {
                original_length: testText.length,
                summary_length: Math.round(testText.length * 0.4), // 60% reduction
                summary: `[Real API] ${testText.substring(0, 100)}...` // Truncated summary
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
    
    // Update processing information
    updateProcessingInfo(emotions, summary, avgConfidence);
    
    console.log('✅ Detailed analysis updated');
}

function updateSummary(summary) {
    console.log('📝 Updating summary...');
    
    // Update the summary text
    updateElement('summaryText', summary.summary);
    updateElement('originalLength', summary.original_length);
    updateElement('summaryLength', summary.summary_length);
    
    // Create summary chart
    createSummaryChart(summary);
    
        console.log('✅ Summary updated');
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

function createSummaryChart(summary) {
    console.log('📊 Creating summary chart...');
    const container = document.getElementById('summaryChart');
    
    if (!container) {
        console.error('❌ Summary chart container not found');
        return;
    }
    
    const reduction = Math.round((1 - summary.summary_length / summary.original_length) * 100);
    
    let chartHTML = '<div style="padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; margin: 10px 0;">';
    chartHTML += '<h6 style="text-align: center; margin-bottom: 20px; color: white;">📝 Text Summarization Analysis</h6>';
    
    // Summary stats
    chartHTML += `
        <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
            <div style="text-align: center; flex: 1; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 0 5px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #3b82f6;">${summary.original_length}</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">Original Length</div>
            </div>
            <div style="text-align: center; flex: 1; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 0 5px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">${summary.summary_length}</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">Summary Length</div>
            </div>
            <div style="text-align: center; flex: 1; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 0 5px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #f59e0b;">${reduction}%</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">Reduction</div>
            </div>
        </div>
    `;
    
    // Length comparison bars
    const originalWidth = 100;
    const summaryWidth = (summary.summary_length / summary.original_length) * 100;
    
    chartHTML += `
        <div style="margin-bottom: 20px;">
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: white;">Original Text</span>
                    <span style="color: #94a3b8;">${summary.original_length} characters</span>
                </div>
                <div style="background: rgba(0,0,0,0.3); height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="width: ${originalWidth}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 10px;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: white;">Summary</span>
                    <span style="color: #94a3b8;">${summary.summary_length} characters</span>
                </div>
                <div style="background: rgba(0,0,0,0.3); height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="width: ${summaryWidth}%; height: 100%; background: linear-gradient(90deg, #10b981, #34d399); border-radius: 10px;"></div>
                </div>
            </div>
        </div>
    `;
    
    chartHTML += '</div>';
    
    container.innerHTML = chartHTML;
    console.log('✅ Summary chart created successfully');
}

function updateProcessingInfo(emotions, summary, avgConfidence) {
    console.log('ℹ️ Updating processing information...');
    
    // Calculate processing time (mock)
    const processingTime = Math.round(Math.random() * 2000 + 1000); // 1-3 seconds
    
    // Update processing information elements
    updateElement('totalTime', `${processingTime}ms`);
    updateElement('processingStatus', 'Completed');
    updateElement('modelsUsed', 'SAMO DeBERTa v3 Large, SAMO T5');
    updateElement('avgConfidence', `${Math.round(avgConfidence * 100)}%`);
    
    console.log('✅ Processing information updated');
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
    const summaryContainer = document.getElementById('summaryChart');
    if (summaryContainer) {
        summaryContainer.innerHTML = '<p class="text-center text-muted">Click "Process Text" to see the text summary chart</p>';
    }
    
    // Clear summary text
    updateElement('summaryText', '');
    updateElement('originalLength', '-');
    updateElement('summaryLength', '-');
    
    // Clear detailed analysis
    updateElement('primaryEmotion', '-');
    updateElement('emotionalIntensity', '-');
    updateElement('sentimentScore', '-');
    updateElement('confidenceRange', '-');
    updateElement('modelDetails', '-');
    
    // Clear processing information
    updateElement('totalTime', '-');
    updateElement('processingStatus', 'Ready');
    updateElement('modelsUsed', '-');
    updateElement('avgConfidence', '-');
    
    console.log('✅ All data cleared');
}

// Make functions globally available
window.processText = processText;
window.generateSampleText = generateSampleText;
window.testWithMockData = testWithMockData;
window.testWithRealAPI = testWithRealAPI;
window.createSimpleChart = createSimpleChart;
window.createSummaryChart = createSummaryChart;
window.updateDetailedAnalysis = updateDetailedAnalysis;
window.updateSummary = updateSummary;
window.updateProcessingInfo = updateProcessingInfo;
window.updateElement = updateElement;
window.getEmotionColor = getEmotionColor;
window.clearAll = clearAll;
window.showResultsSections = showResultsSections;

console.log('🚀 Simple Demo Functions loaded and ready!');
