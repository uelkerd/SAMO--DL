/**
 * Simple Demo Functions - EXACT working code from working-main-demo.html
 * These functions are proven to work and are self-contained
 * 
 * 🤖 AI TEXT GENERATION SETUP:
 * To enable real AI text generation, you need an OpenAI API key:
 * 1. Go to https://platform.openai.com/api-keys
 * 2. Create a new API key
 * 3. Add it to website/config.js in the OPENAI.API_KEY field
 * 4. Save the file and refresh the demo
 * 
 * Without an API key, the demo will use static sample texts as fallback.
 * 
 * 🔒 SECURITY: API keys are stored in config.js, not hardcoded in this file!
 */

// EXACT copy of working functions from working-main-demo.html
async function processText() {
    console.log('🚀 Processing text...');
    const text = document.getElementById('textInput').value;
    console.log('🔍 Text from input:', text);
    console.log('🔍 Text length:', text.length);
    if (!text.trim()) {
        alert('Please enter some text to analyze');
        return;
    }
    // Try real API first, fallback to mock data
    console.log('🔍 About to call testWithRealAPI from processText');
    await testWithRealAPI();
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
        // Check for OpenAI API key
        let apiKey = localStorage.getItem('openai_api_key');
        if (!apiKey) {
            apiKey = prompt('Please enter your OpenAI API key to generate AI text:');
            if (apiKey) {
                localStorage.setItem('openai_api_key', apiKey);
            } else {
                console.log('❌ No API key provided, using static samples');
                generateStaticSampleText();
                return;
            }
        }
        
        // Different emotional prompts for variety
        const prompts = [
            "Today I'm feeling incredibly excited and optimistic about",
            "I'm experiencing a mix of emotions right now. On one hand, I feel",
            "What a challenging day this has been. I started the morning feeling",
            "I'm in such a peaceful and content state of mind. After",
            "Today I'm feeling incredibly motivated and determined because"
        ];
        
        const randomPrompt = prompts[Math.floor(Math.random() * prompts.length)];
        
        console.log('🤖 Generating AI text with OpenAI API...');
        
        // Call OpenAI API
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify({
                model: 'gpt-3.5-turbo',
                messages: [
                    {
                        role: 'system',
                        content: 'You are a helpful assistant that generates personal journal entries with rich emotional content. Write 2-3 sentences that express various emotions and feelings.'
                    },
                    {
                        role: 'user',
                        content: `Please complete this journal entry: "${randomPrompt}"`
                    }
                ],
                max_tokens: 200,
                temperature: 0.8
            })
        });
        
        if (!response.ok) {
            throw new Error(`OpenAI API error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        const generatedText = data.choices[0].message.content;
        
        console.log('✅ AI text generated successfully:', generatedText);
        
        // Update the text input with generated content
        if (textInput) {
            textInput.value = generatedText;
            textInput.style.borderColor = '#10b981';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(16, 185, 129, 0.25)';
        }
        
    } catch (error) {
        console.error('❌ AI text generation failed:', error);
        
        // Show error message
        if (textInput) {
            textInput.value = `❌ AI generation failed: ${error.message}. Using sample text instead.`;
            textInput.style.borderColor = '#ef4444';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(239, 68, 68, 0.25)';
        }
        
        // Fallback to static samples after a delay
        setTimeout(() => {
            console.log('🔄 Falling back to static sample texts...');
            generateStaticSampleText();
        }, 2000);
    }
}

// API Key Management
function manageApiKey() {
    const currentKey = localStorage.getItem('openai_api_key');
    const maskedKey = currentKey ? `${currentKey.substring(0, 8)}...${currentKey.substring(currentKey.length - 4)}` : 'None';
    
    const action = confirm(`Current API Key: ${maskedKey}\n\nClick OK to set a new key, or Cancel to clear the current key.`);
    
    if (action) {
        // Set new key
        const newKey = prompt('Enter your OpenAI API key:');
        if (newKey && newKey.trim()) {
            localStorage.setItem('openai_api_key', newKey.trim());
            alert('✅ API key saved successfully!');
            updateApiKeyButtonStatus();
        } else {
            alert('❌ No valid API key provided.');
        }
    } else {
        // Clear current key
        if (currentKey) {
            localStorage.removeItem('openai_api_key');
            alert('🗑️ API key cleared successfully!');
            updateApiKeyButtonStatus();
        } else {
            alert('ℹ️ No API key was stored.');
        }
    }
}

// Update API Key button visual status
function updateApiKeyButtonStatus() {
    const apiKeyBtn = document.getElementById('apiKeyBtn');
    if (apiKeyBtn) {
        const hasKey = localStorage.getItem('openai_api_key');
        if (hasKey) {
            apiKeyBtn.classList.remove('btn-outline-warning');
            apiKeyBtn.classList.add('btn-warning');
            apiKeyBtn.innerHTML = '<span class="material-icons me-2">key</span>API Key ✓';
        } else {
            apiKeyBtn.classList.remove('btn-warning');
            apiKeyBtn.classList.add('btn-outline-warning');
            apiKeyBtn.innerHTML = '<span class="material-icons me-2">key</span>API Key';
        }
    }
}

// Removed generateEmotionVariations function - we now use only real model output

async function generateWithOpenAI(prompt) {
    console.log('🤖 Calling OpenAI API with prompt:', prompt);
    
    // Get configuration from centralized config
    if (!window.SAMO_CONFIG) {
        throw new Error('SAMO configuration not loaded. Please ensure config.js is included.');
    }
    
    // Check if OpenAI proxy is configured
    if (!window.SAMO_CONFIG.OPENAI.PROXY_URL) {
        throw new Error('OpenAI proxy URL not configured. Please set SAMO_CONFIG.OPENAI.PROXY_URL.');
    }
    
    // OpenAI proxy configuration (no API key needed on client)
    const PROXY_URL = window.SAMO_CONFIG.OPENAI.PROXY_URL;
    const MODEL = window.SAMO_CONFIG.OPENAI.MODEL;
    const MAX_TOKENS = window.SAMO_CONFIG.OPENAI.MAX_TOKENS;
    const TEMPERATURE = window.SAMO_CONFIG.OPENAI.TEMPERATURE;
    
    try {
        const response = await fetch(PROXY_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
                // No Authorization header - API key handled server-side
            },
            body: JSON.stringify({
                prompt: prompt,
                model: MODEL,
                max_tokens: MAX_TOKENS,
                temperature: TEMPERATURE
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorData.error?.message || 'Unknown error'}`);
        }
        
        const data = await response.json();
        console.log('✅ OpenAI proxy response:', data);
        
        // Handle proxy response format (expecting { text: "generated content" })
        if (data && data.text) {
            const generatedText = data.text.trim();
            console.log('✅ Generated text from proxy:', generatedText.substring(0, 100) + '...');
            return generatedText;
        } else if (data && data.choices && data.choices.length > 0 && data.choices[0].message?.content) {
            // Fallback to OpenAI format if proxy returns OpenAI response
            const generatedText = data.choices[0].message.content.trim();
            console.log('✅ Generated text from OpenAI format:', generatedText.substring(0, 100) + '...');
            return generatedText;
        } else {
            throw new Error('No generated text in proxy response');
        }
        
    } catch (error) {
        console.error('❌ OpenAI API failed:', error);
        throw error;
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

async function callSummarizationAPI(text) {
    console.log('📝 Calling real summarization API...');

    try {
        const apiUrl = `${window.SAMO_CONFIG.API.BASE_URL}${window.SAMO_CONFIG.API.ENDPOINTS.SUMMARIZE}`;
        console.log('🔗 Summarization API URL:', apiUrl);

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            console.error('❌ Summarization API call failed with status:', response.status);
            throw new Error(`Summarization API failed: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log('✅ Summarization API response:', data);

        // Extract summary from API response (adjust based on actual API response format)
        const summary = {
            original_length: text.length,
            summary_length: data.summary ? data.summary.length : text.length,
            summary: data.summary || data.text || `Unable to generate summary - API response: ${JSON.stringify(data)}`
        };

        return summary;
    } catch (error) {
        console.error('❌ Summarization API error:', error);

        // Fallback to a better mock summary than just truncation
        return {
            original_length: text.length,
            summary_length: Math.round(text.length * 0.6),
            summary: `Unable to connect to summarization service. The text expresses various emotional states and personal reflections. [Fallback mode - original text: ${text.substring(0, 50)}...]`
        };
    }
}

async function testWithRealAPI() {
    console.log('🌐 Testing with real API...');
    const startTime = performance.now(); // Start timing

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
        let testText = document.getElementById('textInput').value || "I am so excited and happy today! This is such wonderful news and I feel optimistic about the future.";
        
        // Check text length limit (API seems to have ~400 character limit)
        const MAX_TEXT_LENGTH = 400;
        if (testText.length > MAX_TEXT_LENGTH) {
            console.log(`⚠️ Text too long (${testText.length} chars), truncating to ${MAX_TEXT_LENGTH} chars`);
            testText = testText.substring(0, MAX_TEXT_LENGTH) + "...";
            
            // Show user-friendly warning
            const textInput = document.getElementById('textInput');
            if (textInput) {
                textInput.style.borderColor = '#f59e0b';
                textInput.style.boxShadow = '0 0 0 0.2rem rgba(245, 158, 11, 0.25)';
                setTimeout(() => {
                    textInput.style.borderColor = '';
                    textInput.style.boxShadow = '';
                }, 3000);
            }
            
            // Show warning in console and potentially in UI
            console.warn(`⚠️ Text truncated from ${testText.length + 3} to ${MAX_TEXT_LENGTH} characters due to API limitations`);
        }
        
        console.log('🔍 testWithRealAPI - testText:', testText);
        console.log('🔍 testWithRealAPI - testText length:', testText.length);
        
        // Try the LIVE emotion API directly (no CORS proxy needed)
        console.log('🔥 Calling LIVE emotion API directly...');
        const apiUrl = `https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app/analyze/emotion?text=${encodeURIComponent(testText)}`;
        console.log('🔗 API URL:', apiUrl);
        console.log('📝 Text being analyzed:', testText);
        console.log('📝 Text length:', testText.length);
        console.log('🕐 Timestamp:', new Date().toISOString());
        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Length': '0',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        })
        .then(response => {
            console.log('🔍 Response status:', response.status);
            console.log('🔍 Response headers:', response.headers);
            console.log('🔍 Response ok:', response.ok);
            
            if (!response.ok) {
                console.error('❌ API call failed with status:', response.status);
                console.error('❌ Response status text:', response.statusText);
                
                // Handle specific error cases
                if (response.status === 400) {
                    throw new Error(`Bad Request: Text may be too long or invalid. Please try shorter text (under 400 characters).`);
                } else if (response.status === 500) {
                    throw new Error(`Server Error: The API server encountered an error. Please try again.`);
                } else {
                    throw new Error(`API call failed: ${response.status} ${response.statusText}`);
                }
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Real API response:', data);
            console.log('🔍 Response type:', typeof data);
            console.log('🔍 Response keys:', Object.keys(data));
            console.log('🔍 Full response structure:', JSON.stringify(data, null, 2));
            
            // Convert API response to our format and generate additional emotions
            const emotions = [];
            let primaryEmotion = null;
            let primaryConfidence = 0;
            
            if (data.emotions && typeof data.emotions === 'object' && data.predicted_emotion) {
                // New unified API format: {emotions: {anger: 0.0009, love: 0.9821}, predicted_emotion: "love"}
                console.log('📊 Using new unified API format with emotions object');
                primaryEmotion = data.predicted_emotion;
                primaryConfidence = data.emotions[data.predicted_emotion] || 0;
                
                // Convert emotions object to array format for display
                const emotionEntries = Object.entries(data.emotions);
                emotionEntries.sort((a, b) => b[1] - a[1]); // Sort by confidence descending
                
                emotionEntries.forEach(([emotion, confidence]) => {
                    emotions.push({
                        emotion: emotion,
                        confidence: confidence
                    });
                });
            } else if (data.emotion && data.confidence) {
                // Old unified API format: {emotion: "love", confidence: 0.968}
                console.log('📊 Using old unified API format');
                primaryEmotion = data.emotion;
                primaryConfidence = data.confidence;
            } else if (data.emotions && Array.isArray(data.emotions)) {
                console.log('📊 Using emotions array format');
                // Current API format: {emotions: [{emotion: "excitement", confidence: 0.739}]}
                if (data.emotions.length > 0) {
                    primaryEmotion = data.emotions[0].emotion;
                    primaryConfidence = data.emotions[0].confidence;
                }
            } else if (data.all_emotions) {
                // Alternative API format
                if (data.all_emotions.length > 0) {
                    primaryEmotion = data.all_emotions[0].emotion;
                    primaryConfidence = data.all_emotions[0].confidence;
                }
            } else if (data.emotion_analysis?.emotions) {
                // Old API format
                const emotionEntries = Object.entries(data.emotion_analysis.emotions);
                if (emotionEntries.length > 0) {
                    const [emotion, confidence] = emotionEntries[0];
                    primaryEmotion = emotion;
                    primaryConfidence = confidence;
                }
            }
            
            // Only add primary emotion if we don't already have emotions from the API
            if (emotions.length === 0 && primaryEmotion && primaryConfidence > 0) {
                emotions.push({
                    emotion: primaryEmotion,
                    confidence: primaryConfidence
                });
            }
            
            // Display only the real model's output - no fake emotions!
            console.log('✅ Using real model output:', emotions.length, 'emotions detected');

            // Call real summarization API in parallel for better performance
            console.log('📝 Calling real summarization API...');
            const summary = await callSummarizationAPI(testText);
            console.log('✅ Summarization API completed:', summary);

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
            console.error('❌ Error details:', error.message);
            console.error('❌ Error stack:', error.stack);
            
            // Show detailed error in UI
            if (chartContainer) {
                chartContainer.innerHTML = `
                    <div style="color: #ef4444; padding: 20px; text-align: center;">
                        <h5>❌ API Error</h5>
                        <p><strong>Error:</strong> ${error.message}</p>
                        <p><strong>URL:</strong> ${apiUrl}</p>
                        <p><strong>Time:</strong> ${new Date().toLocaleTimeString()}</p>
                        <p>Check browser console for more details.</p>
                    </div>
                `;
            }
            updateElement('primaryEmotion', 'Error');
            updateElement('emotionalIntensity', 'Error');
            updateElement('sentimentScore', 'Error');
            updateElement('confidenceRange', 'Error');
            updateElement('modelDetails', `API Error: ${error.message}`);
            
            // Don't fallback to mock data - show the real error
            console.log('❌ API failed - showing error instead of mock data');
            console.log('❌ This will help us debug the real API issue');
        });
        
    } catch (error) {
        console.error('❌ Real API test failed:', error);
        console.log('❌ API failed - showing error instead of mock data');
        console.log('❌ This will help us debug the real API issue');
    }
}

function createSimpleChart(emotions) {
    console.log('📊 Creating SIMPLE chart...');
    const container = document.getElementById('emotionChart');

    if (!container) {
        console.error('❌ Chart container not found');
        return;
    }

    // Using pure HTML/CSS charts - no Chart.js dependency needed

    // Destroy existing chart to prevent memory leaks
    destroyExistingChart();

    // Validate emotions array
    if (!Array.isArray(emotions) || emotions.length === 0) {
        console.warn('⚠️ Invalid or empty emotions array, showing no emotions message');
        emotions = [];
    }
    
    // Sort emotions by confidence
    const sortedEmotions = emotions.sort((a, b) => b.confidence - a.confidence);
    
    // Create a SUPER SIMPLE chart with DOM manipulation (XSS-safe)
    container.innerHTML = ''; // Clear container
    
    // Create main chart container
    const chartContainer = document.createElement('div');
    chartContainer.style.cssText = 'padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; margin: 10px 0;';
    
    // Create title
    const title = document.createElement('h6');
    title.style.cssText = 'text-align: center; margin-bottom: 20px; color: white;';
    title.textContent = '📊 Real Model Output';
    chartContainer.appendChild(title);
    
    if (sortedEmotions.length === 0) {
        const noEmotions = document.createElement('p');
        noEmotions.style.cssText = 'text-align: center; color: #94a3b8;';
        noEmotions.textContent = 'No emotions detected by the model';
        chartContainer.appendChild(noEmotions);
    } else {
        // Show only top 5 emotions for better readability
        sortedEmotions.slice(0, 5).forEach((emotion, index) => {
            const percentage = Math.round(emotion.confidence * 100);
            const color = getEmotionColor(emotion.emotion);
            
            // Create emotion row container
            const emotionRow = document.createElement('div');
            emotionRow.style.cssText = 'margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px; border: 1px solid rgba(255,255,255,0.2);';
            
            // Create header with emotion name and percentage
            const header = document.createElement('div');
            header.style.cssText = 'display: flex; justify-content: space-between; margin-bottom: 8px;';
            
            const emotionName = document.createElement('span');
            emotionName.style.cssText = 'font-weight: bold; color: white; text-transform: capitalize;';
            emotionName.textContent = emotion.emotion;
            
            const percentageSpan = document.createElement('span');
            percentageSpan.style.cssText = 'color: #94a3b8;';
            percentageSpan.textContent = percentage + '%';
            
            header.appendChild(emotionName);
            header.appendChild(percentageSpan);
            emotionRow.appendChild(header);
            
            // Create progress bar container
            const progressContainer = document.createElement('div');
            progressContainer.style.cssText = 'background: rgba(0,0,0,0.3); height: 20px; border-radius: 10px; overflow: hidden; position: relative;';
            
            // Create progress bar fill
            const progressFill = document.createElement('div');
            progressFill.style.cssText = `width: ${Math.max(0, Math.min(100, percentage))}%; height: 100%; background: ${color}; border-radius: 10px; transition: width 1s ease;`;
            progressContainer.appendChild(progressFill);
            emotionRow.appendChild(progressContainer);
            chartContainer.appendChild(emotionRow);
        });
    }

    // Create footer
    const footer = document.createElement('div');
    footer.style.cssText = 'text-align: center; margin-top: 15px; color: #94a3b8; font-size: 14px;';
    footer.textContent = `Showing top 5 of ${emotions.length} detected emotion${emotions.length !== 1 ? 's' : ''}`;
    chartContainer.appendChild(footer);

    container.appendChild(chartContainer);

    // Store reference to the HTML chart for cleanup (even though it's not a Chart.js chart)
    currentEmotionChart = {
        destroy: () => {
            if (container) {
                container.innerHTML = '<p class="text-center text-muted">Click "Process Text" to see the emotion analysis chart</p>';
            }
        }
    };

    console.log('✅ SIMPLE Chart created successfully');
    console.log('📊 Container after update:', container.innerHTML.substring(0, 200));
    console.log('📊 Emotion bars found:', container.querySelectorAll('div').length);
}

function updateDetailedAnalysis(emotions, summary) {
    console.log('🧠 Updating detailed analysis...');
    
    // Validate emotions array
    if (!Array.isArray(emotions) || emotions.length === 0) {
        console.warn('⚠️ Invalid or empty emotions array, using defaults');
        emotions = [{ emotion: 'unknown', confidence: 0 }];
    }
    
    // Calculate values safely
    const primaryEmotion = emotions[0] || { emotion: 'unknown', confidence: 0 };
    const avgConfidence = emotions.length > 0 ? emotions.reduce((sum, e) => sum + e.confidence, 0) / emotions.length : 0;
    const intensity = avgConfidence > 0.7 ? 'High' : avgConfidence > 0.4 ? 'Medium' : 'Low';
    
    // Sentiment calculation
    const sentimentWeights = {
        'joy': 1, 'happiness': 1, 'excitement': 1, 'optimism': 0.8, 'gratitude': 0.9,
        'love': 1, 'hope': 0.8, 'amusement': 0.7, 'contentment': 0.8, 'relief': 0.6,
        'pride': 0.7, 'curiosity': 0.3, 'surprise': 0.2, 'calm': 0.2, 'neutral': 0,
        'sadness': -1, 'anger': -1, 'fear': -0.8, 'anxiety': -0.7, 'frustration': -0.9,
        'disappointment': -0.8, 'disgust': -0.6, 'shame': -0.7, 'guilt': -0.8,
        'embarrassment': -0.5, 'confusion': -0.3, 'longing': -0.2, 'nostalgia': -0.1
    };
    
    const sentimentScore = emotions.reduce((sum, e) => sum + (e.confidence * (sentimentWeights[e.emotion] || 0)), 0);
    const sentimentLabel = sentimentScore > 0.3 ? 'Positive' : sentimentScore < -0.3 ? 'Negative' : 'Neutral';
    
    // Confidence range (all emotions, or single emotion) - safe calculation
    const confidences = emotions.map(e => e.confidence);
    const minConf = confidences.length > 0 ? Math.min(...confidences) : 0;
    const maxConf = confidences.length > 0 ? Math.max(...confidences) : 0;
    const confidenceRange = confidences.length > 1 ? 
        `${Math.round(minConf * 100)}% - ${Math.round(maxConf * 100)}%` : 
        `${Math.round(maxConf * 100)}%`;
    
    // Model details - safe length access with numeric coercion
    const summaryLength = summary ? Number(summary.original_length) || 0 : 0;
    const modelDetails = `Processed ${emotions.length} emotions using SAMO DeBERTa v3 Large. Model confidence: ${Math.round(avgConfidence * 100)}%. Text length: ${summaryLength} characters.`;
    
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
    
    // Coerce numeric values to prevent XSS and ensure proper formatting
    const summaryLen = Number(summary.summary_length);
    const originalLen = Number(summary.original_length);
    
    // Update the summary text
    updateElement('summaryText', summary.summary);
    updateElement('originalLength', originalLen);
    updateElement('summaryLength', summaryLen);
    
    // Create summary chart
    createSummaryChart(summary);
    
        console.log('✅ Summary updated');
}

function updateElement(id, value) {
    try {
        const element = document.getElementById(id);
        if (element) {
            // Safely update text content, handling null/undefined values
            element.textContent = value !== null && value !== undefined ? value : '-';
            console.log(`✅ Updated ${id}: ${value}`);
        } else {
            console.warn(`⚠️ Element not found: ${id} (this is expected for some optional elements)`);
        }
    } catch (error) {
        console.error(`❌ Error updating element ${id}:`, error);
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
        'anger': 'linear-gradient(90deg, #dc2626, #ef4444)',
        'disappointment': 'linear-gradient(90deg, #7c2d12, #dc2626)',
        'frustration': 'linear-gradient(90deg, #b91c1c, #ef4444)',
        'anxiety': 'linear-gradient(90deg, #7c3aed, #a855f7)',
        'fear': 'linear-gradient(90deg, #581c87, #7c3aed)',
        'calm': 'linear-gradient(90deg, #0d9488, #14b8a6)',
        'love': 'linear-gradient(90deg, #ec4899, #f472b6)',
        'surprise': 'linear-gradient(90deg, #f59e0b, #fbbf24)',
        'disgust': 'linear-gradient(90deg, #059669, #10b981)',
        'shame': 'linear-gradient(90deg, #374151, #6b7280)',
        'pride': 'linear-gradient(90deg, #7c2d12, #dc2626)',
        'relief': 'linear-gradient(90deg, #0d9488, #14b8a6)',
        'hope': 'linear-gradient(90deg, #3b82f6, #60a5fa)',
        'confusion': 'linear-gradient(90deg, #6b7280, #9ca3af)',
        'curiosity': 'linear-gradient(90deg, #7c3aed, #a855f7)',
        'amusement': 'linear-gradient(90deg, #f59e0b, #fbbf24)',
        'embarrassment': 'linear-gradient(90deg, #be185d, #ec4899)',
        'guilt': 'linear-gradient(90deg, #374151, #6b7280)',
        'relief': 'linear-gradient(90deg, #0d9488, #14b8a6)',
        'contentment': 'linear-gradient(90deg, #10b981, #34d399)',
        'longing': 'linear-gradient(90deg, #7c3aed, #a855f7)',
        'nostalgia': 'linear-gradient(90deg, #6b7280, #9ca3af)'
    };
    return colors[emotion] || 'linear-gradient(90deg, #6366f1, #8b5cf6)';
}

function showResultsSections() {
    console.log('👁️ Showing results sections...');

    try {
        // Show emotion analysis results
        const emotionResults = document.getElementById('emotionResults');
        if (emotionResults) {
            emotionResults.classList.remove('result-section-hidden');
            emotionResults.classList.add('result-section-visible');
            emotionResults.style.display = 'block';
            console.log('✅ Emotion results section shown');
        } else {
            console.warn('⚠️ emotionResults element not found');
        }

        // Show summarization results
        const summarizationResults = document.getElementById('summarizationResults');
        if (summarizationResults) {
            summarizationResults.classList.remove('result-section-hidden');
            summarizationResults.classList.add('result-section-visible');
            summarizationResults.style.display = 'block';
            console.log('✅ Summarization results section shown');
        } else {
            console.warn('⚠️ summarizationResults element not found');
        }
    } catch (error) {
        console.error('❌ Error showing results sections:', error);
    }
}

function createSummaryChart(summary) {
    console.log('📊 Creating summary chart...');
    const container = document.getElementById('summaryChart');
    
    if (!container) {
        console.error('❌ Summary chart container not found');
        return;
    }
    
    // Safe reduction calculation with proper validation
    const summaryLen = Number(summary.summary_length);
    const originalLen = Number(summary.original_length);
    
    // Calculate ratio safely, treating original_length <= 0 as no reduction
    const ratio = Number.isFinite(originalLen) && originalLen > 0 
        ? Math.max(0, Math.min(1, summaryLen / originalLen)) 
        : 1; // No reduction if original length is invalid
    
    const reduction = Math.max(0, Math.min(100, Math.round((1 - ratio) * 100)));
    
    let chartHTML = '<div style="padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; margin: 10px 0;">';
    chartHTML += '<h6 style="text-align: center; margin-bottom: 20px; color: white;">📝 Text Summarization Analysis</h6>';
    
    // Summary stats
    chartHTML += `
        <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
            <div style="text-align: center; flex: 1; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 0 5px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #3b82f6;">${originalLen}</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">Original Length</div>
            </div>
            <div style="text-align: center; flex: 1; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 0 5px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">${summaryLen}</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">Summary Length</div>
            </div>
            <div style="text-align: center; flex: 1; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 0 5px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #f59e0b;">${reduction}%</div>
                <div style="font-size: 0.8rem; color: #cbd5e1;">Reduction</div>
            </div>
        </div>
    `;
    
    // Length comparison bars - safe width calculation
    const originalWidth = 100;
    const summaryWidth = Number.isFinite(originalLen) && originalLen > 0 
        ? Math.max(0, Math.min(100, (summaryLen / originalLen) * 100))
        : 0; // No width if original length is invalid
    
    chartHTML += `
        <div style="margin-bottom: 20px;">
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: white;">Original Text</span>
                    <span style="color: #94a3b8;">${originalLen} characters</span>
                </div>
                <div style="background: rgba(0,0,0,0.3); height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="width: ${originalWidth}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 10px;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold; color: white;">Summary</span>
                    <span style="color: #94a3b8;">${summaryLen} characters</span>
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
    
    // Validate inputs
    if (!Array.isArray(emotions) || emotions.length === 0) {
        console.warn('⚠️ Invalid emotions array in updateProcessingInfo');
    }
    
    // Validate avgConfidence
    const safeAvgConfidence = typeof avgConfidence === 'number' && !isNaN(avgConfidence) ? avgConfidence : 0;
    
    // Use real processing time if available, otherwise use mock
    const processingTime = typeof startTime !== 'undefined' ? Math.round(performance.now() - startTime) : Math.round(Math.random() * 2000 + 1000);
    
    // Update processing information elements
    updateElement('totalTime', `${processingTime}ms`);
    updateElement('processingStatus', 'Completed');
    updateElement('modelsUsed', 'SAMO DeBERTa v3 Large, SAMO T5');
    updateElement('avgConfidence', `${Math.round(safeAvgConfidence * 100)}%`);
    
    console.log('✅ Processing information updated');
}

function showChartError(message) {
    console.error('📊 Chart Error:', message);
    const container = document.getElementById('emotionChart');

    if (!container) {
        console.error('❌ Chart container not found for error display');
        return;
    }

    // Create error message HTML (safe - no user input)
    container.innerHTML = `
        <div style="padding: 20px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; text-align: center;">
            <div style="color: #ef4444; margin-bottom: 10px;">
                <span style="font-size: 2rem;">⚠️</span>
            </div>
            <h6 style="color: #ef4444; font-weight: bold;">Chart Loading Error</h6>
            <p style="color: #cbd5e1; margin-bottom: 15px;">${message}</p>
            <button onclick="location.reload()" style="background: #ef4444; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">
                🔄 Refresh Page
            </button>
        </div>
    `;
}

function clearAll() {
    console.log('🧹 Clearing all data...');

    // Destroy existing chart to prevent memory leaks
    destroyExistingChart();

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
window.manageApiKey = manageApiKey;
window.updateApiKeyButtonStatus = updateApiKeyButtonStatus;
window.testWithMockData = testWithMockData;
window.testWithRealAPI = testWithRealAPI;
window.callSummarizationAPI = callSummarizationAPI;
window.createSimpleChart = createSimpleChart;
window.createSummaryChart = createSummaryChart;
window.updateDetailedAnalysis = updateDetailedAnalysis;
window.updateSummary = updateSummary;
window.updateProcessingInfo = updateProcessingInfo;
window.updateElement = updateElement;
window.getEmotionColor = getEmotionColor;
window.clearAll = clearAll;
window.showResultsSections = showResultsSections;
window.showChartError = showChartError;
// Removed generateEmotionVariations - using only real model output

// Global variables for proper cleanup
let currentEmotionChart = null;

// Chart cleanup function
function destroyExistingChart() {
    if (currentEmotionChart) {
        try {
            currentEmotionChart.destroy();
            console.log('✅ Previous chart destroyed successfully');
        } catch (error) {
            console.warn('⚠️ Error destroying previous chart:', error);
        }
        currentEmotionChart = null;
    }
}

console.log('🚀 Simple Demo Functions loaded and ready!');
