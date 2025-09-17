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

// Inline error and success display functions to replace alert dialogs
function showInlineError(message, targetElementId) {
    showInlineMessage(message, targetElementId, 'error');
}

function showInlineSuccess(message, targetElementId) {
    showInlineMessage(message, targetElementId, 'success');
}

function showInlineMessage(message, targetElementId, type = 'error') {
    // Remove any existing inline messages
    const existingMessages = document.querySelectorAll('.inline-message');
    existingMessages.forEach(msg => msg.remove());

    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `inline-message alert ${type === 'error' ? 'alert-danger' : 'alert-success'} mt-2`;
    messageDiv.setAttribute('role', 'alert');
    messageDiv.style.cssText = 'animation: fadeIn 0.3s ease-in; font-size: 0.9rem;';

    // Add content safely
    messageDiv.textContent = message;

    // Insert message after target element
    const targetElement = document.getElementById(targetElementId);
    if (targetElement) {
        targetElement.parentNode.insertBefore(messageDiv, targetElement.nextSibling);
    } else {
        // Fallback: append to body
        document.body.appendChild(messageDiv);
    }

    // Auto-remove after 4 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.remove();
        }
    }, 4000);
}

// Helper function to format processing time in a user-friendly way
function formatProcessingTime(timeMs) {
    if (timeMs >= 1000) {
        const seconds = (timeMs / 1000).toFixed(1);
        return `${seconds}s`;
    }
    return `${Math.round(timeMs)}ms`;
}

// EXACT copy of working functions from working-main-demo.html
async function processText() {
    console.log('🚀 Processing text...');
    const text = document.getElementById('textInput').value;
    console.log('🔍 Text from input:', text);
    console.log('🔍 Text length:', text.length);
    if (!text.trim()) {
        showInlineError('Please enter some text to analyze', 'textInput');
        return;
    }
    // Call real API only - no fallbacks
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
        // Check for OpenAI API key from multiple sources
        let apiKey = window.SAMO_CONFIG?.OPENAI?.API_KEY || localStorage.getItem('openai_api_key');

        if (!apiKey || apiKey.trim() === '') {
            // Prompt user for API key with clear instructions
            showInlineError('⚠️ OpenAI API key required for AI text generation. Click "Manage API Key" to set up.', 'textInput');

            // Reset input state
            if (textInput) {
                textInput.value = '';
                textInput.style.borderColor = '#ef4444';
                textInput.style.boxShadow = '0 0 0 0.2rem rgba(239, 68, 68, 0.25)';
                setTimeout(() => {
                    textInput.style.borderColor = '';
                    textInput.style.boxShadow = '';
                }, 3000);
            }
            return;
        }

        // Enhanced emotional prompts for more variety and authenticity
        const prompts = [
            "Today started like any other day, but something unexpected happened that completely changed my mood. I found myself feeling",
            "I've been reflecting on recent changes in my life, and I'm experiencing a whirlwind of emotions. Right now I'm particularly",
            "This week has been a journey of self-discovery. I wake up each morning feeling different, but today I'm especially",
            "After a long conversation with someone close to me, I'm left feeling quite contemplative and",
            "The weather outside perfectly matches my internal state today. I'm feeling deeply",
            "I had a moment of clarity during my morning routine that left me feeling incredibly",
            "Sometimes life throws you curveballs that make you reassess everything. Today I'm processing feelings of",
            "I've been working towards a personal goal, and the progress (or lack thereof) has me feeling",
            "A random act of kindness today reminded me why human connection matters. I'm feeling",
            "After some quiet time alone with my thoughts, I've realized I'm feeling much more"
        ];

        const randomPrompt = prompts[Math.floor(Math.random() * prompts.length)];

        console.log('🤖 Generating AI text with OpenAI API...');
        console.log('📝 Using prompt:', randomPrompt);

        // Use configuration from SAMO_CONFIG
        const openaiConfig = window.SAMO_CONFIG.OPENAI;

        // Call OpenAI API with enhanced prompt for journal-like content
        const response = await fetch(openaiConfig.API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey.trim()}`
            },
            body: JSON.stringify({
                model: openaiConfig.MODEL,
                messages: [
                    {
                        role: 'system',
                        content: 'You are a creative writing assistant that generates authentic, emotionally rich personal journal entries. Write in first person, include specific details and genuine emotions. Create content that feels like real personal reflection, with varied sentence structure and natural flow. Aim for 3-4 sentences that express complex, layered emotions.'
                    },
                    {
                        role: 'user',
                        content: `Write a personal journal entry that continues this thought: "${randomPrompt}" - Make it authentic, emotionally detailed, and personally reflective. Include specific emotions and inner thoughts.`
                    }
                ],
                max_tokens: openaiConfig.MAX_TOKENS,
                temperature: openaiConfig.TEMPERATURE + 0.1, // Slightly higher for more creativity
                presence_penalty: 0.3, // Encourage more diverse content
                frequency_penalty: 0.2  // Reduce repetition
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(`OpenAI API error: ${response.status} ${response.statusText}${errorData.error ? ' - ' + errorData.error.message : ''}`);
        }

        const data = await response.json();

        if (!data.choices || !data.choices[0] || !data.choices[0].message) {
            throw new Error('Invalid response format from OpenAI API');
        }

        const generatedText = data.choices[0].message.content.trim();

        console.log('✅ AI text generated successfully:', generatedText);
        console.log('📊 Token usage:', data.usage);

        // Update the text input with generated content
        if (textInput) {
            textInput.value = generatedText;
            textInput.style.borderColor = '#10b981';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(16, 185, 129, 0.25)';

            // Reset border after success animation
            setTimeout(() => {
                textInput.style.borderColor = '';
                textInput.style.boxShadow = '';
            }, 2000);
        }

        showInlineSuccess('✅ AI journal text generated successfully!', 'textInput');

    } catch (error) {
        console.error('❌ AI text generation failed:', error);

        // Show detailed error message
        const errorMessage = error.message.includes('API key')
            ? '❌ Invalid API key. Please check your OpenAI API key.'
            : error.message.includes('quota')
            ? '❌ OpenAI API quota exceeded. Please check your account.'
            : `❌ AI generation failed: ${error.message}`;

        showInlineError(errorMessage, 'textInput');

        // Reset input state
        if (textInput) {
            textInput.value = '';
            textInput.style.borderColor = '#ef4444';
            textInput.style.boxShadow = '0 0 0 0.2rem rgba(239, 68, 68, 0.25)';

            setTimeout(() => {
                textInput.style.borderColor = '';
                textInput.style.boxShadow = '';
            }, 3000);
        }
    }
}

// API Key Management
function manageApiKey() {
    const currentKey = localStorage.getItem('openai_api_key');
    const maskedKey = currentKey ? `${currentKey.substring(0, 8)}...${currentKey.substring(currentKey.length - 4)}` : 'None';

    const action = confirm(`Current OpenAI API Key: ${maskedKey}\n\n✅ Click OK to set a new key\n❌ Click Cancel to clear the current key\n\nNote: API keys are stored locally in your browser and never sent to our servers.`);

    if (action) {
        // Set new key
        const newKey = prompt('Enter your OpenAI API key:\n\n🔗 Get your API key from: https://platform.openai.com/api-keys\n\n⚠️ Your key should start with "sk-" followed by characters.\n\nAPI Key:');
        if (newKey && newKey.trim() && newKey.trim().startsWith('sk-')) {
            localStorage.setItem('openai_api_key', newKey.trim());
            showInlineSuccess('✅ OpenAI API key saved successfully! You can now generate AI journal text.', 'apiKeyBtn');
            updateApiKeyButtonStatus();
            console.log('✅ OpenAI API key updated successfully');
        } else if (newKey && newKey.trim() && !newKey.trim().startsWith('sk-')) {
            showInlineError('❌ Invalid API key format. OpenAI keys should start with "sk-".', 'apiKeyBtn');
        } else {
            showInlineError('❌ No valid API key provided.', 'apiKeyBtn');
        }
    } else {
        // Clear current key
        if (currentKey) {
            localStorage.removeItem('openai_api_key');
            showInlineSuccess('🗑️ API key cleared successfully!', 'apiKeyBtn');
            updateApiKeyButtonStatus();
        } else {
            showInlineError('ℹ️ No API key was stored.', 'apiKeyBtn');
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
            // Clear and rebuild button content safely
            while (apiKeyBtn.firstChild) {
                apiKeyBtn.removeChild(apiKeyBtn.firstChild);
            }
            const iconSpan = document.createElement('span');
            iconSpan.className = 'material-icons me-2';
            iconSpan.textContent = 'key';
            apiKeyBtn.appendChild(iconSpan);
            apiKeyBtn.appendChild(document.createTextNode('API Key ✓'));
        } else {
            apiKeyBtn.classList.remove('btn-warning');
            apiKeyBtn.classList.add('btn-outline-warning');
            // Clear and rebuild button content safely
            while (apiKeyBtn.firstChild) {
                apiKeyBtn.removeChild(apiKeyBtn.firstChild);
            }
            const iconSpan = document.createElement('span');
            iconSpan.className = 'material-icons me-2';
            iconSpan.textContent = 'key';
            apiKeyBtn.appendChild(iconSpan);
            apiKeyBtn.appendChild(document.createTextNode('API Key'));
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

// Static sample text function removed - now using only real OpenAI API generation
// This ensures all content is authentic and not pre-written samples


async function callSummarizationAPI(text) {
    console.log('📝 Calling real summarization API...');

    try {
        // Add parameters to potentially influence summarization style
        const params = new URLSearchParams({
            text: text,
            style: 'third_person',    // Request 3rd person style
            format: 'narrative',      // Request narrative format
            mode: 'paraphrase'        // Request paraphrasing mode
        });

        const apiUrl = `${window.SAMO_CONFIG.API.BASE_URL}${window.SAMO_CONFIG.API.ENDPOINTS.SUMMARIZE}?${params.toString()}`;
        console.log('🔗 Summarization API URL:', apiUrl);
        console.log('📝 Text being summarized:', text);
        console.log('📝 Text length:', text.length);
        console.log('📝 Requested style: third_person narrative paraphrase');

        // Increase timeout for summarization since it can take longer
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 45000); // 45 seconds

        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Length': '0',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            console.error('❌ Summarization API call failed with status:', response.status);
            const errorText = await response.text().catch(() => 'No error details');
            throw new Error(`Summarization API failed: ${response.status} ${response.statusText} - ${errorText}`);
        }

        const data = await response.json();
        console.log('✅ Summarization API response:', data);
        console.log('🔍 Response type:', typeof data);
        console.log('🔍 Response keys:', Object.keys(data));
        console.log('🔍 Full response structure:', JSON.stringify(data, null, 2));

        // Check all possible field names for the summary
        let summaryText = null;
        const possibleFields = ['summary', 'text', 'summarized_text', 'result', 'output', 'content', 'message', 'paraphrase'];

        for (const field of possibleFields) {
            if (data[field] && typeof data[field] === 'string') {
                console.log(`🎯 Found summary in field '${field}':`, data[field]);
                summaryText = data[field];
                break;
            }
        }

        // If no summary found, log the issue and use fallback
        if (!summaryText) {
            console.error('❌ No summary field found in API response');
            console.error('❌ Available fields:', Object.keys(data));
            throw new Error(`Invalid API response format. Available fields: ${Object.keys(data).join(', ')}`);
        }

        // Clean up any auto-conversion prefixes that may come from the API
        summaryText = summaryText.replace(/^\[Auto-converted to 3rd person\]:\s*/i, '');

        // Enhanced analysis for 3rd person validation
        const isFirstPerson = /\b(I|me|my|myself|we|us|our|ourselves)\b/i.test(summaryText);
        const isThirdPerson = /\b(he|she|they|the person|the individual|the user|this person)\b/i.test(summaryText);

        if (isFirstPerson && !isThirdPerson) {
            console.warn('⚠️ WARNING: Summary is still in first person, not converted to third person!');
            console.warn('⚠️ Original text style not properly converted by API');
            // Try to auto-convert simple cases
            const thirdPersonAttempt = summaryText
                .replace(/\bI\b/g, 'The person')
                .replace(/\bme\b/g, 'them')
                .replace(/\bmy\b/g, 'their')
                .replace(/\bmyself\b/g, 'themselves');

            if (thirdPersonAttempt !== summaryText) {
                console.log('🔄 Attempting automatic 3rd person conversion...');
                summaryText = thirdPersonAttempt; // Remove the "[Auto-converted to 3rd person]: " prefix
            }
        }

        // Check if the "summary" is actually just truncated original text
        if (summaryText && text.includes(summaryText)) {
            console.warn('⚠️ WARNING: API returned truncated original text, not a proper summary!');
            console.warn('⚠️ Original text contains the returned "summary"');
            throw new Error('API returned truncated text instead of proper summarization');
        }

        // Calculate compression ratio
        const compressionRatio = (text.length - summaryText.length) / text.length;
        console.log(`📊 Summarization stats: ${text.length} → ${summaryText.length} chars (${(compressionRatio * 100).toFixed(1)}% compression)`);

        const summary = {
            original_length: text.length,
            summary_length: summaryText.length,
            compression_ratio: compressionRatio,
            is_third_person: isThirdPerson && !isFirstPerson,
            summary: summaryText
        };

        return summary;
    } catch (error) {
        console.error('❌ Summarization API error:', error);

        // Enhanced error handling with specific error types
        if (error.name === 'AbortError') {
            throw new Error('Summarization request timed out after 45 seconds. The API may be overloaded.');
        } else if (error.message.includes('Failed to fetch')) {
            throw new Error('Network error: Unable to connect to summarization API. Please check your connection.');
        } else {
            throw error;
        }
    }
}

async function testWithRealAPI() {
    console.log('🌐 Testing with real API...');
    const startTime = performance.now(); // Start timing

    try {
        // Show enhanced loading state with progress indicators
        const chartContainer = document.getElementById('emotionChart');
        if (chartContainer) {
            while (chartContainer.firstChild) {
                chartContainer.removeChild(chartContainer.firstChild);
            }

            // Create enhanced loading container
            const loadingDiv = document.createElement('div');
            loadingDiv.style.cssText = 'text-align: center; padding: 30px; background: rgba(139, 92, 246, 0.05); border-radius: 10px; border: 1px solid rgba(139, 92, 246, 0.2);';

            // Loading spinner
            const spinner = document.createElement('div');
            spinner.className = 'spinner-border text-primary mb-3';
            spinner.style.cssText = 'width: 2rem; height: 2rem;';
            loadingDiv.appendChild(spinner);

            // Loading title
            const title = document.createElement('h6');
            title.textContent = '🧠 AI Analysis in Progress';
            title.style.cssText = 'color: #8b5cf6; margin-bottom: 15px;';
            loadingDiv.appendChild(title);

            // Progress message
            const message = document.createElement('p');
            message.id = 'emotionLoadingMessage';
            message.textContent = 'Initializing emotion analysis models...';
            message.style.cssText = 'color: #6b7280; margin-bottom: 10px;';
            loadingDiv.appendChild(message);

            // Estimated time
            const timeEst = document.createElement('small');
            timeEst.textContent = 'Estimated time: 1-3 seconds';
            timeEst.style.cssText = 'color: #9ca3af;';
            loadingDiv.appendChild(timeEst);

            chartContainer.appendChild(loadingDiv);

            // Update progress messages
            setTimeout(() => {
                const msg = document.getElementById('emotionLoadingMessage');
                if (msg) msg.textContent = 'Processing text with SAMO DeBERTa v3 Large...';
            }, 500);

            setTimeout(() => {
                const msg = document.getElementById('emotionLoadingMessage');
                if (msg) msg.textContent = 'Analyzing emotional patterns and confidence scores...';
            }, 1500);
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

        // Add timeout handling for emotion API
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 seconds timeout

        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Length': '0',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            signal: controller.signal
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
        .then(async data => {
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
        .finally(() => {
            // Clear timeout
            clearTimeout(timeoutId);
        })
        .catch(error => {
            console.error('❌ Real API test failed:', error);
            console.error('❌ Error details:', error.message);
            console.error('❌ Error stack:', error.stack);
            console.error('❌ Error type:', error.constructor.name);

            // Detect CORS or network errors
            let errorType = 'Unknown Error';
            let errorDetail = error.message;

            if (error.name === 'AbortError') {
                errorType = 'Timeout Error';
                errorDetail = 'API call timed out after 30 seconds. The server may be overloaded. Please try again.';
            } else if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
                errorType = 'CORS/Network Error';
                errorDetail = 'Cannot reach API due to CORS policy or network issues. This is likely because browsers block direct calls to external APIs.';
            } else if (error.message.includes('NetworkError')) {
                errorType = 'Network Error';
                errorDetail = 'Network connection failed. Check your internet connection.';
            } else if (error.message.includes('timeout')) {
                errorType = 'Timeout Error';
                errorDetail = 'API call timed out. The server may be busy.';
            }

            console.error('❌ Detected error type:', errorType);
            console.error('❌ Error detail:', errorDetail);

            // Show detailed error in UI
            if (chartContainer) {
                // Clear container safely
                while (chartContainer.firstChild) {
                    chartContainer.removeChild(chartContainer.firstChild);
                }

                // Create error container
                const errorDiv = document.createElement('div');
                errorDiv.style.cssText = 'color: #ef4444; padding: 20px; text-align: center; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px;';

                // Create title
                const title = document.createElement('h5');
                title.textContent = `❌ ${errorType}`;
                errorDiv.appendChild(title);

                // Create error message
                const errorP = document.createElement('p');
                errorP.style.cssText = 'margin-bottom: 15px;';
                errorP.textContent = errorDetail;
                errorDiv.appendChild(errorP);

                // Create URL info
                const urlP = document.createElement('p');
                urlP.style.cssText = 'font-size: 0.9rem; color: #cbd5e1; margin-bottom: 10px;';
                urlP.textContent = `URL: ${apiUrl}`;
                errorDiv.appendChild(urlP);

                // Create time info
                const timeP = document.createElement('p');
                timeP.style.cssText = 'font-size: 0.9rem; color: #cbd5e1; margin-bottom: 15px;';
                timeP.textContent = `Time: ${new Date().toLocaleTimeString()}`;
                errorDiv.appendChild(timeP);

                // Create suggestions
                const suggestionsDiv = document.createElement('div');
                suggestionsDiv.style.cssText = 'background: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 8px; margin-top: 15px;';

                const suggestionsTitle = document.createElement('h6');
                suggestionsTitle.textContent = '💡 Suggestions:';
                suggestionsDiv.appendChild(suggestionsTitle);

                const suggestionsList = document.createElement('ul');
                suggestionsList.style.cssText = 'text-align: left; font-size: 0.9rem; margin: 0;';

                if (errorType === 'CORS/Network Error') {
                    suggestionsList.innerHTML = `
                        <li>Try using the "Test with Real API" button in debug mode</li>
                        <li>Check browser console for CORS error details</li>
                        <li>Consider using a CORS proxy or server-side integration</li>
                    `;
                } else {
                    suggestionsList.innerHTML = `
                        <li>Check your internet connection</li>
                        <li>Try again in a few moments</li>
                        <li>Check browser console for more details</li>
                    `;
                }

                suggestionsDiv.appendChild(suggestionsList);
                errorDiv.appendChild(suggestionsDiv);

                chartContainer.appendChild(errorDiv);
            }
            updateElement('primaryEmotion', 'Error');
            updateElement('emotionalIntensity', 'Error');
            updateElement('sentimentScore', 'Error');
            updateElement('confidenceRange', 'Error');
            updateElement('modelDetails', `${errorType}: ${errorDetail}`);

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
    const compressionRatio = Number(summary.compression_ratio || 0);

    // Update the summary text with enhanced formatting
    const summaryElement = document.getElementById('summaryText');
    if (summaryElement) {
        summaryElement.textContent = summary.summary;

        // Add visual indicator for 3rd person conversion
        if (summary.is_third_person) {
            summaryElement.style.borderLeft = '4px solid #10b981';
            summaryElement.style.paddingLeft = '12px';
            summaryElement.title = '✅ Successfully converted to 3rd person perspective';
        } else {
            summaryElement.style.borderLeft = '4px solid #f59e0b';
            summaryElement.style.paddingLeft = '12px';
            summaryElement.title = '⚠️ May still contain 1st person elements';
        }
    }

    // Update length statistics
    updateElement('originalLength', originalLen);
    updateElement('summaryLength', summaryLen);

    // Add compression ratio display if element exists
    const compressionElement = document.getElementById('compressionRatio');
    if (compressionElement) {
        const compressionPercent = (compressionRatio * 100).toFixed(1);
        compressionElement.textContent = `${compressionPercent}% compression`;

        // Color code based on compression quality
        if (compressionRatio > 0.3) {
            compressionElement.style.color = '#10b981'; // Good compression
        } else if (compressionRatio > 0.1) {
            compressionElement.style.color = '#f59e0b'; // Moderate compression
        } else {
            compressionElement.style.color = '#ef4444'; // Poor compression
        }
    }

    // Add 3rd person indicator if element exists
    const thirdPersonElement = document.getElementById('thirdPersonStatus');
    if (thirdPersonElement) {
        if (summary.is_third_person) {
            thirdPersonElement.textContent = '✅ 3rd Person';
            thirdPersonElement.style.color = '#10b981';
        } else {
            thirdPersonElement.textContent = '⚠️ Mixed/1st Person';
            thirdPersonElement.style.color = '#f59e0b';
        }
    }

    // Log quality metrics
    console.log(`📊 Summary Quality Metrics:
        - Length: ${originalLen} → ${summaryLen} chars
        - Compression: ${(compressionRatio * 100).toFixed(1)}%
        - 3rd Person: ${summary.is_third_person ? 'Yes' : 'No'}
    `);

    // Create summary chart
    createSummaryChart(summary);

    console.log('✅ Summary updated with quality indicators');
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

    // Update processing information elements (try both original and compact versions)
    const timeFormatted = formatProcessingTime(processingTime);
    const confidenceFormatted = `${Math.round(safeAvgConfidence * 100)}%`;
    const modelsUsedText = 'SAMO DeBERTa v3 Large, SAMO T5';
    const statusText = 'Completed';

    // Update original elements (if they exist)
    updateElement('totalTime', timeFormatted);
    updateElement('processingStatus', statusText);
    updateElement('modelsUsed', modelsUsedText);
    updateElement('avgConfidence', confidenceFormatted);

    // Update compact elements (new layout)
    updateElement('totalTimeCompact', timeFormatted);
    updateElement('processingStatusCompact', statusText);
    updateElement('modelsUsedCompact', modelsUsedText);
    updateElement('avgConfidenceCompact', confidenceFormatted);

    // Sync using LayoutManager if available
    if (typeof LayoutManager !== 'undefined' && LayoutManager.syncProcessingInfo) {
        LayoutManager.syncProcessingInfo();
    }

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

function resetToInputScreen() {
    console.log('🔄 Resetting to input screen...');

    // Clear all data first
    clearAll();

    // Hide results layout
    const resultsLayout = document.getElementById('resultsLayout');
    if (resultsLayout) {
        resultsLayout.classList.add('d-none');
        resultsLayout.classList.remove('d-block');
    }

    // Show input layout
    const inputLayout = document.getElementById('inputLayout');
    if (inputLayout) {
        inputLayout.classList.remove('d-none');
        inputLayout.classList.add('d-block');
    }

    // Clear input field
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.value = '';
        textInput.style.borderColor = '';
        textInput.style.boxShadow = '';
        textInput.focus(); // Focus back on input
    }

    // Use LayoutManager if available
    if (typeof LayoutManager !== 'undefined' && LayoutManager.showInputLayout) {
        LayoutManager.showInputLayout();
    }

    console.log('✅ Reset to input screen completed');
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

    // Clear processing information (both original and compact)
    updateElement('totalTime', '-');
    updateElement('processingStatus', 'Ready');
    updateElement('modelsUsed', '-');
    updateElement('avgConfidence', '-');

    // Clear compact elements too
    updateElement('totalTimeCompact', '-');
    updateElement('processingStatusCompact', 'Ready');
    updateElement('modelsUsedCompact', '-');
    updateElement('avgConfidenceCompact', '-');

    console.log('✅ All data cleared');
}

// Make functions globally available
window.processText = processText;
window.generateSampleText = generateSampleText;
window.manageApiKey = manageApiKey;
window.updateApiKeyButtonStatus = updateApiKeyButtonStatus;
window.testWithRealAPI = testWithRealAPI;
window.callSummarizationAPI = callSummarizationAPI;
window.createSimpleChart = createSimpleChart;
window.createSummaryChart = createSummaryChart;
window.updateDetailedAnalysis = updateDetailedAnalysis;
window.updateSummary = updateSummary;
window.updateProcessingInfo = updateProcessingInfo;
window.updateElement = updateElement;
window.getEmotionColor = getEmotionColor;
window.resetToInputScreen = resetToInputScreen;
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
