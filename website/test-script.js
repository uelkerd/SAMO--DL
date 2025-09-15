/**
 * Automated Test Script for SAMO Demo
 * Tests all core functionality programmatically
 */

console.log('🧪 Starting SAMO Demo Automated Tests...');

// Test 1: Material Icons
function testMaterialIcons() {
    console.log('🔍 Testing Material Icons...');
    
    try {
        const testIcon = document.createElement('span');
        testIcon.className = 'material-icons';
        testIcon.textContent = 'check';
        document.body.appendChild(testIcon);
        const computedStyle = window.getComputedStyle(testIcon);
        
        if (computedStyle.fontFamily.includes('Material Icons')) {
            console.log('✅ Material Icons are working');
            return true;
        } else {
            console.error('❌ Material Icons are not working');
            return false;
        }
    } catch (error) {
        console.error('❌ Material Icons test error:', error);
        return false;
    } finally {
        if (document.body.contains(testIcon)) {
            document.body.removeChild(testIcon);
        }
    }
}

// Test 2: API Client Initialization
async function testAPIClient() {
    console.log('🔍 Testing API client initialization...');
    
    try {
        if (typeof SAMOAPIClient === 'undefined') {
            console.error('❌ SAMOAPIClient not defined');
            return false;
        }
        
        const apiClient = new SAMOAPIClient();
        await apiClient.waitForInitialization();
        
        console.log('✅ API client initialized successfully');
        console.log('📊 Base URL:', apiClient.baseURL);
        return true;
    } catch (error) {
        console.error('❌ API client initialization failed:', error);
        return false;
    }
}

// Test 3: Chart.js
function testChartJS() {
    console.log('🔍 Testing Chart.js...');
    
    try {
        if (typeof Chart === 'undefined') {
            console.error('❌ Chart.js not loaded');
            return false;
        }
        
        console.log('✅ Chart.js is loaded');
        return true;
    } catch (error) {
        console.error('❌ Chart.js test error:', error);
        return false;
    }
}

// Test 4: Mock Data Generation
async function testMockData() {
    console.log('🔍 Testing mock data generation...');
    
    try {
        const apiClient = new SAMOAPIClient();
        await apiClient.waitForInitialization();
        
        const testText = "I am so happy and excited today!";
        
        // Test emotion detection mock
        const emotionResult = apiClient.getMockEmotionResponse(testText);
        if (emotionResult && emotionResult.emotions && emotionResult.emotions.length > 0) {
            console.log('✅ Emotion detection mock data working');
        } else {
            console.error('❌ Emotion detection mock data failed');
            return false;
        }
        
        // Test summarization mock
        const summaryResult = apiClient.getMockSummaryResponse(testText);
        if (summaryResult && summaryResult.summary) {
            console.log('✅ Text summarization mock data working');
        } else {
            console.error('❌ Text summarization mock data failed');
            return false;
        }
        
        return true;
    } catch (error) {
        console.error('❌ Mock data test error:', error);
        return false;
    }
}

// Test 5: API Fallback (Rate Limited)
async function testAPIFallback() {
    console.log('🔍 Testing API fallback (rate limited scenario)...');
    
    try {
        const apiClient = new SAMOAPIClient();
        await apiClient.waitForInitialization();
        
        const testText = "I am so happy and excited today!";
        
        // This should fall back to mock data due to rate limiting
        const result = await apiClient.detectEmotions(testText);
        
        if (result && result.emotions) {
            console.log('✅ API fallback working (using mock data)');
            console.log('📊 Result:', result);
            return true;
        } else {
            console.error('❌ API fallback failed');
            return false;
        }
    } catch (error) {
        console.error('❌ API fallback test error:', error);
        return false;
    }
}

// Test 6: Chart Rendering
function testChartRendering() {
    console.log('🔍 Testing chart rendering...');
    
    try {
        if (typeof ChartUtils === 'undefined') {
            console.error('❌ ChartUtils not defined');
            return false;
        }
        
        const chartUtils = new ChartUtils();
        
        // Create a test canvas
        const testCanvas = document.createElement('canvas');
        testCanvas.id = 'testChart';
        testCanvas.width = 400;
        testCanvas.height = 300;
        document.body.appendChild(testCanvas);
        
        // Test emotion chart creation
        const mockEmotions = [
            { emotion: 'joy', confidence: 0.85 },
            { emotion: 'excitement', confidence: 0.72 },
            { emotion: 'optimism', confidence: 0.68 }
        ];
        
        const chartCreated = chartUtils.createEmotionChart('testChart', mockEmotions);
        
        if (chartCreated) {
            console.log('✅ Chart rendering working');
            return true;
        } else {
            console.error('❌ Chart rendering failed');
            return false;
        }
    } catch (error) {
        console.error('❌ Chart rendering test error:', error);
        return false;
    } finally {
        // Clean up test canvas
        const testCanvas = document.getElementById('testChart');
        if (testCanvas && document.body.contains(testCanvas)) {
            document.body.removeChild(testCanvas);
        }
    }
}

// Run all tests
async function runAllTests() {
    console.log('🚀 Running comprehensive test suite...');
    
    const results = {
        materialIcons: testMaterialIcons(),
        apiClient: await testAPIClient(),
        chartJS: testChartJS(),
        mockData: await testMockData(),
        apiFallback: await testAPIFallback(),
        chartRendering: testChartRendering()
    };
    
    const passedTests = Object.values(results).filter(result => result === true).length;
    const totalTests = Object.keys(results).length;
    
    console.log('\n📊 Test Results Summary:');
    console.log('========================');
    Object.entries(results).forEach(([test, passed]) => {
        console.log(`${passed ? '✅' : '❌'} ${test}: ${passed ? 'PASS' : 'FAIL'}`);
    });
    
    console.log(`\n🎯 Overall: ${passedTests}/${totalTests} tests passed`);
    
    if (passedTests === totalTests) {
        console.log('🎉 All tests passed! The SAMO Demo is working perfectly!');
    } else {
        console.log('⚠️ Some tests failed. Check the logs above for details.');
    }
    
    return results;
}

// Export for use in browser console
window.runSAMOTests = runAllTests;

// Auto-run tests when script loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', runAllTests);
} else {
    runAllTests();
}
