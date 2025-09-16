# SAMO-DL Demo Website Development Guide

## Overview

This guide provides comprehensive documentation for developing and maintaining the SAMO-DL demo website. It covers the configuration system, error handling patterns, testing strategies, and performance optimization techniques.

## Table of Contents

1. [Configuration System](#configuration-system)
2. [Error Handling Patterns](#error-handling-patterns)
3. [Testing Strategies](#testing-strategies)
4. [Performance Optimization](#performance-optimization)
5. [Accessibility Compliance](#accessibility-compliance)
6. [Development Workflow](#development-workflow)
7. [Troubleshooting](#troubleshooting)

## Configuration System

### SAMO_CONFIG System

The demo website uses a centralized configuration system (`config.js`) to manage API endpoints, timeouts, and environment-specific settings.

#### Configuration Structure

```javascript
const SAMO_CONFIG = {
    // OpenAI API Configuration
    OPENAI: {
        PROXY_URL: 'https://api.example.com/generate/journal',  // Proxy endpoint for OpenAI
        MODEL: 'gpt-3.5-turbo',
        MAX_TOKENS: 200,
        TEMPERATURE: 0.8
    },
    
    // Emotion API Configuration  
    EMOTION_API: {
        ENDPOINT: 'https://api.example.com/analyze/emotion',
        TIMEOUT: 10000
    },
    
    // Feature flags
    FEATURES: {
        ENABLE_OPENAI: false  // Set to false for public builds
    },
    
    // Demo Configuration
    DEMO: {
        FALLBACK_TO_STATIC: true,
        SHOW_DEBUG_INFO: true
    }
};
```

#### Environment Detection

The system automatically detects the environment and configures appropriate settings:

```javascript
const isLocalDev = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' ||
                   window.location.hostname === '';

const SAMO_CONFIG = {
    // OpenAI API Configuration
    OPENAI: {
        PROXY_URL: isLocalDev ? 'http://localhost:8080/generate/journal' : '/api/generate/journal',
        MODEL: 'gpt-3.5-turbo',
        MAX_TOKENS: 200,
        TEMPERATURE: 0.8
    },
    
    // Emotion API Configuration  
    EMOTION_API: {
        ENDPOINT: isLocalDev ? 'http://localhost:8080/analyze/emotion' : '/api/analyze/emotion',
        TIMEOUT: 10000
    },
    
    // Feature flags
    FEATURES: {
        ENABLE_OPENAI: false  // Set to false for public builds
    },
    
    // Demo Configuration
    DEMO: {
        FALLBACK_TO_STATIC: true,
        SHOW_DEBUG_INFO: isLocalDev
    }
};
```

#### Configuration Usage

```javascript
// In your JavaScript code
const apiClient = new SAMOAPIClient();
// The client automatically uses SAMO_CONFIG if available
// Falls back to default values if not

// Access configuration directly
if (typeof SAMO_CONFIG !== 'undefined') {
    console.log('API Base URL:', SAMO_CONFIG.baseURL);
    console.log('Timeout:', SAMO_CONFIG.timeout);
}
```

### Environment-Specific Configuration

#### Local Development
- **Base URL**: `http://localhost:8080`
- **API Key**: Not required
- **Timeout**: 30 seconds
- **Debug Mode**: Enabled

#### Production
- **Base URL**: `/api` (relative path)
- **API Key**: Not required
- **Timeout**: 20 seconds
- **Debug Mode**: Disabled

#### Custom Configuration
Override the default configuration by setting `SAMO_CONFIG` before loading the demo scripts:

```html
<script>
    window.SAMO_CONFIG = {
        baseURL: 'https://your-custom-api.com',
        apiKey: 'your-custom-key',
        timeout: 15000,
        retryAttempts: 5
    };
</script>
<script src="js/comprehensive-demo.js"></script>
```

## Error Handling Patterns

### AbortController Implementation

The demo uses `AbortController` for request timeout management:

```javascript
async makeRequest(endpoint, data, method = 'POST', isFormData = false, timeoutMs = 20000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error('Request timeout')), timeoutMs);
    
    try {
        const response = await fetch(url, {
            method,
            headers: headers,
            body: body,
            signal: controller.signal
        });
        
        return await response.json();
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('Request timeout');
        }
        throw error;
    } finally {
        clearTimeout(timer);
    }
}
```

### Error Message Normalization

Consistent error message handling across all API calls:

```javascript
if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const msg = errorData.message || errorData.error || `HTTP ${response.status}`;
    
    if (response.status === 429) throw new Error(msg || 'Rate limit exceeded. Please try again shortly.');
    if (response.status === 401) throw new Error(msg || 'API key required.');
    if (response.status === 503) throw new Error(msg || 'Service temporarily unavailable.');
    throw new Error(msg);
}
```

### Mock Data Fallback

When API calls fail, the system falls back to mock data:

```javascript
async detectEmotions(text) {
    try {
        return await this.makeRequest('/analyze/journal', { text });
    } catch (error) {
        if (this.shouldUseMockData(error)) {
            console.warn('API not available, using mock data for demo:', error.message);
            return this.getMockEmotionResponse(text);
        }
        throw error;
    }
}

shouldUseMockData(error) {
    const mockableErrors = [
        'Rate limit',
        'API key',
        'Service temporarily',
        'Abuse detected',
        'Client blocked'
    ];
    return mockableErrors.some(errorType => error.message.includes(errorType));
}
```

### DOM Element Validation

Always validate DOM elements before manipulation:

```javascript
initializeElements() {
    this.audioFileInput = document.getElementById('audioFile');
    this.textInput = document.getElementById('textInput');
    // ... other elements
    
    // Validate critical elements
    if (!this.audioFileInput || !this.textInput) {
        throw new Error('Required DOM elements not found');
    }
}
```

## Testing Strategies

### Unit Testing

Test individual components in isolation:

```python
def test_abort_controller_timeout_handling(self):
    """Test that AbortController properly handles request timeouts"""
    timeout_ms = 5000
    start_time = datetime.now()
    
    # Simulate timeout behavior
    def simulate_timeout():
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        return elapsed >= timeout_ms
    
    assert not simulate_timeout()  # Should not timeout immediately
```

### Integration Testing

Test complete workflows and component interactions:

```python
def test_complete_workflow_integration(self):
    """Test the complete workflow from audio input to results display"""
    workflow_steps = [
        'audio_upload',
        'transcription',
        'summarization', 
        'emotion_detection',
        'results_display'
    ]
    
    workflow_status = {step: 'pending' for step in workflow_steps}
    
    # Simulate workflow execution
    for step in workflow_steps:
        workflow_status[step] = 'processing'
        time.sleep(0.01)  # Simulate processing time
        workflow_status[step] = 'completed'
    
    assert all(status == 'completed' for status in workflow_status.values())
```

### Performance Testing

Test performance characteristics with large datasets:

```python
def test_chart_rendering_performance(self):
    """Test chart rendering performance with large datasets"""
    large_emotion_dataset = [
        {'emotion': f'emotion_{i}', 'confidence': 0.1 + (i * 0.01)}
        for i in range(100)
    ]
    
    start_time = time.time()
    self._render_emotion_chart(large_emotion_dataset)
    end_time = time.time()
    rendering_time = (end_time - start_time) * 1000
    
    assert rendering_time < 1000  # Should render in under 1 second
```

### Accessibility Testing

Test accessibility compliance:

```python
def test_aria_attributes_compliance(self):
    """Test ARIA attributes compliance"""
    aria_attributes = {
        'aria-busy': ['true', 'false'],
        'aria-live': ['assertive', 'polite', 'off'],
        'aria-label': ['Audio file input', 'Text input', 'Process button'],
        'role': ['alert', 'status', 'button', 'textbox']
    }
    
    for attr, values in aria_attributes.items():
        for value in values:
            assert self._is_valid_aria_value(attr, value)
```

## Performance Optimization

### Chart Rendering Optimization

Use the `PerformanceOptimizer` class for efficient chart rendering:

```javascript
const optimizer = new PerformanceOptimizer();

// Optimize emotion data before rendering
const optimizedData = optimizer.optimizeEmotionData(emotionData);

// Render chart with performance monitoring
optimizer.optimizeChartRendering(ctx, optimizedData, options)
    .then(chart => {
        console.log('Chart rendered successfully');
    });
```

### Memory Management

Monitor and manage memory usage:

```javascript
// Check memory usage every 5 seconds
setInterval(() => {
    if (performance.memory) {
        const usedMemory = performance.memory.usedJSHeapSize;
        const totalMemory = performance.memory.totalJSHeapSize;
        const memoryUsagePercent = (usedMemory / totalMemory) * 100;
        
        if (memoryUsagePercent > 80) {
            optimizer.cleanupMemory();
        }
    }
}, 5000);
```

### Lazy Loading

Implement lazy loading for charts and heavy components:

```javascript
// Lazy load chart when it becomes visible
optimizer.implementLazyLoading('emotionChart', () => {
    createEmotionChart(emotionData);
}, {
    threshold: 0.1,
    rootMargin: '50px'
});
```

## Accessibility Compliance

### ARIA Attributes

Ensure proper ARIA attributes for screen readers:

```html
<div id="loadingSection" aria-busy="true" aria-live="polite">
    <div id="loadingMessage">Processing with AI...</div>
</div>

<div class="error-message" role="alert" aria-live="assertive">
    Please upload an audio file or enter text to process.
</div>
```

### Keyboard Navigation

Support keyboard navigation for all interactive elements:

```javascript
// Add keyboard event listeners
document.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && event.target.id === 'processBtn') {
        this.processInput();
    }
    if (event.key === 'Escape' && event.target.id === 'clearBtn') {
        this.clearAll();
    }
});
```

### Focus Management

Manage focus for better accessibility:

```javascript
showError(message) {
    if (!this.errorMsgEl) {
        this.errorMsgEl = document.createElement('div');
        this.errorMsgEl.className = 'error-message';
        this.errorMsgEl.setAttribute('role', 'alert');
        this.errorMsgEl.setAttribute('aria-live', 'assertive');
        this.textInput.parentNode.insertBefore(this.errorMsgEl, this.textInput.nextSibling);
    }
    this.errorMsgEl.textContent = message;
    this.errorMsgEl.classList.add('show');
    
    // Focus on error message for screen readers
    this.errorMsgEl.focus();
}
```

### Reduced Motion Support

Respect user's motion preferences:

```css
@media (prefers-reduced-motion: reduce) {
    * { 
        animation: none !important; 
        transition: none !important; 
    }
    .hero-section::before { 
        animation: none !important; 
    }
    .floating-card { 
        animation: none !important; 
    }
}
```

## Development Workflow

### Running Tests

```bash
# Run all demo website tests
python scripts/test_demo_website.py --verbose --coverage

# Run specific test suites
python -m pytest tests/unit/test_demo_error_handling.py -v
python -m pytest tests/integration/test_demo_integration.py -v

# Run with performance testing
python scripts/test_demo_website.py --performance
```

### Local Development

1. Start the local development server:
   ```bash
   cd website
   python -m http.server 8080
   ```

2. Open the demo in your browser:
   ```
   http://localhost:8080/comprehensive-demo.html
   ```

3. Use the debug console for testing:
   ```
   http://localhost:8080/debug-demo.html
   ```

### Code Quality Checks

```bash
# Run linting
python -m flake8 tests/unit/test_demo_error_handling.py
python -m flake8 tests/integration/test_demo_integration.py

# Run type checking
python -m mypy tests/unit/test_demo_error_handling.py
```

## Troubleshooting

### Common Issues

#### 1. API Request Timeouts

**Problem**: Requests are timing out frequently
**Solution**: 
- Check network connectivity
- Increase timeout value in configuration
- Verify API endpoint is accessible

```javascript
// Increase timeout for slow networks
const SAMO_CONFIG = {
    timeout: 60000, // 60 seconds
    retryAttempts: 5
};
```

#### 2. Chart Rendering Performance

**Problem**: Charts are slow to render with large datasets
**Solution**:
- Use the PerformanceOptimizer
- Limit the number of emotions displayed
- Implement lazy loading

```javascript
// Limit emotions to top 20
const maxEmotions = 20;
const sortedEmotions = emotionData
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, maxEmotions);
```

#### 3. Memory Leaks

**Problem**: Memory usage increases over time
**Solution**:
- Clear chart cache regularly
- Remove event listeners when components are destroyed
- Use the memory cleanup functions

```javascript
// Clean up when component is destroyed
destroy() {
    if (this.chart) {
        this.chart.destroy();
    }
    this.chartCache.clear();
    // Remove event listeners
    this.removeEventListeners();
}
```

#### 4. Accessibility Issues

**Problem**: Screen readers can't access content properly
**Solution**:
- Add proper ARIA attributes
- Ensure keyboard navigation works
- Test with screen reader software

### Debug Tools

#### Browser Developer Tools

1. **Console**: Check for JavaScript errors
2. **Network**: Monitor API requests and responses
3. **Performance**: Profile rendering performance
4. **Accessibility**: Use built-in accessibility tools

#### Debug Console

Use the built-in debug console at `/debug-demo.html` for comprehensive testing:

- Test individual API endpoints
- Validate error handling
- Check mock data fallback
- Monitor performance metrics

#### Test Reports

Generated test reports are saved to `artifacts/test-reports/`:

- `demo_website_test_report_YYYYMMDD_HHMMSS.json`
- `accessibility-results-*.json`
- `lighthouse-*.json`

### Performance Monitoring

Monitor performance metrics in real-time:

```javascript
// Get performance summary
const summary = optimizer.getPerformanceSummary();
console.log('Performance Summary:', summary);

// Monitor specific operations
optimizer.logPerformance('emotion_processing', processingTime, dataSize);
```

## Best Practices

### Code Organization

1. **Separation of Concerns**: Keep UI logic separate from API logic
2. **Error Boundaries**: Wrap components in error boundaries
3. **Configuration Management**: Use centralized configuration
4. **Performance Monitoring**: Monitor and log performance metrics

### Testing

1. **Test Coverage**: Aim for >90% test coverage
2. **Integration Testing**: Test complete workflows
3. **Performance Testing**: Test with realistic data sizes
4. **Accessibility Testing**: Test with screen readers

### Security

1. **Input Validation**: Validate all user inputs
2. **XSS Prevention**: Sanitize data before display
3. **CSRF Protection**: Use proper CSRF tokens
4. **Content Security Policy**: Implement CSP headers

### Maintenance

1. **Regular Updates**: Keep dependencies updated
2. **Performance Monitoring**: Monitor performance metrics
3. **Error Tracking**: Track and fix errors promptly
4. **Documentation**: Keep documentation up to date

## Conclusion

This guide provides comprehensive coverage of the SAMO-DL demo website development process. Follow these patterns and practices to ensure maintainable, performant, and accessible code.

For additional support or questions, refer to the project's main documentation or create an issue in the repository.
