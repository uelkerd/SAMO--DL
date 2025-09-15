/**
 * Performance Optimizer for SAMO-DL Demo Website
 * 
 * This module provides performance optimizations for:
 * - Chart rendering with large datasets
 * - API response processing
 * - Memory management
 * - Lazy loading
 */

class PerformanceOptimizer {
    constructor() {
        this.chartCache = new Map();
        this.processingQueue = [];
        this.isProcessing = false;
        this.memoryThreshold = 50 * 1024 * 1024; // 50MB threshold
        this.maxCacheSize = 10;
        
        this.initializePerformanceMonitoring();
    }
    
    /**
     * Initialize performance monitoring
     */
    initializePerformanceMonitoring() {
        // Monitor memory usage
        if (performance.memory) {
            setInterval(() => this.checkMemoryUsage(), 5000);
        }
        
        // Monitor chart rendering performance
        this.chartRenderTimes = [];
        this.maxRenderTime = 1000; // 1 second max render time
    }
    
    /**
     * Optimize emotion data processing for large datasets
     * @param {Array} emotionData - Raw emotion data
     * @returns {Array} Optimized emotion data
     */
    optimizeEmotionData(emotionData) {
        const startTime = performance.now();
        
        try {
            // Limit to top emotions to improve performance
            const maxEmotions = 20;
            const sortedEmotions = emotionData
                .sort((a, b) => (b.confidence || b.score || 0) - (a.confidence || a.score || 0))
                .slice(0, maxEmotions);
            
            // Normalize confidence values efficiently
            const optimizedEmotions = sortedEmotions.map(emotion => ({
                emotion: emotion.emotion || emotion.label || 'Unknown',
                confidence: Math.max(0, Math.min(1, emotion.confidence || emotion.score || 0))
            }));
            
            const processingTime = performance.now() - startTime;
            this.logPerformance('emotion_processing', processingTime, emotionData.length);
            
            return optimizedEmotions;
            
        } catch (error) {
            console.error('Error optimizing emotion data:', error);
            return emotionData.slice(0, 10); // Fallback to first 10 items
        }
    }
    
    /**
     * Optimize chart rendering with performance monitoring
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} emotionData - Emotion data to render
     * @param {Object} options - Chart options
     * @returns {Object} Chart instance
     */
    optimizeChartRendering(ctx, emotionData, options = {}) {
        const startTime = performance.now();
        
        try {
            // Check if we should use cached chart
            const cacheKey = this.generateCacheKey(emotionData, options);
            if (this.chartCache.has(cacheKey)) {
                console.log('Using cached chart');
                return this.chartCache.get(cacheKey);
            }
            
            // Optimize data before rendering
            const optimizedData = this.optimizeEmotionData(emotionData);
            
            // Use requestAnimationFrame for smooth rendering
            return new Promise((resolve) => {
                requestAnimationFrame(() => {
                    const chart = this.createOptimizedChart(ctx, optimizedData, options);
                    
                    // Cache the chart if it's not too large
                    if (this.chartCache.size < this.maxCacheSize) {
                        this.chartCache.set(cacheKey, chart);
                    }
                    
                    const renderTime = performance.now() - startTime;
                    this.logPerformance('chart_rendering', renderTime, optimizedData.length);
                    
                    resolve(chart);
                });
            });
            
        } catch (error) {
            console.error('Error optimizing chart rendering:', error);
            throw error;
        }
    }
    
    /**
     * Create optimized chart with performance considerations
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} emotionData - Optimized emotion data
     * @param {Object} options - Chart options
     * @returns {Object} Chart instance
     */
    createOptimizedChart(ctx, emotionData, options) {
        // Simplified chart creation for better performance
        const labels = emotionData.map(e => e.emotion);
        const data = emotionData.map(e => e.confidence * 100);
        const colors = labels.map(label => this.getEmotionColor(label));
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence (%)',
                    data: data,
                    backgroundColor: colors,
                    borderColor: colors.map(c => this.adjustColorOpacity(c, 1)),
                    borderWidth: 2,
                    borderRadius: 8,
                    borderSkipped: false,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 300, // Reduced animation duration
                    easing: 'easeOutQuart'
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(139, 92, 246, 0.1)',
                            borderColor: 'rgba(139, 92, 246, 0.2)'
                        },
                        ticks: {
                            color: '#cbd5e1',
                            maxRotation: 45,
                            maxTicksLimit: 10 // Limit number of ticks
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(139, 92, 246, 0.1)',
                            borderColor: 'rgba(139, 92, 246, 0.2)'
                        },
                        ticks: {
                            color: '#cbd5e1',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 15, 35, 0.9)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#e2e8f0',
                        borderColor: 'rgba(139, 92, 246, 0.5)',
                        borderWidth: 1,
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed.y.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Process API responses with performance optimization
     * @param {Object} response - API response
     * @param {string} endpoint - API endpoint
     * @returns {Object} Processed response
     */
    optimizeApiResponseProcessing(response, endpoint) {
        const startTime = performance.now();
        
        try {
            let processedResponse = response;
            
            // Optimize based on endpoint type
            switch (endpoint) {
                case '/analyze/journal':
                    processedResponse = this.optimizeEmotionResponse(response);
                    break;
                case '/summarize/text':
                    processedResponse = this.optimizeSummaryResponse(response);
                    break;
                case '/transcribe/voice':
                    processedResponse = this.optimizeTranscriptionResponse(response);
                    break;
                default:
                    processedResponse = response;
            }
            
            const processingTime = performance.now() - startTime;
            this.logPerformance('api_response_processing', processingTime, endpoint);
            
            return processedResponse;
            
        } catch (error) {
            console.error('Error optimizing API response:', error);
            return response;
        }
    }
    
    /**
     * Optimize emotion detection response
     * @param {Object} response - Raw emotion response
     * @returns {Object} Optimized response
     */
    optimizeEmotionResponse(response) {
        if (!response || !response.emotions) {
            return response;
        }
        
        // Limit emotions to top performers
        const maxEmotions = 15;
        const sortedEmotions = response.emotions
            .sort((a, b) => (b.confidence || b.score || 0) - (a.confidence || a.score || 0))
            .slice(0, maxEmotions);
        
        return {
            ...response,
            emotions: sortedEmotions
        };
    }
    
    /**
     * Optimize summarization response
     * @param {Object} response - Raw summary response
     * @returns {Object} Optimized response
     */
    optimizeSummaryResponse(response) {
        if (!response || !response.summary) {
            return response;
        }
        
        // Truncate very long summaries
        const maxSummaryLength = 500;
        if (response.summary.length > maxSummaryLength) {
            response.summary = response.summary.substring(0, maxSummaryLength) + '...';
        }
        
        return response;
    }
    
    /**
     * Optimize transcription response
     * @param {Object} response - Raw transcription response
     * @returns {Object} Optimized response
     */
    optimizeTranscriptionResponse(response) {
        if (!response) {
            return response;
        }
        
        // Normalize text field names
        const text = response.text || response.transcription || '';
        
        return {
            ...response,
            text: text,
            confidence: response.confidence || 0,
            duration: response.duration || 0
        };
    }
    
    /**
     * Implement lazy loading for charts
     * @param {string} chartId - Chart container ID
     * @param {Function} chartCreator - Function to create the chart
     * @param {Object} options - Lazy loading options
     */
    implementLazyLoading(chartId, chartCreator, options = {}) {
        const chartContainer = document.getElementById(chartId);
        if (!chartContainer) {
            console.warn(`Chart container ${chartId} not found`);
            return;
        }
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Chart is visible, create it
                    chartCreator();
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: options.threshold || 0.1,
            rootMargin: options.rootMargin || '50px'
        });
        
        observer.observe(chartContainer);
    }
    
    /**
     * Check memory usage and clean up if necessary
     */
    checkMemoryUsage() {
        if (!performance.memory) {
            return;
        }
        
        const usedMemory = performance.memory.usedJSHeapSize;
        const totalMemory = performance.memory.totalJSHeapSize;
        const memoryUsagePercent = (usedMemory / totalMemory) * 100;
        
        if (memoryUsagePercent > 80) {
            console.warn('High memory usage detected, cleaning up...');
            this.cleanupMemory();
        }
    }
    
    /**
     * Clean up memory by clearing caches and unused objects
     */
    cleanupMemory() {
        // Clear chart cache
        this.chartCache.clear();
        
        // Clear processing queue
        this.processingQueue = [];
        
        // Force garbage collection if available
        if (window.gc) {
            window.gc();
        }
        
        console.log('Memory cleanup completed');
    }
    
    /**
     * Generate cache key for chart data
     * @param {Array} data - Chart data
     * @param {Object} options - Chart options
     * @returns {string} Cache key
     */
    generateCacheKey(data, options) {
        const dataString = JSON.stringify(data);
        const optionsString = JSON.stringify(options);
        return btoa(dataString + optionsString).substring(0, 32);
    }
    
    /**
     * Get emotion color with caching
     * @param {string} emotion - Emotion name
     * @returns {string} Color string
     */
    getEmotionColor(emotion) {
        const colors = {
            'joy': 'rgba(34, 197, 94, 0.8)',
            'happiness': 'rgba(34, 197, 94, 0.8)',
            'excitement': 'rgba(34, 197, 94, 0.8)',
            'sadness': 'rgba(59, 130, 246, 0.8)',
            'grief': 'rgba(59, 130, 246, 0.8)',
            'anger': 'rgba(239, 68, 68, 0.8)',
            'annoyance': 'rgba(239, 68, 68, 0.8)',
            'fear': 'rgba(245, 158, 11, 0.8)',
            'nervousness': 'rgba(245, 158, 11, 0.8)',
            'surprise': 'rgba(139, 92, 246, 0.8)',
            'love': 'rgba(244, 63, 94, 0.8)',
            'caring': 'rgba(244, 63, 94, 0.8)',
            'gratitude': 'rgba(16, 185, 129, 0.8)',
            'pride': 'rgba(16, 185, 129, 0.8)',
            'optimism': 'rgba(16, 185, 129, 0.8)',
            'disgust': 'rgba(107, 114, 128, 0.8)',
            'confusion': 'rgba(107, 114, 128, 0.8)',
            'neutral': 'rgba(107, 114, 128, 0.8)'
        };
        return colors[emotion] || 'rgba(139, 92, 246, 0.8)';
    }
    
    /**
     * Adjust color opacity
     * @param {string} color - Color string
     * @param {number} opacity - New opacity
     * @returns {string} Adjusted color
     */
    adjustColorOpacity(color, opacity) {
        if (color.startsWith('rgba(')) {
            return color.replace(/rgba\((\d+\s*,\s*\d+\s*,\s*\d+),\s*[\d.]+\)/, `rgba($1, ${opacity})`);
        } else if (color.startsWith('rgb(')) {
            const rgb = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
            if (rgb) {
                return `rgba(${rgb[1]}, ${rgb[2]}, ${rgb[3]}, ${opacity})`;
            }
        }
        return color;
    }
    
    /**
     * Log performance metrics
     * @param {string} operation - Operation name
     * @param {number} time - Execution time in ms
     * @param {number} dataSize - Size of data processed
     */
    logPerformance(operation, time, dataSize = 0) {
        const metric = {
            operation,
            time,
            dataSize,
            timestamp: Date.now()
        };
        
        // Store performance metrics
        if (!this.performanceMetrics) {
            this.performanceMetrics = [];
        }
        
        this.performanceMetrics.push(metric);
        
        // Keep only last 100 metrics
        if (this.performanceMetrics.length > 100) {
            this.performanceMetrics = this.performanceMetrics.slice(-100);
        }
        
        // Log slow operations
        if (time > this.maxRenderTime) {
            console.warn(`Slow operation detected: ${operation} took ${time.toFixed(2)}ms`);
        }
        
        console.log(`Performance: ${operation} - ${time.toFixed(2)}ms (${dataSize} items)`);
    }
    
    /**
     * Get performance summary
     * @returns {Object} Performance summary
     */
    getPerformanceSummary() {
        if (!this.performanceMetrics || this.performanceMetrics.length === 0) {
            return { message: 'No performance data available' };
        }
        
        const operations = {};
        
        this.performanceMetrics.forEach(metric => {
            if (!operations[metric.operation]) {
                operations[metric.operation] = {
                    count: 0,
                    totalTime: 0,
                    avgTime: 0,
                    maxTime: 0,
                    minTime: Infinity
                };
            }
            
            const op = operations[metric.operation];
            op.count++;
            op.totalTime += metric.time;
            op.maxTime = Math.max(op.maxTime, metric.time);
            op.minTime = Math.min(op.minTime, metric.time);
        });
        
        // Calculate averages
        Object.values(operations).forEach(op => {
            op.avgTime = op.totalTime / op.count;
        });
        
        return operations;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceOptimizer;
} else {
    window.PerformanceOptimizer = PerformanceOptimizer;
}
