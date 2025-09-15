/**
 * Pure HTML/CSS Chart Utilities Module
 * No external dependencies - works 100% of the time!
 */
class ChartUtils {
    constructor() {
        this.charts = {};
    }

    createEmotionChart(containerId, emotions) {
        console.log('Creating emotion chart for container:', containerId, emotions);
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('Container not found:', containerId);
            return false;
        }

        // Try Highcharts first if available
        if (typeof Highcharts !== 'undefined') {
            return this.createHighchartsEmotionChart(containerId, emotions);
        }

        // Fallback to pure HTML/CSS charts
        return this.createHTMLCSSEmotionChart(containerId, emotions);
    }

    createHighchartsEmotionChart(containerId, emotions) {
        console.log('Creating Highcharts emotion chart');
        const container = document.getElementById(containerId);
        
        // Sort emotions by confidence
        const sortedEmotions = emotions.sort((a, b) => b.confidence - a.confidence);
        
        const chartData = sortedEmotions.map(emotion => ({
            name: emotion.emotion,
            y: Math.round(emotion.confidence * 100),
            color: this.getEmotionColor(emotion.emotion)
        }));

        Highcharts.chart(containerId, {
            chart: {
                type: 'bar',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                borderRadius: 16,
                style: {
                    fontFamily: 'Inter, system-ui, sans-serif'
                }
            },
            title: {
                text: '📊 Emotion Analysis',
                style: {
                    color: '#fbbf24',
                    fontSize: '1.25rem',
                    fontWeight: 'bold'
                }
            },
            subtitle: {
                text: 'Confidence levels for detected emotions',
                style: {
                    color: '#cbd5e1',
                    fontSize: '0.9rem'
                }
            },
            xAxis: {
                categories: sortedEmotions.map(e => e.emotion),
                labels: {
                    style: {
                        color: '#f1f5f9',
                        fontWeight: 'bold'
                    }
                }
            },
            yAxis: {
                title: {
                    text: 'Confidence (%)',
                    style: {
                        color: '#cbd5e1'
                    }
                },
                labels: {
                    style: {
                        color: '#cbd5e1'
                    }
                },
                gridLineColor: 'rgba(255, 255, 255, 0.1)'
            },
            series: [{
                name: 'Emotion Confidence',
                data: chartData,
                dataLabels: {
                    enabled: true,
                    format: '{y}%',
                    style: {
                        color: '#a855f7',
                        fontWeight: 'bold'
                    }
                }
            }],
            legend: {
                enabled: false
            },
            plotOptions: {
                bar: {
                    borderRadius: 8,
                    borderWidth: 0,
                    animation: {
                        duration: 1000
                    }
                }
            },
            credits: {
                enabled: false
            }
        });

        // Store reference for cleanup
        this.charts[containerId] = {
            type: 'emotion',
            container: container,
            chart: Highcharts.charts[Highcharts.charts.length - 1]
        };

        return true;
    }

    createHTMLCSSEmotionChart(containerId, emotions) {
        console.log('Creating HTML/CSS emotion chart');
        const container = document.getElementById(containerId);

        // Sort emotions by confidence
        const sortedEmotions = emotions.sort((a, b) => b.confidence - a.confidence);
        
        // Create the chart HTML with much better styling
        let chartHTML = `
            <div class="emotion-chart-container">
                <div class="chart-header">
                    <h5 class="chart-title">📊 Emotion Analysis</h5>
                    <div class="chart-subtitle">Confidence levels for detected emotions</div>
                </div>
                <div class="emotion-bars">
        `;
        
        sortedEmotions.forEach((emotion, index) => {
            const percentage = Math.round(emotion.confidence * 100);
            const barWidth = Math.max(percentage, 5); // Use percentage, minimum 5%
            const delay = index * 150; // Staggered animation
            
            chartHTML += `
                <div class="emotion-bar" style="animation-delay: ${delay}ms;">
                    <div class="emotion-label">
                        <span class="emotion-name">${emotion.emotion}</span>
                        <span class="emotion-percentage">${percentage}%</span>
                    </div>
                    <div class="emotion-bar-bg">
                        <div class="emotion-bar-fill" style="width: ${barWidth}%; background: linear-gradient(90deg, ${this.getEmotionColor(emotion.emotion)}, ${this.getEmotionColor(emotion.emotion, true)});"></div>
                    </div>
                </div>
            `;
        });
        
        chartHTML += `
                </div>
                <div class="chart-footer">
                    <small class="text-muted">Based on ${emotions.length} detected emotions</small>
                </div>
            </div>
        `;
        
        container.innerHTML = chartHTML;
        
        // Store reference for cleanup
        this.charts[containerId] = {
            type: 'emotion',
            container: container
        };
        
        return true;
    }

    createSummaryChart(containerId, summaryData) {
        console.log('Creating summary chart for container:', containerId);
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('Container not found:', containerId);
            return false;
        }

        // Try Highcharts first if available
        if (typeof Highcharts !== 'undefined') {
            return this.createHighchartsSummaryChart(containerId, summaryData);
        }

        // Fallback to pure HTML/CSS charts
        return this.createHTMLCSSSummaryChart(containerId, summaryData);
    }

    createHighchartsSummaryChart(containerId, summaryData) {
        console.log('Creating Highcharts summary chart');
        const container = document.getElementById(containerId);
        
        const originalLength = Number(summaryData?.original_length ?? 0);
        const summaryLength = Number(summaryData?.summary_length ?? 0);
        const compressionRatio = originalLength > 0 ? Math.round((1 - summaryLength / originalLength) * 100) : 0;

        Highcharts.chart(containerId, {
            chart: {
                type: 'column',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                borderRadius: 16,
                style: {
                    fontFamily: 'Inter, system-ui, sans-serif'
                }
            },
            title: {
                text: '📝 Text Compression',
                style: {
                    color: '#fbbf24',
                    fontSize: '1.25rem',
                    fontWeight: 'bold'
                }
            },
            subtitle: {
                text: `Compression: ${compressionRatio}%`,
                style: {
                    color: '#cbd5e1',
                    fontSize: '0.9rem'
                }
            },
            xAxis: {
                categories: ['Original Text', 'Summary'],
                labels: {
                    style: {
                        color: '#f1f5f9',
                        fontWeight: 'bold'
                    }
                }
            },
            yAxis: {
                title: {
                    text: 'Word Count',
                    style: {
                        color: '#cbd5e1'
                    }
                },
                labels: {
                    style: {
                        color: '#cbd5e1'
                    }
                },
                gridLineColor: 'rgba(255, 255, 255, 0.1)'
            },
            series: [{
                name: 'Word Count',
                data: [
                    { name: 'Original Text', y: originalLength, color: '#3b82f6' },
                    { name: 'Summary', y: summaryLength, color: '#10b981' }
                ],
                dataLabels: {
                    enabled: true,
                    format: '{y} words',
                    style: {
                        color: '#cbd5e1',
                        fontWeight: 'bold'
                    }
                }
            }],
            legend: {
                enabled: false
            },
            plotOptions: {
                column: {
                    borderRadius: 8,
                    borderWidth: 0,
                    animation: {
                        duration: 1000
                    }
                }
            },
            credits: {
                enabled: false
            }
        });

        // Store reference for cleanup
        this.charts[containerId] = {
            type: 'summary',
            container: container,
            chart: Highcharts.charts[Highcharts.charts.length - 1]
        };

        return true;
    }

    createHTMLCSSSummaryChart(containerId, summaryData) {
        console.log('Creating HTML/CSS summary chart');
        const container = document.getElementById(containerId);
        
        const originalLength = Number(summaryData?.original_length ?? 0);
        const summaryLength = Number(summaryData?.summary_length ?? 0);
        const compressionRatio = originalLength > 0 ? Math.round((1 - summaryLength / originalLength) * 100) : 0;
        
        let chartHTML = `
            <div class="summary-chart-container">
                <div class="chart-header">
                    <h6 class="chart-title">📝 Text Compression</h6>
                    <div class="chart-subtitle">Original vs Summary length comparison</div>
                </div>
                <div class="summary-stats">
                    <div class="stat-item">
                        <div class="stat-value">${originalLength}</div>
                        <div class="stat-label">Original Words</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${summaryLength}</div>
                        <div class="stat-label">Summary Words</div>
                    </div>
                    <div class="stat-item highlight">
                        <div class="stat-value">${compressionRatio}%</div>
                        <div class="stat-label">Compression</div>
                    </div>
                </div>
                <div class="summary-bars">
                    <div class="summary-bar">
                        <div class="bar-label">Original Text</div>
                        <div class="bar-bg">
                            <div class="bar-fill original" style="width: 100%;"></div>
                        </div>
                        <div class="bar-value">${originalLength} words</div>
                    </div>
                    <div class="summary-bar">
                        <div class="bar-label">Summary</div>
                        <div class="bar-bg">
                            <div class="bar-fill summary" style="width: ${originalLength > 0 ? (summaryLength / originalLength) * 100 : 0}%;"></div>
                        </div>
                        <div class="bar-value">${summaryLength} words</div>
                    </div>
                </div>
            </div>
        `;
        
        container.innerHTML = chartHTML;
        
        // Store reference for cleanup
        this.charts[containerId] = {
            type: 'summary',
            container: container
        };
        
        return true;
    }

    getEmotionColor(emotion, isLight = false) {
        const colors = {
            'joy': isLight ? '#fbbf24' : '#f59e0b',
            'happiness': isLight ? '#fbbf24' : '#f59e0b',
            'sadness': isLight ? '#60a5fa' : '#3b82f6',
            'anger': isLight ? '#f87171' : '#ef4444',
            'fear': isLight ? '#a78bfa' : '#8b5cf6',
            'surprise': isLight ? '#34d399' : '#10b981',
            'disgust': isLight ? '#fbbf24' : '#f59e0b',
            'neutral': isLight ? '#9ca3af' : '#6b7280',
            'excitement': isLight ? '#fbbf24' : '#f59e0b',
            'anxiety': isLight ? '#a78bfa' : '#8b5cf6',
            'calm': isLight ? '#60a5fa' : '#3b82f6',
            'frustration': isLight ? '#f87171' : '#ef4444'
        };
        
        return colors[emotion.toLowerCase()] || (isLight ? '#a78bfa' : '#8b5cf6');
    }

    destroyChart(containerId) {
        if (this.charts[containerId]) {
            // Destroy Highcharts chart if it exists
            if (this.charts[containerId].chart && typeof this.charts[containerId].chart.destroy === 'function') {
                this.charts[containerId].chart.destroy();
            }
            // Clear container
            this.charts[containerId].container.innerHTML = '';
            delete this.charts[containerId];
        }
    }

    destroyAllCharts() {
        Object.keys(this.charts).forEach(containerId => {
            this.destroyChart(containerId);
        });
    }
}