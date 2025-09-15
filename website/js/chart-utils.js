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

        // Force HTML/CSS charts for now to ensure they work
        console.log('Using HTML/CSS emotion chart (forced)');
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
        console.log('Creating SIMPLE HTML/CSS emotion chart');
        const container = document.getElementById(containerId);

        // Sort emotions by confidence
        const sortedEmotions = emotions.sort((a, b) => b.confidence - a.confidence);
        
        // Create SIMPLE chart HTML that will definitely work
        let chartHTML = `
            <div style="background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; margin: 10px 0; border: 1px solid rgba(255, 255, 255, 0.2);">
                <h5 style="color: #fbbf24; text-align: center; margin-bottom: 20px;">📊 Emotion Analysis</h5>
                <div style="color: #cbd5e1; text-align: center; margin-bottom: 20px; font-size: 0.9rem;">Confidence levels for detected emotions</div>
        `;
        
        sortedEmotions.forEach((emotion, index) => {
            const percentage = Math.round(emotion.confidence * 100);
            const barWidth = Math.max(percentage, 5); // Use percentage, minimum 5%
            
            chartHTML += `
                <div style="margin-bottom: 15px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="font-weight: bold; color: #f1f5f9; text-transform: capitalize;">${emotion.emotion}</span>
                        <span style="color: #a855f7; font-weight: bold; background: rgba(168, 85, 247, 0.1); padding: 4px 12px; border-radius: 20px;">${percentage}%</span>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.3); border-radius: 10px; height: 20px; overflow: hidden;">
                        <div style="height: 100%; width: ${barWidth}%; background: linear-gradient(90deg, ${this.getEmotionColor(emotion.emotion)}, ${this.getEmotionColor(emotion.emotion, true)}); border-radius: 10px; transition: width 1s ease-out;"></div>
                    </div>
                </div>
            `;
        });
        
        chartHTML += `
                <div style="text-align: center; margin-top: 15px; color: #94a3b8; font-size: 0.85rem;">
                    Based on ${emotions.length} detected emotions
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

        // Force HTML/CSS charts for now to ensure they work
        console.log('Using HTML/CSS summary chart (forced)');
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
        console.log('Creating SIMPLE HTML/CSS summary chart');
        const container = document.getElementById(containerId);
        
        const originalLength = Number(summaryData?.original_length ?? 0);
        const summaryLength = Number(summaryData?.summary_length ?? 0);
        const compressionRatio = originalLength > 0 ? Math.round((1 - summaryLength / originalLength) * 100) : 0;
        
        let chartHTML = `
            <div style="background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; margin: 10px 0; border: 1px solid rgba(255, 255, 255, 0.2);">
                <h5 style="color: #fbbf24; text-align: center; margin-bottom: 20px;">📝 Text Compression</h5>
                <div style="color: #cbd5e1; text-align: center; margin-bottom: 20px; font-size: 0.9rem;">Original vs Summary length comparison</div>
                
                <div style="display: flex; justify-content: space-around; margin-bottom: 20px; gap: 15px;">
                    <div style="text-align: center; flex: 1; background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px;">
                        <div style="font-size: 1.8rem; font-weight: bold; color: #fbbf24; margin-bottom: 8px;">${originalLength}</div>
                        <div style="font-size: 0.8rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;">Original Words</div>
                    </div>
                    <div style="text-align: center; flex: 1; background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px;">
                        <div style="font-size: 1.8rem; font-weight: bold; color: #fbbf24; margin-bottom: 8px;">${summaryLength}</div>
                        <div style="font-size: 0.8rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;">Summary Words</div>
                    </div>
                    <div style="text-align: center; flex: 1; background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(168, 85, 247, 0.1)); border: 2px solid rgba(139, 92, 246, 0.4); border-radius: 8px; padding: 15px;">
                        <div style="font-size: 2rem; font-weight: bold; color: #c084fc; margin-bottom: 8px;">${compressionRatio}%</div>
                        <div style="font-size: 0.8rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;">Compression</div>
                    </div>
                </div>
                
                <div style="margin: 20px 0;">
                    <div style="font-weight: bold; color: #f1f5f9; margin-bottom: 10px;">Original Text</div>
                    <div style="background: rgba(0, 0, 0, 0.3); border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 5px;">
                        <div style="height: 100%; width: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa, #93c5fd); border-radius: 8px;"></div>
                    </div>
                    <div style="text-align: right; font-size: 0.9rem; color: #cbd5e1;">${originalLength} words</div>
                </div>
                
                <div style="margin: 20px 0;">
                    <div style="font-weight: bold; color: #f1f5f9; margin-bottom: 10px;">Summary</div>
                    <div style="background: rgba(0, 0, 0, 0.3); border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 5px;">
                        <div style="height: 100%; width: ${originalLength > 0 ? (summaryLength / originalLength) * 100 : 0}%; background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7); border-radius: 8px; transition: width 1.5s ease-out;"></div>
                    </div>
                    <div style="text-align: right; font-size: 0.9rem; color: #cbd5e1;">${summaryLength} words</div>
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