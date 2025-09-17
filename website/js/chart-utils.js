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

        // Clear container safely
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }

        // Create main chart container
        const chartDiv = document.createElement('div');
        chartDiv.style.cssText = 'background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; margin: 10px 0; border: 1px solid rgba(255, 255, 255, 0.2);';

        // Create title
        const title = document.createElement('h5');
        title.style.cssText = 'color: #fbbf24; text-align: center; margin-bottom: 20px;';
        title.textContent = '📊 Emotion Analysis';
        chartDiv.appendChild(title);

        // Create subtitle
        const subtitle = document.createElement('div');
        subtitle.style.cssText = 'color: #cbd5e1; text-align: center; margin-bottom: 20px; font-size: 0.9rem;';
        subtitle.textContent = 'Confidence levels for detected emotions';
        chartDiv.appendChild(subtitle);

        // Create emotion bars safely
        sortedEmotions.forEach((emotion, index) => {
            const percentage = Math.round(emotion.confidence * 100);
            const barWidth = Math.max(percentage, 5); // Use percentage, minimum 5%

            // Create emotion container
            const emotionDiv = document.createElement('div');
            emotionDiv.style.cssText = 'margin-bottom: 15px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px;';

            // Create label container
            const labelDiv = document.createElement('div');
            labelDiv.style.cssText = 'display: flex; justify-content: space-between; margin-bottom: 10px;';

            // Create emotion name span
            const nameSpan = document.createElement('span');
            nameSpan.style.cssText = 'font-weight: bold; color: #f1f5f9; text-transform: capitalize;';
            nameSpan.textContent = emotion.emotion; // Safe text content

            // Create percentage span
            const percentSpan = document.createElement('span');
            percentSpan.style.cssText = 'color: #a855f7; font-weight: bold; background: rgba(168, 85, 247, 0.1); padding: 4px 12px; border-radius: 20px;';
            percentSpan.textContent = `${percentage}%`;

            labelDiv.appendChild(nameSpan);
            labelDiv.appendChild(percentSpan);

            // Create progress bar container
            const progressContainer = document.createElement('div');
            progressContainer.style.cssText = 'background: rgba(0, 0, 0, 0.3); border-radius: 10px; height: 20px; overflow: hidden;';

            // Create progress bar
            const progressBar = document.createElement('div');
            progressBar.style.cssText = `height: 100%; width: ${barWidth}%; background: linear-gradient(90deg, ${this.getEmotionColor(emotion.emotion)}, ${this.getEmotionColor(emotion.emotion, true)}); border-radius: 10px; transition: width 1s ease-out;`;

            progressContainer.appendChild(progressBar);

            emotionDiv.appendChild(labelDiv);
            emotionDiv.appendChild(progressContainer);
            chartDiv.appendChild(emotionDiv);
        });

        // Create footer
        const footer = document.createElement('div');
        footer.style.cssText = 'text-align: center; margin-top: 15px; color: #94a3b8; font-size: 0.85rem;';
        footer.textContent = `Based on ${emotions.length} detected emotions`;
        chartDiv.appendChild(footer);

        container.appendChild(chartDiv);
        
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

        // Clear container safely
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }

        // Create main chart container
        const chartDiv = document.createElement('div');
        chartDiv.style.cssText = 'background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; margin: 10px 0; border: 1px solid rgba(255, 255, 255, 0.2);';

        // Create title
        const title = document.createElement('h5');
        title.style.cssText = 'color: #fbbf24; text-align: center; margin-bottom: 20px;';
        title.textContent = '📝 Text Compression';
        chartDiv.appendChild(title);

        // Create subtitle
        const subtitle = document.createElement('div');
        subtitle.style.cssText = 'color: #cbd5e1; text-align: center; margin-bottom: 20px; font-size: 0.9rem;';
        subtitle.textContent = 'Original vs Summary length comparison';
        chartDiv.appendChild(subtitle);

        // Create stats container
        const statsContainer = document.createElement('div');
        statsContainer.style.cssText = 'display: flex; justify-content: space-around; margin-bottom: 20px; gap: 15px;';

        // Original words stat
        const originalStat = document.createElement('div');
        originalStat.style.cssText = 'text-align: center; flex: 1; background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px;';

        const originalValue = document.createElement('div');
        originalValue.style.cssText = 'font-size: 1.8rem; font-weight: bold; color: #fbbf24; margin-bottom: 8px;';
        originalValue.textContent = originalLength.toString();

        const originalLabel = document.createElement('div');
        originalLabel.style.cssText = 'font-size: 0.8rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;';
        originalLabel.textContent = 'Original Words';

        originalStat.appendChild(originalValue);
        originalStat.appendChild(originalLabel);

        // Summary words stat
        const summaryStat = document.createElement('div');
        summaryStat.style.cssText = 'text-align: center; flex: 1; background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px;';

        const summaryValue = document.createElement('div');
        summaryValue.style.cssText = 'font-size: 1.8rem; font-weight: bold; color: #fbbf24; margin-bottom: 8px;';
        summaryValue.textContent = summaryLength.toString();

        const summaryLabel = document.createElement('div');
        summaryLabel.style.cssText = 'font-size: 0.8rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;';
        summaryLabel.textContent = 'Summary Words';

        summaryStat.appendChild(summaryValue);
        summaryStat.appendChild(summaryLabel);

        // Compression ratio stat
        const compressionStat = document.createElement('div');
        compressionStat.style.cssText = 'text-align: center; flex: 1; background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(168, 85, 247, 0.1)); border: 2px solid rgba(139, 92, 246, 0.4); border-radius: 8px; padding: 15px;';

        const compressionValue = document.createElement('div');
        compressionValue.style.cssText = 'font-size: 2rem; font-weight: bold; color: #c084fc; margin-bottom: 8px;';
        compressionValue.textContent = `${compressionRatio}%`;

        const compressionLabel = document.createElement('div');
        compressionLabel.style.cssText = 'font-size: 0.8rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 1px;';
        compressionLabel.textContent = 'Compression';

        compressionStat.appendChild(compressionValue);
        compressionStat.appendChild(compressionLabel);

        statsContainer.appendChild(originalStat);
        statsContainer.appendChild(summaryStat);
        statsContainer.appendChild(compressionStat);
        chartDiv.appendChild(statsContainer);

        // Create original text bar
        const originalSection = document.createElement('div');
        originalSection.style.cssText = 'margin: 20px 0;';

        const originalTitle = document.createElement('div');
        originalTitle.style.cssText = 'font-weight: bold; color: #f1f5f9; margin-bottom: 10px;';
        originalTitle.textContent = 'Original Text';

        const originalBarContainer = document.createElement('div');
        originalBarContainer.style.cssText = 'background: rgba(0, 0, 0, 0.3); border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 5px;';

        const originalBar = document.createElement('div');
        originalBar.style.cssText = 'height: 100%; width: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa, #93c5fd); border-radius: 8px;';

        const originalCount = document.createElement('div');
        originalCount.style.cssText = 'text-align: right; font-size: 0.9rem; color: #cbd5e1;';
        originalCount.textContent = `${originalLength} words`;

        originalBarContainer.appendChild(originalBar);
        originalSection.appendChild(originalTitle);
        originalSection.appendChild(originalBarContainer);
        originalSection.appendChild(originalCount);
        chartDiv.appendChild(originalSection);

        // Create summary text bar
        const summarySection = document.createElement('div');
        summarySection.style.cssText = 'margin: 20px 0;';

        const summaryTitle = document.createElement('div');
        summaryTitle.style.cssText = 'font-weight: bold; color: #f1f5f9; margin-bottom: 10px;';
        summaryTitle.textContent = 'Summary';

        const summaryBarContainer = document.createElement('div');
        summaryBarContainer.style.cssText = 'background: rgba(0, 0, 0, 0.3); border-radius: 8px; height: 24px; overflow: hidden; margin-bottom: 5px;';

        const summaryBar = document.createElement('div');
        const summaryWidth = originalLength > 0 ? (summaryLength / originalLength) * 100 : 0;
        summaryBar.style.cssText = `height: 100%; width: ${summaryWidth}%; background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7); border-radius: 8px; transition: width 1.5s ease-out;`;

        const summaryCount = document.createElement('div');
        summaryCount.style.cssText = 'text-align: right; font-size: 0.9rem; color: #cbd5e1;';
        summaryCount.textContent = `${summaryLength} words`;

        summaryBarContainer.appendChild(summaryBar);
        summarySection.appendChild(summaryTitle);
        summarySection.appendChild(summaryBarContainer);
        summarySection.appendChild(summaryCount);
        chartDiv.appendChild(summarySection);

        container.appendChild(chartDiv);
        
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
            // Clear container safely
            const container = this.charts[containerId].container;
            while (container.firstChild) {
                container.removeChild(container.firstChild);
            }
            delete this.charts[containerId];
        }
    }

    destroyAllCharts() {
        Object.keys(this.charts).forEach(containerId => {
            this.destroyChart(containerId);
        });
    }
}