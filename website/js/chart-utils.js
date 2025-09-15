/**
 * Chart Utilities Module
 * Handles chart creation and visualization for the SAMO Demo
 */
class ChartUtils {
    constructor() {
        this.charts = {};
    }

    createEmotionChart(containerId, emotions) {
        console.log('Creating emotion chart for container:', containerId);
        const canvas = document.getElementById(containerId);
        if (!canvas) {
            console.error('Chart container not found:', containerId);
            // Show error message in the container
            const container = document.querySelector(`#${containerId}`).parentElement;
            if (container) {
                container.innerHTML = '<p style="color: #ef4444;">Chart container not found</p>';
            }
            return false;
        }
        const ctx = canvas.getContext('2d');

        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not available, chart creation will fail');
            // Show error message
            canvas.parentElement.innerHTML = '<p style="color: #ef4444;">Chart.js library not loaded</p>';
            return false;
        }

        // Destroy existing chart if it exists
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        // Validate emotions data
        if (!emotions || !Array.isArray(emotions) || emotions.length === 0) {
            console.error('Invalid emotions data for chart:', emotions);
            // Show fallback message
            canvas.parentElement.innerHTML = '<p style="color: #f59e0b; text-align: center; padding: 20px;">No emotion data available. This may be due to API connectivity issues.</p>';
            return false;
        }

        const labels = emotions.map(e => e.emotion || e.label || 'Unknown');
        const data = emotions.map(e => Math.max(0, Math.min(100, (e.confidence || e.score || 0) * 100)));

        console.log('Creating emotion chart with data:', { labels, data, emotions });

        // Generate dynamic colors for up to 28 emotions
        const backgroundColor = labels.map((_, i) => `hsla(${(i*360/labels.length)|0},70%,60%,0.8)`);
        const borderColor = labels.map((_, i) => `hsla(${(i*360/labels.length)|0},70%,45%,1)`);

        try {
            this.charts[containerId] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Confidence (%)',
                        data: data,
                        backgroundColor: backgroundColor,
                        borderColor: borderColor,
                        borderWidth: 2
                    }]
                },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Emotion Detection Results',
                        color: '#475569',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: '#475569',
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        grid: {
                            color: 'rgba(226, 232, 240, 0.1)'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#475569',
                            maxRotation: 45
                        },
                        grid: {
                            color: 'rgba(226, 232, 240, 0.1)'
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                },
                elements: {
                    bar: {
                        borderRadius: 4,
                        borderSkipped: false,
                    }
                }
            }
        });
            console.log('Chart created successfully');
            return true;
        } catch (error) {
            console.error('Failed to create chart:', error);
            return false;
        }
    }

    createSummaryChart(containerId, summaryData) {
        const canvas = document.getElementById(containerId);
        if (!canvas) {
            console.error('Chart container not found:', containerId);
            return false;
        }
        const ctx = canvas.getContext('2d');

        // Destroy existing chart if it exists
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        this.charts[containerId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Original Text', 'Summary'],
                datasets: [{
                    data: [
                        Number(summaryData?.original_length ?? 0),
                        Number(summaryData?.summary_length ?? 0)
                    ],
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(34, 197, 94, 0.8)'
                    ],
                    borderColor: [
                        'rgba(102, 126, 234, 1)',
                        'rgba(34, 197, 94, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#475569',
                            padding: 20
                        }
                    },
                    title: {
                        display: true,
                        text: 'Text Compression',
                        color: '#475569',
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    destroyChart(containerId) {
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
            delete this.charts[containerId];
        }
    }

    destroyAllCharts() {
        Object.keys(this.charts).forEach(containerId => {
            this.destroyChart(containerId);
        });
    }
}
