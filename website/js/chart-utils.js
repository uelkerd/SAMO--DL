/**
 * Chart Utilities Module
 * Handles chart creation and visualization for the SAMO Demo
 */
class ChartUtils {
    constructor() {
        this.charts = {};
    }

    createEmotionChart(containerId, emotions) {
        const ctx = document.getElementById(containerId);
        if (!ctx) {
            console.error('Chart container not found:', containerId);
            return false;
        }

        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not available, chart creation will fail');
            return false;
        }

        // Destroy existing chart if it exists
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        // Validate emotions data
        if (!emotions || !Array.isArray(emotions) || emotions.length === 0) {
            console.error('Invalid emotions data for chart:', emotions);
            return false;
        }

        const labels = emotions.map(e => e.emotion || e.label || 'Unknown');
        const data = emotions.map(e => Math.max(0, Math.min(100, (e.confidence || e.score || 0) * 100)));

        console.log('Creating emotion chart with data:', { labels, data, emotions });

        try {
            this.charts[containerId] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Confidence (%)',
                        data: data,
                        backgroundColor: [
                            'rgba(102, 126, 234, 0.8)',
                            'rgba(168, 85, 247, 0.8)',
                            'rgba(192, 132, 252, 0.8)',
                            'rgba(34, 197, 94, 0.8)',
                            'rgba(251, 191, 36, 0.8)',
                            'rgba(239, 68, 68, 0.8)',
                            'rgba(59, 130, 246, 0.8)',
                            'rgba(16, 185, 129, 0.8)'
                        ],
                        borderColor: [
                            'rgba(102, 126, 234, 1)',
                            'rgba(168, 85, 247, 1)',
                            'rgba(192, 132, 252, 1)',
                            'rgba(34, 197, 94, 1)',
                            'rgba(251, 191, 36, 1)',
                            'rgba(239, 68, 68, 1)',
                            'rgba(59, 130, 246, 1)',
                            'rgba(16, 185, 129, 1)'
                        ],
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
                        color: '#e2e8f0',
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
                            color: '#e2e8f0',
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
                            color: '#e2e8f0',
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
            });
            console.log('Chart created successfully');
            return true;
        } catch (error) {
            console.error('Failed to create chart:', error);
            return false;
        }
    }

    createSummaryChart(containerId, summaryData) {
        const ctx = document.getElementById(containerId);
        if (!ctx) {
            return false;
        }

        // Destroy existing chart if it exists
        if (this.charts[containerId]) {
            this.charts[containerId].destroy();
        }

        this.charts[containerId] = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Original Text', 'Summary'],
                datasets: [{
                    data: [summaryData.original_length, summaryData.summary_length],
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
                            color: '#e2e8f0',
                            padding: 20
                        }
                    },
                    title: {
                        display: true,
                        text: 'Text Compression',
                        color: '#e2e8f0',
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
