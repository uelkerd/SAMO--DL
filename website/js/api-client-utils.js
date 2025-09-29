/**
 * API Client Utilities
 * Reusable utilities for API client initialization and management
 */

class ApiClientManager {
    constructor() {
        this.initializationPromise = null;
        this.eventListeners = new Set();
        this.isInitialized = false;
    }

    /**
     * Wait for API client to be available using polling
     * @param {Object} options - Configuration options
     * @param {number} options.timeoutMs - Timeout in milliseconds
     * @param {number} options.pollInterval - Polling interval in milliseconds
     * @returns {Promise<Object>} The API client instance
     */
    async waitForApiClientPolling(options = {}) {
        const timeoutMs = options.timeoutMs || (window.SAMO_CONFIG?.API?.TIMEOUTS?.API_CLIENT_INIT || 5000);
        const pollInterval = options.pollInterval || 100;
        const maxAttempts = Math.ceil(timeoutMs / pollInterval);

        return new Promise((resolve, reject) => {
            let attempts = 0;
            const checkClient = () => {
                if (window.apiClient) {
                    resolve(window.apiClient);
                } else if (attempts >= maxAttempts) {
                    reject(new Error(`API client not available within ${timeoutMs}ms timeout`));
                } else {
                    attempts++;
                    setTimeout(checkClient, pollInterval);
                }
            };
            checkClient();
        });
    }

    /**
     * Wait for API client using event-based initialization
     * @param {Object} options - Configuration options
     * @param {number} options.timeoutMs - Timeout in milliseconds
     * @returns {Promise<Object>} The API client instance
     */
    async waitForApiClientEvent(options = {}) {
        const timeoutMs = options.timeoutMs || (window.SAMO_CONFIG?.API?.TIMEOUTS?.API_CLIENT_INIT || 5000);

        return new Promise((resolve, reject) => {
            // If already available, resolve immediately
            if (window.apiClient) {
                resolve(window.apiClient);
                return;
            }

            // Set up timeout
            const timeoutId = setTimeout(() => {
                this.removeEventListener('apiClientReady', onApiClientReady);
                reject(new Error(`API client not available within ${timeoutMs}ms timeout`));
            }, timeoutMs);

            // Set up event listener
            const onApiClientReady = (event) => {
                clearTimeout(timeoutId);
                this.removeEventListener('apiClientReady', onApiClientReady);
                resolve(event.detail.apiClient);
            };

            this.addEventListener('apiClientReady', onApiClientReady);
        });
    }

    /**
     * Wait for API client using hybrid approach (event + polling fallback)
     * @param {Object} options - Configuration options
     * @returns {Promise<Object>} The API client instance
     */
    async waitForApiClient(options = {}) {
        const timeoutMs = options.timeoutMs || (window.SAMO_CONFIG?.API?.TIMEOUTS?.API_CLIENT_INIT || 5000);
        const useEventBased = options.useEventBased !== false; // Default to true

        try {
            if (useEventBased) {
                return await this.waitForApiClientEvent(options);
            } else {
                return await this.waitForApiClientPolling(options);
            }
        } catch (error) {
            // If event-based fails and we haven't tried polling, try polling as fallback
            if (useEventBased) {
                console.warn('⚠️ Event-based API client wait failed, trying polling fallback:', error.message);
                return await this.waitForApiClientPolling(options);
            }
            throw error;
        }
    }

    /**
     * Notify that API client is ready
     * @param {Object} apiClient - The API client instance
     */
    notifyApiClientReady(apiClient) {
        this.isInitialized = true;
        const event = new CustomEvent('apiClientReady', {
            detail: { apiClient }
        });
        this.dispatchEvent(event);
    }

    /**
     * Add event listener for API client events
     * @param {string} eventName - Event name
     * @param {Function} callback - Event callback
     */
    addEventListener(eventName, callback) {
        this.eventListeners.add({ eventName, callback });
        window.addEventListener(eventName, callback);
    }

    /**
     * Remove event listener for API client events
     * @param {string} eventName - Event name
     * @param {Function} callback - Event callback
     */
    removeEventListener(eventName, callback) {
        this.eventListeners.delete({ eventName, callback });
        window.removeEventListener(eventName, callback);
    }

    /**
     * Clean up all event listeners
     */
    cleanup() {
        this.eventListeners.forEach(({ eventName, callback }) => {
            window.removeEventListener(eventName, callback);
        });
        this.eventListeners.clear();
    }

    /**
     * Check if API client is available
     * @returns {boolean} True if API client is available
     */
    isApiClientAvailable() {
        return !!window.apiClient;
    }

    /**
     * Get API client with retry logic
     * @param {Object} options - Configuration options
     * @param {number} options.maxRetries - Maximum number of retries
     * @param {number} options.retryDelay - Delay between retries in milliseconds
     * @returns {Promise<Object>} The API client instance
     */
    async getApiClientWithRetry(options = {}) {
        const maxRetries = options.maxRetries || 3;
        const retryDelay = options.retryDelay || 1000;

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                return await this.waitForApiClient(options);
            } catch (error) {
                if (attempt === maxRetries) {
                    throw new Error(`Failed to get API client after ${maxRetries} attempts: ${error.message}`);
                }
                
                console.warn(`⚠️ API client initialization attempt ${attempt} failed, retrying in ${retryDelay}ms...`, error.message);
                await new Promise(resolve => setTimeout(resolve, retryDelay));
            }
        }
    }
}

// Create global instance
window.ApiClientManager = new ApiClientManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ApiClientManager;
}
