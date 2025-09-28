/**
 * Notification Manager
 * Handles toast notifications and user feedback messages
 * Provides a clean separation of UI concerns from business logic
 */

class NotificationManager {
    constructor() {
        this.activeToasts = new Set();
        this.maxToasts = 3; // Limit concurrent toasts

        // Create or reuse a container for proper toast stacking
        this.ensureContainer();
    }

    /**
     * Ensure container exists, creating it if necessary
     * @private
     */
    ensureContainer() {
        this.container = document.getElementById('toastContainer') || (() => {
            const c = document.createElement('div');
            c.id = 'toastContainer';
            c.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                z-index: 10000;
                pointer-events: none;
            `;
            document.body.appendChild(c);
            return c;
        })();
    }

    /**
     * Show a toast notification
     * @param {string} message - The message to display
     * @param {string} type - The notification type ('success', 'error', 'info', 'warning')
     * @param {number} duration - Duration in milliseconds (default: 3000)
     */
    show(message, type = 'info', duration = 3000) {
        // Limit concurrent toasts
        if (this.activeToasts.size >= this.maxToasts) {
            console.warn('⚠️ Too many active toasts, ignoring new notification');
            return;
        }

        // Create toast element
        const toast = this.createToast(message, type);
        this.container.appendChild(toast);
        this.activeToasts.add(toast);

        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateY(0)';
        }, 10);

        // Auto-remove after duration
        setTimeout(() => {
            this.removeToast(toast);
        }, duration);
    }

    /**
     * Show success notification
     * @param {string} message - The success message
     * @param {number} duration - Duration in milliseconds
     */
    success(message, duration = 3000) {
        this.show(message, 'success', duration);
    }

    /**
     * Show error notification
     * @param {string} message - The error message
     * @param {number} duration - Duration in milliseconds
     */
    error(message, duration = 5000) {
        this.show(message, 'error', duration);
    }

    /**
     * Show info notification
     * @param {string} message - The info message
     * @param {number} duration - Duration in milliseconds
     */
    info(message, duration = 3000) {
        this.show(message, 'info', duration);
    }

    /**
     * Show warning notification
     * @param {string} message - The warning message
     * @param {number} duration - Duration in milliseconds
     */
    warning(message, duration = 4000) {
        this.show(message, 'warning', duration);
    }

    /**
     * Create a toast element
     * @private
     */
    createToast(message, type) {
        const toast = document.createElement('div');
        toast.className = `notification-toast toast-${type}`;

        // Add accessibility attributes
        toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
        toast.setAttribute('aria-live', type === 'error' ? 'assertive' : 'polite');

        // Set initial animation state
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-20px)';

        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'toast-close-btn';
        closeBtn.setAttribute('aria-label', 'Close notification');
        closeBtn.textContent = '×';
        closeBtn.addEventListener('click', () => this.removeToast(toast));

        // Add message text
        const textNode = document.createTextNode(message);

        toast.appendChild(closeBtn);
        toast.appendChild(textNode);

        return toast;
    }

    /**
     * Remove a toast with animation
     * @private
     */
    removeToast(toast) {
        if (!toast || !this.activeToasts.has(toast)) {
            return;
        }

        // Animate out
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-20px)';

        // Remove from DOM after animation
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
            this.activeToasts.delete(toast);
        }, 300);
    }

    /**
     * Clear all active toasts
     */
    clearAll() {
        const toasts = Array.from(this.activeToasts);
        toasts.forEach(toast => this.removeToast(toast));
    }

    /**
     * Get number of active toasts
     */
    getActiveCount() {
        return this.activeToasts.size;
    }
}

// Create global instance
window.NotificationManager = new NotificationManager();

// Legacy compatibility - expose common methods globally
window.showSuccess = (message) => window.NotificationManager.success(message);
window.showError = (message) => window.NotificationManager.error(message);
window.showInfo = (message) => window.NotificationManager.info(message);
window.showWarning = (message) => window.NotificationManager.warning(message);

console.log('🔔 Notification Manager loaded successfully');
