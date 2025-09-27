/**
 * Tests for NotificationManager
 * Tests the toast notification system with dependency injection
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('NotificationManager', () => {
  let notificationManager;

  beforeEach(async () => {
    // Clean up DOM
    document.body.innerHTML = '';

    // Reset the global instance
    delete window.NotificationManager;

    // Import fresh instance
    await import('../js/notification-manager.js');
    notificationManager = window.NotificationManager;
  });

  describe('Initialization', () => {
    it('should create a NotificationManager instance', () => {
      expect(notificationManager).toBeInstanceOf(Object);
      expect(notificationManager.activeToasts).toBeInstanceOf(Set);
      expect(notificationManager.maxToasts).toBe(3);
    });
  });

  describe('Toast Creation', () => {
    it('should show a success toast', () => {
      notificationManager.success('Test success message');

      const toast = document.querySelector('.notification-toast');
      expect(toast).toBeTruthy();
      expect(toast.textContent).toContain('Test success message');
      expect(toast.classList.contains('toast-success')).toBe(true);
    });

    it('should show an error toast with custom duration', () => {
      notificationManager.error('Test error message', 3000);

      const toast = document.querySelector('.notification-toast');
      expect(toast).toBeTruthy();
      expect(toast.classList.contains('toast-error')).toBe(true);
    });

    it('should limit concurrent toasts', () => {
      // Mock console.warn
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Add max toasts
      for (let i = 0; i < 3; i++) {
        notificationManager.show(`Message ${i}`);
      }

      // Try to add one more
      notificationManager.show('Overflow message');

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Too many active toasts')
      );

      consoleWarnSpy.mockRestore();
    });
  });

  describe('Toast Removal', () => {
    it('should remove toast after duration', async () => {
      notificationManager.show('Test message', 'info', 100);

      const toast = document.querySelector('.notification-toast');
      expect(toast).toBeTruthy();

      // Wait for auto-removal
      await new Promise(resolve => setTimeout(resolve, 200));

      expect(document.querySelector('.notification-toast')).toBeFalsy();
    });

    it('should remove toast on close button click', () => {
      notificationManager.show('Test message');

      const closeBtn = document.querySelector('.notification-toast span');
      expect(closeBtn).toBeTruthy();

      closeBtn.click();

      expect(document.querySelector('.notification-toast')).toBeFalsy();
    });
  });

  describe('Toast Management', () => {
    it('should clear all active toasts', () => {
      notificationManager.show('Toast 1');
      notificationManager.show('Toast 2');
      notificationManager.show('Toast 3');

      expect(notificationManager.getActiveCount()).toBe(3);

      notificationManager.clearAll();

      expect(notificationManager.getActiveCount()).toBe(0);
      expect(document.querySelectorAll('.notification-toast').length).toBe(0);
    });

    it('should track active toast count', () => {
      expect(notificationManager.getActiveCount()).toBe(0);

      notificationManager.show('Test toast');
      expect(notificationManager.getActiveCount()).toBe(1);

      notificationManager.clearAll();
      expect(notificationManager.getActiveCount()).toBe(0);
    });
  });

  describe('Legacy API Compatibility', () => {
    it('should expose legacy global methods', () => {
      expect(typeof window.showSuccess).toBe('function');
      expect(typeof window.showError).toBe('function');
      expect(typeof window.showInfo).toBe('function');
      expect(typeof window.showWarning).toBe('function');
    });

    it('should call manager methods through legacy API', () => {
      const successSpy = vi.spyOn(notificationManager, 'success');
      const errorSpy = vi.spyOn(notificationManager, 'error');

      window.showSuccess('Legacy success');
      window.showError('Legacy error');

      expect(successSpy).toHaveBeenCalledWith('Legacy success');
      expect(errorSpy).toHaveBeenCalledWith('Legacy error');
    });
  });
});
