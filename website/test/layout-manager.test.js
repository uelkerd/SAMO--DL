/**
 * Tests for LayoutManager
 * Tests the processing state management and null safety fixes
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('LayoutManager', () => {
  let layoutManager;

  beforeEach(async () => {
    // Import the LayoutManager object
    await import('../js/layout-manager.js');
    layoutManager = window.LayoutManager;

    // Reset to clean state for each test
    layoutManager.isProcessing = false;
    layoutManager.processingStartTime = null;
    layoutManager.activeRequests.clear();
    layoutManager.currentState = 'initial';
  });

  describe('Processing State Safety', () => {
    it('should handle null processingStartTime gracefully', () => {
      // Simulate corrupted state where isProcessing is true but processingStartTime is null
      layoutManager.isProcessing = true;
      layoutManager.processingStartTime = null;

      // Mock forceResetProcessing to track if it's called
      const forceResetSpy = vi.spyOn(layoutManager, 'forceResetProcessing');

      const result = layoutManager.startProcessing();

      expect(forceResetSpy).toHaveBeenCalled();
      expect(result).toBe(false);

      forceResetSpy.mockRestore();
    });

    it('should not compute NaN when processingStartTime is null', () => {
      layoutManager.isProcessing = true;
      layoutManager.processingStartTime = null;

      // Mock console.warn to capture warnings
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const result = layoutManager.startProcessing();

      expect(consoleWarnSpy).toHaveBeenCalledWith('⚠️ Missing processingStartTime; forcing reset...');
      expect(result).toBe(false);

      consoleWarnSpy.mockRestore();
    });

    it('should work normally when processingStartTime is valid', () => {
      layoutManager.isProcessing = true;
      layoutManager.processingStartTime = Date.now() - 1000; // 1 second ago

      // Mock forceResetProcessing to ensure it's not called
      const forceResetSpy = vi.spyOn(layoutManager, 'forceResetProcessing');

      const result = layoutManager.startProcessing();

      // Should not force reset since time hasn't exceeded maxProcessingTime
      expect(forceResetSpy).not.toHaveBeenCalled();
      expect(result).toBe(false); // Still false because processing is already in progress

      forceResetSpy.mockRestore();
    });
  });

  describe('Processing Timeout Detection', () => {
    it('should detect and reset stuck processing', () => {
      layoutManager.isProcessing = true;
      layoutManager.processingStartTime = Date.now() - (layoutManager.maxProcessingTime + 1000); // Exceeded timeout

      const forceResetSpy = vi.spyOn(layoutManager, 'forceResetProcessing');
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const result = layoutManager.startProcessing();

      expect(forceResetSpy).toHaveBeenCalled();
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Processing stuck for')
      );
      // After force reset, processing state is cleared, so startProcessing should succeed
      expect(result).toBe(true);
      expect(layoutManager.isProcessing).toBe(true); // Should be set to true after successful start

      forceResetSpy.mockRestore();
      consoleWarnSpy.mockRestore();
    });
  });

  describe('Normal Processing Flow', () => {
    it('should start processing when not already processing', () => {
      layoutManager.isProcessing = false;

      const result = layoutManager.startProcessing();

      expect(result).toBe(true);
      expect(layoutManager.isProcessing).toBe(true);
      expect(typeof layoutManager.processingStartTime).toBe('number');
      expect(layoutManager.processingStartTime).toBeGreaterThan(0);
      expect(layoutManager.activeRequests.size).toBe(0);
    });

    it('should prevent concurrent processing', () => {
      // Start first processing
      layoutManager.startProcessing();
      expect(layoutManager.isProcessing).toBe(true);

      // Try to start second processing
      const result = layoutManager.startProcessing();
      expect(result).toBe(false);
      expect(layoutManager.isProcessing).toBe(true);
    });

    it('should end processing correctly', () => {
      layoutManager.startProcessing();
      expect(layoutManager.isProcessing).toBe(true);

      layoutManager.endProcessing();
      expect(layoutManager.isProcessing).toBe(false);
      expect(layoutManager.processingStartTime).toBe(null);
      expect(layoutManager.activeRequests.size).toBe(0);
    });
  });

  describe('Reset Functionality', () => {
    it('should force reset processing state', () => {
      // Set up some state
      layoutManager.isProcessing = true;
      layoutManager.processingStartTime = Date.now();
      layoutManager.activeRequests.add('test-request');
      layoutManager.currentState = 'processing';

      layoutManager.forceResetProcessing();

      expect(layoutManager.isProcessing).toBe(false);
      expect(layoutManager.processingStartTime).toBe(null);
      expect(layoutManager.activeRequests.size).toBe(0);
      expect(layoutManager.currentState).toBe('initial');
    });
  });
});
