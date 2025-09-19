/**
 * LayoutManager Comprehensive Test Suite
 * Tests for UI state management, processing guards, and layout transitions
 */

// Import the LayoutManager from our test module
const LayoutManager = require('../modules/LayoutManager');

/**
 * Setup test environment and mocks
 */
beforeAll(() => {
  // Mock clearAllResultContent function
  global.clearAllResultContent = jest.fn();

  // Mock console methods to reduce test noise
  console.log = jest.fn();
  console.warn = jest.fn();
  console.error = jest.fn();

  // Mock Date.now for consistent timing tests
  const mockNow = jest.spyOn(Date, 'now');
  let currentTime = 1000;
  mockNow.mockImplementation(() => {
    const time = currentTime;
    currentTime += 1000; // Add 1 second for each call
    return time;
  });
});

describe('LayoutManager', () => {
  let mockElements;

  beforeEach(() => {
    // Reset LayoutManager state
    LayoutManager.currentState = 'initial';
    LayoutManager.isProcessing = false;
    LayoutManager.activeRequests.clear();
    LayoutManager.processingStartTime = null;

    // Reset all mocks
    jest.clearAllMocks();

    // Create mock DOM environment
    mockElements = createMockDOMEnvironment();
  });

  describe('State Management', () => {
    test('should initialize with correct default state', () => {
      expect(LayoutManager.currentState).toBe('initial');
      expect(LayoutManager.isProcessing).toBe(false);
      expect(LayoutManager.activeRequests.size).toBe(0);
      expect(LayoutManager.processingStartTime).toBe(null);
      expect(LayoutManager.maxProcessingTime).toBe(120000);
    });

    test('should reset processing state completely', () => {
      // Set some state to reset
      LayoutManager.currentState = 'processing';
      LayoutManager.isProcessing = true;
      LayoutManager.activeRequests.add({ controller: 'fake' });

      LayoutManager.resetProcessingState();

      expect(LayoutManager.currentState).toBe('initial');
      expect(LayoutManager.isProcessing).toBe(false);
      expect(LayoutManager.activeRequests.size).toBe(0);
      expect(console.log).toHaveBeenCalledWith('🔄 Safety reset: clearing processing state...');
    });

    test('should perform emergency reset when stuck', () => {
      // Set stuck state
      LayoutManager.currentState = 'processing';
      LayoutManager.isProcessing = true;
      LayoutManager.activeRequests.add({ controller: 'fake' });

      LayoutManager.emergencyReset();

      expect(LayoutManager.currentState).toBe('initial');
      expect(LayoutManager.isProcessing).toBe(false);
      expect(LayoutManager.activeRequests.size).toBe(0);
      expect(clearAllResultContent).toHaveBeenCalled();
      expect(console.warn).toHaveBeenCalledWith('🚨 Emergency reset: processing state appears stuck, forcing reset...');
    });
  });

  describe('Processing Guard Logic', () => {
    test('should allow processing when not already processing', () => {
      expect(LayoutManager.canStartProcessing()).toBe(true);
    });

    test('should prevent processing when already processing', () => {
      LayoutManager.isProcessing = true;

      expect(LayoutManager.canStartProcessing()).toBe(false);
    });

    test('should start processing successfully when allowed', () => {
      const result = LayoutManager.startProcessing();

      expect(result).toBe(true);
      expect(LayoutManager.isProcessing).toBe(true);
      expect(LayoutManager.processingStartTime).toBeGreaterThan(0);
      expect(LayoutManager.activeRequests.size).toBe(0);
      expect(console.log).toHaveBeenCalledWith('🚀 Processing started - locked for concurrent operations');
    });

    test('should prevent starting processing when already processing (within time limit)', () => {
      // Start processing first
      LayoutManager.startProcessing();
      const firstStartTime = LayoutManager.processingStartTime;

      // Try to start again
      const result = LayoutManager.startProcessing();

      expect(result).toBe(false);
      expect(LayoutManager.processingStartTime).toBe(firstStartTime); // Should not change
      expect(console.warn).toHaveBeenCalledWith('⚠️ Processing already in progress, ignoring request');
    });

    test('should force reset when processing is stuck too long', () => {
      // Mock Date.now to simulate time passage
      const originalNow = Date.now;
      let currentTime = 1000;
      Date.now = jest.fn(() => currentTime);

      // Start processing
      LayoutManager.startProcessing();
      expect(LayoutManager.isProcessing).toBe(true);

      // Simulate time passage beyond max processing time
      currentTime = 1000 + LayoutManager.maxProcessingTime + 1000; // 121 seconds later

      // Try to start processing again (should force reset)
      const result = LayoutManager.startProcessing();

      expect(result).toBe(true); // Should succeed after force reset
      expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('Processing stuck for'));

      // Restore original Date.now
      Date.now = originalNow;
    });

    test('should end processing correctly', () => {
      // Start processing first
      LayoutManager.startProcessing();
      LayoutManager.activeRequests.add({ controller: 'fake' });

      LayoutManager.endProcessing();

      expect(LayoutManager.isProcessing).toBe(false);
      expect(LayoutManager.processingStartTime).toBe(null);
      expect(LayoutManager.activeRequests.size).toBe(0);
      expect(console.log).toHaveBeenCalledWith('✅ Processing completed - ready for new operations');
    });
  });

  describe('Request Tracking', () => {
    test('should add active request controller', () => {
      const mockController = { abort: jest.fn(), id: 'test-controller' };

      LayoutManager.addActiveRequest(mockController);

      expect(LayoutManager.activeRequests.has(mockController)).toBe(true);
      expect(console.log).toHaveBeenCalledWith('📡 Added request to tracking (1 active)');
    });

    test('should handle null controller gracefully', () => {
      LayoutManager.addActiveRequest(null);

      expect(LayoutManager.activeRequests.size).toBe(0);
    });

    test('should remove active request controller', () => {
      const mockController = { abort: jest.fn(), id: 'test-controller' };
      LayoutManager.activeRequests.add(mockController);

      LayoutManager.removeActiveRequest(mockController);

      expect(LayoutManager.activeRequests.has(mockController)).toBe(false);
      expect(console.log).toHaveBeenCalledWith('📡 Removed request from tracking (0 remaining)');
    });

    test('should handle removing non-existent controller', () => {
      const mockController = { abort: jest.fn(), id: 'test-controller' };

      LayoutManager.removeActiveRequest(mockController);

      // Should not log removal message since controller wasn't in the set
      expect(console.log).not.toHaveBeenCalledWith(expect.stringContaining('Removed request from tracking'));
    });

    test('should cancel all active requests', () => {
      const mockController1 = { abort: jest.fn(), id: 'controller1' };
      const mockController2 = { abort: jest.fn(), id: 'controller2' };
      const mockController3 = { id: 'controller3' }; // No abort method

      LayoutManager.activeRequests.add(mockController1);
      LayoutManager.activeRequests.add(mockController2);
      LayoutManager.activeRequests.add(mockController3);

      LayoutManager.cancelActiveRequests();

      expect(mockController1.abort).toHaveBeenCalled();
      expect(mockController2.abort).toHaveBeenCalled();
      expect(LayoutManager.activeRequests.size).toBe(0);
      expect(console.log).toHaveBeenCalledWith('🚫 Cancelling 3 active requests...');
    });

    test('should force reset processing and cancel requests', () => {
      const mockController = { abort: jest.fn(), id: 'test-controller' };
      LayoutManager.isProcessing = true;
      LayoutManager.processingStartTime = 1000;
      LayoutManager.currentState = 'processing';
      LayoutManager.activeRequests.add(mockController);

      LayoutManager.forceResetProcessing();

      expect(mockController.abort).toHaveBeenCalled();
      expect(LayoutManager.isProcessing).toBe(false);
      expect(LayoutManager.processingStartTime).toBe(null);
      expect(LayoutManager.currentState).toBe('initial');
      expect(LayoutManager.activeRequests.size).toBe(0);
      expect(console.warn).toHaveBeenCalledWith('🚨 Force resetting processing state and cancelling all requests...');
      expect(console.log).toHaveBeenCalledWith('✅ Processing force reset completed');
    });
  });

  describe('Layout Transitions', () => {
    test('should show processing state successfully', () => {
      const result = LayoutManager.showProcessingState();

      expect(result).toBe(true);
      expect(LayoutManager.currentState).toBe('processing');
      expect(LayoutManager.isProcessing).toBe(true);

      // Check DOM manipulations
      expect(mockElements.inputLayout.style.display).toBe('none');
      expect(mockElements.resultsLayout.classList.remove).toHaveBeenCalledWith('d-none');
      expect(mockElements.resultsLayout.style.display).toBe('block');

      expect(console.log).toHaveBeenCalledWith('📺 Transitioning to processing state...');
      expect(console.log).toHaveBeenCalledWith('✅ Processing state transition complete');
    });

    test('should fail to show processing state when already processing', () => {
      // Start processing first
      LayoutManager.startProcessing();

      const result = LayoutManager.showProcessingState();

      expect(result).toBe(false);
      expect(console.error).toHaveBeenCalledWith('❌ Cannot start processing - already in progress');
    });

    test('should show results and hide loading', () => {
      // Setup processing state first
      LayoutManager.currentState = 'processing';

      LayoutManager.showResults();

      expect(LayoutManager.currentState).toBe('results');

      // Check DOM manipulations
      expect(mockElements.emotionResults.classList.remove).toHaveBeenCalledWith('result-section-hidden');
      expect(mockElements.emotionResults.classList.add).toHaveBeenCalledWith('result-section-visible');
      expect(mockElements.summarizationResults.classList.remove).toHaveBeenCalledWith('result-section-hidden');
      expect(mockElements.summarizationResults.classList.add).toHaveBeenCalledWith('result-section-visible');

      expect(console.log).toHaveBeenCalledWith('📊 Showing results...');
      expect(console.log).toHaveBeenCalledWith('✅ Results display complete');
    });

    test('should reset to initial state correctly', () => {
      // Setup some state to reset
      LayoutManager.currentState = 'results';
      LayoutManager.isProcessing = true;
      const mockController = { abort: jest.fn() };
      LayoutManager.activeRequests.add(mockController);

      LayoutManager.resetToInitialState();

      expect(LayoutManager.currentState).toBe('initial');
      expect(LayoutManager.isProcessing).toBe(false);
      expect(LayoutManager.activeRequests.size).toBe(0);

      // Check DOM manipulations
      expect(mockElements.inputLayout.style.display).toBe('block');
      expect(mockElements.inputLayout.classList.remove).toHaveBeenCalledWith('d-none');
      expect(mockElements.resultsLayout.classList.add).toHaveBeenCalledWith('d-none');
      expect(mockElements.resultsLayout.style.display).toBe('none');
      expect(mockElements.textInput.value).toBe('');

      // Check result sections are hidden
      expect(mockElements.emotionResults.classList.add).toHaveBeenCalledWith('result-section-hidden');
      expect(mockElements.emotionResults.classList.remove).toHaveBeenCalledWith('result-section-visible');
      expect(mockElements.summarizationResults.classList.add).toHaveBeenCalledWith('result-section-hidden');
      expect(mockElements.summarizationResults.classList.remove).toHaveBeenCalledWith('result-section-visible');

      expect(console.log).toHaveBeenCalledWith('🔄 Resetting to initial state...');
      expect(console.log).toHaveBeenCalledWith('✅ Reset to initial state complete');
    });

    test('should handle missing DOM elements gracefully', () => {
      // Clear document body to simulate missing elements
      document.body.innerHTML = '';

      // These should not throw errors
      expect(() => LayoutManager.showProcessingState()).not.toThrow();
      expect(() => LayoutManager.showResults()).not.toThrow();
      expect(() => LayoutManager.resetToInitialState()).not.toThrow();
    });
  });

  describe('Debug Section Toggle', () => {
    test('should toggle debug section visibility', () => {
      const debugSection = createMockElement('div', {
        id: 'debugTestSection',
        class: 'd-none'
      });
      const toggleButton = createMockElement('button', {
        id: 'debugToggleBtn'
      });

      document.body.appendChild(debugSection);
      document.body.appendChild(toggleButton);

      LayoutManager.toggleDebugSection();

      expect(debugSection.classList.toggle).toHaveBeenCalledWith('d-none', false);

      // Mock the button text change
      expect(toggleButton.textContent).toBe('Hide Debug');
    });

    test('should handle missing debug elements gracefully', () => {
      expect(() => LayoutManager.toggleDebugSection()).not.toThrow();
    });
  });

  describe('Edge Cases and Error Handling', () => {
    test('should handle concurrent processing attempts gracefully', () => {
      // Start processing
      const result1 = LayoutManager.startProcessing();
      expect(result1).toBe(true);

      // Try to start again multiple times
      const result2 = LayoutManager.startProcessing();
      const result3 = LayoutManager.startProcessing();
      const result4 = LayoutManager.startProcessing();

      expect(result2).toBe(false);
      expect(result3).toBe(false);
      expect(result4).toBe(false);

      // Should still be in processing state
      expect(LayoutManager.isProcessing).toBe(true);
    });

    test('should handle request tracking with various controller types', () => {
      const validController = { abort: jest.fn() };
      const invalidController = {}; // No abort method
      const nullController = null;
      const undefinedController = undefined;

      LayoutManager.addActiveRequest(validController);
      LayoutManager.addActiveRequest(invalidController);
      LayoutManager.addActiveRequest(nullController);
      LayoutManager.addActiveRequest(undefinedController);

      expect(LayoutManager.activeRequests.size).toBe(2); // Only valid ones added

      LayoutManager.cancelActiveRequests();

      expect(validController.abort).toHaveBeenCalled();
      expect(LayoutManager.activeRequests.size).toBe(0);
    });

    test('should maintain state consistency during rapid state changes', () => {
      // Test proper state transitions
      expect(LayoutManager.currentState).toBe('initial');

      LayoutManager.showProcessingState();
      expect(LayoutManager.currentState).toBe('processing');

      LayoutManager.showResults();
      expect(LayoutManager.currentState).toBe('results');

      LayoutManager.resetToInitialState();
      expect(LayoutManager.currentState).toBe('initial');
      expect(LayoutManager.isProcessing).toBe(false);
    });

    test('should handle processing timeout calculation correctly', () => {
      const originalNow = Date.now;
      let currentTime = 1000;
      Date.now = jest.fn(() => currentTime);

      LayoutManager.startProcessing();
      const startTime = LayoutManager.processingStartTime;

      // Simulate time passage
      currentTime = 1000 + 60000; // 60 seconds later

      // This should not trigger force reset (within limit)
      LayoutManager.startProcessing();
      expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('Time elapsed: 60s'));

      // Simulate longer time passage
      currentTime = 1000 + 121000; // 121 seconds later

      // This should trigger force reset
      LayoutManager.startProcessing();
      expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('Processing stuck for 121s'));

      Date.now = originalNow;
    });
  });
});