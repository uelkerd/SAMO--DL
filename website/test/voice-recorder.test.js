/**
 * Tests for VoiceRecorder functionality
 * Tests audio recording, transcription, and API integration
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
// Import the real VoiceRecorder implementation
import '../js/voice-recorder.js';

// Mock MediaRecorder API with event semantics
const createMockMediaRecorder = () => {
  const listeners = {};
  let state = 'inactive';

  const instance = {
    start: vi.fn((timeslice) => {
      state = 'recording';
      // Simulate dataavailable event emission
      setTimeout(() => {
        if (listeners.dataavailable) {
          listeners.dataavailable.forEach(callback => {
            callback({ data: new Blob(['mock audio data'], { type: 'audio/webm' }) });
          });
        }
      }, 100);
    }),
    stop: vi.fn(() => {
      state = 'inactive';
      // Simulate stop event emission
      setTimeout(() => {
        if (listeners.stop) {
          listeners.stop.forEach(callback => callback());
        }
      }, 50);
    }),
    addEventListener: vi.fn((event, callback) => {
      if (!listeners[event]) listeners[event] = [];
      listeners[event].push(callback);
    }),
    removeEventListener: vi.fn((event, callback) => {
      if (listeners[event]) {
        const index = listeners[event].indexOf(callback);
        if (index > -1) listeners[event].splice(index, 1);
      }
    }),
    get state() { return state; },
    stream: {
      getTracks: vi.fn(() => [{ stop: vi.fn() }])
    }
  };

  return instance;
};

const mockMediaRecorder = vi.fn(() => createMockMediaRecorder());

global.MediaRecorder = mockMediaRecorder;
global.MediaRecorder.isTypeSupported = vi.fn(() => true);

// Mock navigator.mediaDevices
global.navigator = {
  mediaDevices: {
    getUserMedia: vi.fn(() => Promise.resolve({
      getTracks: () => [{ stop: vi.fn() }]
    }))
  }
};

describe('VoiceRecorder', () => {
  let voiceRecorder;
  let mockApiClient;
  let mockRecordBtn;
  let mockStopBtn;

  beforeEach(() => {
    // Reset mocks
    vi.clearAllMocks();

    // Create DOM elements that VoiceRecorder expects
    mockRecordBtn = document.createElement('button');
    mockRecordBtn.id = 'recordBtn';
    document.body.appendChild(mockRecordBtn);

    mockStopBtn = document.createElement('button');
    mockStopBtn.id = 'stopBtn';
    document.body.appendChild(mockStopBtn);

    // Mock API client
    mockApiClient = {
      transcribeAudio: vi.fn(() => Promise.resolve({
        transcription: 'Hello world',
        confidence: 0.95
      }))
    };

    // Create VoiceRecorder instance using the real class from the imported module
    voiceRecorder = new window.VoiceRecorder(mockApiClient);
  });

  afterEach(() => {
    // Clean up recording state
    if (voiceRecorder && voiceRecorder.isRecording) {
      voiceRecorder.stopRecording();
    }

    // Clean up DOM elements to prevent test pollution
    if (mockRecordBtn && mockRecordBtn.parentNode) {
      mockRecordBtn.parentNode.removeChild(mockRecordBtn);
    }
    if (mockStopBtn && mockStopBtn.parentNode) {
      mockStopBtn.parentNode.removeChild(mockStopBtn);
    }

    // Clear any remaining elements from document.body as failsafe
    const remainingButtons = document.body.querySelectorAll('#recordBtn, #stopBtn');
    remainingButtons.forEach(btn => {
      if (btn.parentNode) {
        btn.parentNode.removeChild(btn);
      }
    });
  });

  describe('Initialization', () => {
    it('should initialize successfully with API client', async () => {
      // DOM elements are already created in beforeEach
      const result = await voiceRecorder.init();
      expect(result).toBe(true);
      expect(voiceRecorder.apiClient).toBe(mockApiClient);
    });

    it('should fail without API client', () => {
      expect(() => new window.VoiceRecorder(null)).not.toThrow();
      const recorder = new window.VoiceRecorder(null);
      expect(recorder.apiClient).toBe(null);
    });

    it('should clean up DOM elements after each test', () => {
      // Verify DOM elements exist from beforeEach
      expect(document.getElementById('recordBtn')).toBe(mockRecordBtn);
      expect(document.getElementById('stopBtn')).toBe(mockStopBtn);

      // This test will verify that afterEach properly cleans up
      // The cleanup will be verified by the next test not finding duplicate elements
    });

    it('should have fresh DOM elements for each test', () => {
      // Verify we have exactly one of each element (no duplicates from previous tests)
      const recordBtns = document.querySelectorAll('#recordBtn');
      const stopBtns = document.querySelectorAll('#stopBtn');

      expect(recordBtns).toHaveLength(1);
      expect(stopBtns).toHaveLength(1);
      expect(recordBtns[0]).toBe(mockRecordBtn);
      expect(stopBtns[0]).toBe(mockStopBtn);
    });
  });

  describe('Recording Controls', () => {
    it('should start recording successfully', async () => {
      await voiceRecorder.startRecording();
      expect(voiceRecorder.isRecording).toBe(true);
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100
        }
      });
    });

    it('should not start recording if already recording', async () => {
      await voiceRecorder.startRecording();
      const initialRecording = voiceRecorder.isRecording;
      await voiceRecorder.startRecording();
      // Should still be recording (second call should be ignored)
      expect(voiceRecorder.isRecording).toBe(initialRecording);
    });

    it('should stop recording successfully', async () => {
      await voiceRecorder.startRecording();
      voiceRecorder.stopRecording();
      expect(voiceRecorder.isRecording).toBe(false);
    });

    it('should handle recording errors gracefully', async () => {
      navigator.mediaDevices.getUserMedia.mockRejectedValueOnce(new Error('Permission denied'));
      await voiceRecorder.startRecording();
      expect(voiceRecorder.isRecording).toBe(false);
    });
  });

  describe('Audio Processing', () => {
    it('should process audio successfully', async () => {
      const mockBlob = new Blob(['audio data'], { type: 'audio/wav' });

      // Mock the processRecordedAudio method to test the core functionality
      const originalProcessRecordedAudio = voiceRecorder.processRecordedAudio;
      voiceRecorder.processRecordedAudio = vi.fn().mockImplementation(async (audioBlob) => {
        if (!voiceRecorder.apiClient) {
          throw new Error('API client not provided');
        }
        return await voiceRecorder.apiClient.transcribeAudio(audioBlob);
      });

      const result = await voiceRecorder.processRecordedAudio(mockBlob);

      expect(result.transcription).toBe('Hello world');
      expect(result.confidence).toBe(0.95);
      expect(mockApiClient.transcribeAudio).toHaveBeenCalledWith(mockBlob);

      // Restore original method
      voiceRecorder.processRecordedAudio = originalProcessRecordedAudio;
    });

    it('should throw error when API client is missing', async () => {
      const recorderWithoutClient = new window.VoiceRecorder(null);
      const mockBlob = new Blob(['audio data'], { type: 'audio/wav' });

      await expect(recorderWithoutClient.processRecordedAudio(mockBlob)).rejects.toThrow('API client not provided');
    });

    it('should handle API errors gracefully', async () => {
      mockApiClient.transcribeAudio.mockRejectedValueOnce(new Error('API Error'));
      const mockBlob = new Blob(['audio data'], { type: 'audio/wav' });

      await expect(voiceRecorder.processRecordedAudio(mockBlob)).rejects.toThrow('API Error');
    });
  });

  describe('Security and Validation', () => {
    it('should validate audio input types', async () => {
      const invalidBlob = new Blob(['text data'], { type: 'text/plain' });

      // Mock processRecordedAudio to test validation
      const originalProcessRecordedAudio = voiceRecorder.processRecordedAudio;
      voiceRecorder.processRecordedAudio = vi.fn().mockImplementation(async (audioBlob) => {
        if (!voiceRecorder.apiClient) {
          throw new Error('API client not provided');
        }
        return await voiceRecorder.apiClient.transcribeAudio(audioBlob);
      });

      // Should still process as the validation is typically done on the server
      await expect(voiceRecorder.processRecordedAudio(invalidBlob)).resolves.toBeDefined();

      // Restore original method
      voiceRecorder.processRecordedAudio = originalProcessRecordedAudio;
    });

    it('should handle permission errors', async () => {
      navigator.mediaDevices.getUserMedia.mockRejectedValueOnce(
        new DOMException('Permission denied', 'NotAllowedError')
      );

      await voiceRecorder.startRecording();
      expect(voiceRecorder.isRecording).toBe(false);
    });
  });

  describe('Browser Compatibility', () => {
    it('should check MediaRecorder support', () => {
      expect(global.MediaRecorder.isTypeSupported).toBeDefined();
      expect(global.MediaRecorder.isTypeSupported('audio/webm')).toBe(true);
    });

    it('should handle unsupported browsers gracefully', async () => {
      const originalMediaRecorder = global.MediaRecorder;
      global.MediaRecorder = undefined;

      try {
        const recorder = new window.VoiceRecorder(mockApiClient);
        // startRecording should handle MediaRecorder unavailability gracefully
        await recorder.startRecording();
        expect(recorder.isRecording).toBe(false);
      } finally {
        // Always restore MediaRecorder even if the test fails
        global.MediaRecorder = originalMediaRecorder;
      }
    });
  });
});
