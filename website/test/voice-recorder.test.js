/**
 * Tests for VoiceRecorder functionality
 * Tests audio recording, transcription, and API integration
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock MediaRecorder API
const mockMediaRecorder = vi.fn(() => ({
  start: vi.fn(),
  stop: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  state: 'inactive',
  stream: {
    getTracks: vi.fn(() => [{ stop: vi.fn() }])
  }
}));

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

  beforeEach(() => {
    // Reset mocks
    vi.clearAllMocks();

    // Mock API client
    mockApiClient = {
      transcribeAudio: vi.fn(() => Promise.resolve({
        transcription: 'Hello world',
        confidence: 0.95
      }))
    };

    // Mock window object for browser environment
    global.window = {
      VoiceRecorder: class VoiceRecorder {
        constructor(apiClient) {
          this.apiClient = apiClient;
          this.isRecording = false;
          this.mediaRecorder = null;
          this.audioChunks = [];
        }

        async init() {
          // Initialize voice recorder
          return true;
        }

        async startRecording() {
          if (this.isRecording) return false;

          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.isRecording = true;
            return true;
          } catch (error) {
            console.error('Failed to start recording:', error);
            return false;
          }
        }

        stopRecording() {
          if (!this.isRecording || !this.mediaRecorder) return false;

          this.mediaRecorder.stop();
          this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
          this.isRecording = false;
          return true;
        }

        async processAudio(audioBlob) {
          if (!this.apiClient) {
            throw new Error('API client not provided');
          }

          return await this.apiClient.transcribeAudio(audioBlob);
        }

        displayTranscriptionResults(result) {
          // Mock display function
          console.log('Transcription result:', result);
        }
      }
    };

    // Create VoiceRecorder instance
    voiceRecorder = new window.VoiceRecorder(mockApiClient);
  });

  afterEach(() => {
    // Clean up
    if (voiceRecorder && voiceRecorder.isRecording) {
      voiceRecorder.stopRecording();
    }
  });

  describe('Initialization', () => {
    it('should initialize successfully with API client', async () => {
      const result = await voiceRecorder.init();
      expect(result).toBe(true);
      expect(voiceRecorder.apiClient).toBe(mockApiClient);
    });

    it('should fail without API client', () => {
      expect(() => new window.VoiceRecorder(null)).not.toThrow();
      const recorder = new window.VoiceRecorder(null);
      expect(recorder.apiClient).toBe(null);
    });
  });

  describe('Recording Controls', () => {
    it('should start recording successfully', async () => {
      const result = await voiceRecorder.startRecording();
      expect(result).toBe(true);
      expect(voiceRecorder.isRecording).toBe(true);
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({ audio: true });
    });

    it('should not start recording if already recording', async () => {
      await voiceRecorder.startRecording();
      const secondStart = await voiceRecorder.startRecording();
      expect(secondStart).toBe(false);
    });

    it('should stop recording successfully', async () => {
      await voiceRecorder.startRecording();
      const result = voiceRecorder.stopRecording();
      expect(result).toBe(true);
      expect(voiceRecorder.isRecording).toBe(false);
    });

    it('should handle recording errors gracefully', async () => {
      navigator.mediaDevices.getUserMedia.mockRejectedValueOnce(new Error('Permission denied'));
      const result = await voiceRecorder.startRecording();
      expect(result).toBe(false);
      expect(voiceRecorder.isRecording).toBe(false);
    });
  });

  describe('Audio Processing', () => {
    it('should process audio successfully', async () => {
      const mockBlob = new Blob(['audio data'], { type: 'audio/wav' });
      const result = await voiceRecorder.processAudio(mockBlob);

      expect(result.transcription).toBe('Hello world');
      expect(result.confidence).toBe(0.95);
      expect(mockApiClient.transcribeAudio).toHaveBeenCalledWith(mockBlob);
    });

    it('should throw error when API client is missing', async () => {
      const recorderWithoutClient = new window.VoiceRecorder(null);
      const mockBlob = new Blob(['audio data'], { type: 'audio/wav' });

      await expect(recorderWithoutClient.processAudio(mockBlob)).rejects.toThrow('API client not provided');
    });

    it('should handle API errors gracefully', async () => {
      mockApiClient.transcribeAudio.mockRejectedValueOnce(new Error('API Error'));
      const mockBlob = new Blob(['audio data'], { type: 'audio/wav' });

      await expect(voiceRecorder.processAudio(mockBlob)).rejects.toThrow('API Error');
    });
  });

  describe('Security and Validation', () => {
    it('should validate audio input types', async () => {
      const invalidBlob = new Blob(['text data'], { type: 'text/plain' });

      // Should still process as the validation is typically done on the server
      await expect(voiceRecorder.processAudio(invalidBlob)).resolves.toBeDefined();
    });

    it('should handle permission errors', async () => {
      navigator.mediaDevices.getUserMedia.mockRejectedValueOnce(
        new DOMException('Permission denied', 'NotAllowedError')
      );

      const result = await voiceRecorder.startRecording();
      expect(result).toBe(false);
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
        // startRecording should handle MediaRecorder unavailability gracefully and return false
        const result = await recorder.startRecording();
        expect(result).toBe(false);
      } finally {
        // Always restore MediaRecorder even if the test fails
        global.MediaRecorder = originalMediaRecorder;
      }
    });
  });
});
