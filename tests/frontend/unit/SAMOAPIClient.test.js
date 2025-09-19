/**
 * SAMOAPIClient Comprehensive Test Suite
 * Tests for API client functionality, error handling, retries, and edge cases
 */

// Import the SAMOAPIClient class from our test module
const SAMOAPIClient = require('../modules/SAMOAPIClient');

/**
 * Setup test environment and mocks
 */
beforeAll(() => {
  // Mock window.SAMO_CONFIG with complete structure
  global.window = {
    SAMO_CONFIG: {
      API: {
        BASE_URL: 'https://test-api.com',
        ENDPOINTS: {
          EMOTION: '/analyze/emotion',
          SUMMARIZE: '/analyze/summarize',
          VOICE_JOURNAL: '/analyze/voice-journal',
          HEALTH: '/health'
        },
        TIMEOUT: 15000,
        COLD_START_TIMEOUT: 45000,
        RETRY_ATTEMPTS: 3,
        API_KEY: null,
        API_KEY_ENV: null
      }
    }
  };

  // Mock LayoutManager
  global.LayoutManager = {
    addActiveRequest: jest.fn(),
    removeActiveRequest: jest.fn()
  };

  // Mock addToProgressConsole
  global.addToProgressConsole = jest.fn();

  // Mock Date.now for processing time tests
  const mockNow = jest.spyOn(Date, 'now');
  let currentTime = 1000;
  mockNow.mockImplementation(() => {
    const time = currentTime;
    currentTime += 100; // Add 100ms for each call
    return time;
  });
});

describe('SAMOAPIClient', () => {
  let apiClient;

  beforeEach(() => {
    // Create fresh instance for each test
    apiClient = new SAMOAPIClient();

    // Reset all mocks
    jest.clearAllMocks();

    // Reset fetch mock
    global.fetch = jest.fn();

    // Reset localStorage mock (it should already be mocked from setup.js)
    if (localStorage.getItem && localStorage.getItem.mockReturnValue) {
      localStorage.getItem.mockReturnValue(null);
    }
  });

  describe('Constructor and Configuration', () => {
    test('should initialize with default configuration', () => {
      expect(apiClient.baseURL).toBe('https://test-api.com');
      expect(apiClient.timeout).toBe(15000);
      expect(apiClient.coldStartTimeout).toBe(45000);
      expect(apiClient.retryAttempts).toBe(3);
      expect(apiClient.isColdStart).toBe(true);
    });

    test('should handle missing SAMO_CONFIG gracefully', () => {
      // Temporarily remove config
      const originalConfig = global.window.SAMO_CONFIG;
      global.window.SAMO_CONFIG = null;

      const client = new SAMOAPIClient();

      expect(client.baseURL).toBe('https://samo-unified-api-optimized-frrnetyhfa-uc.a.run.app');
      expect(client.timeout).toBe(15000);

      // Restore config
      global.window.SAMO_CONFIG = originalConfig;
    });

    test('should use fallback endpoints when missing', () => {
      const originalConfig = global.window.SAMO_CONFIG;
      global.window.SAMO_CONFIG = {
        API: {
          BASE_URL: 'https://test-api.com',
          ENDPOINTS: {}
        }
      };

      const client = new SAMOAPIClient();

      expect(client.endpoints.VOICE_JOURNAL).toBe('/analyze/voice-journal');

      global.window.SAMO_CONFIG = originalConfig;
    });
  });

  describe('API Key Management', () => {
    test('should get API key from SAMO_CONFIG', () => {
      global.window.SAMO_CONFIG.API.API_KEY = 'config-key-123';

      const key = apiClient.getApiKey();

      expect(key).toBe('config-key-123');

      // Cleanup
      delete global.window.SAMO_CONFIG.API.API_KEY;
    });

    test('should get API key from localStorage', () => {
      if (localStorage.getItem && localStorage.getItem.mockReturnValue) {
        localStorage.getItem.mockReturnValue('stored-key-456');

        const key = apiClient.getApiKey();

        expect(key).toBe('stored-key-456');
        expect(localStorage.getItem).toHaveBeenCalledWith('samo_api_key');
      }
    });

    test('should trim API key from localStorage', () => {
      if (localStorage.getItem && localStorage.getItem.mockReturnValue) {
        localStorage.getItem.mockReturnValue('  key-with-spaces  ');

        const key = apiClient.getApiKey();

        expect(key).toBe('key-with-spaces');
      }
    });

    test('should return null for empty localStorage key', () => {
      if (localStorage.getItem && localStorage.getItem.mockReturnValue) {
        localStorage.getItem.mockReturnValue('   ');

        const key = apiClient.getApiKey();

        expect(key).toBe(null);
      }
    });

    test('should get API key from environment variable', () => {
      global.window.SAMO_CONFIG.API.API_KEY_ENV = 'env-key-789';

      const key = apiClient.getApiKey();

      expect(key).toBe('env-key-789');

      // Cleanup
      delete global.window.SAMO_CONFIG.API.API_KEY_ENV;
    });

    test('should return null when no API key available', () => {
      const key = apiClient.getApiKey();

      expect(key).toBe(null);
    });
  });

  describe('Query String Building', () => {
    test('should build query string from object', () => {
      const data = { text: 'hello world', threshold: 0.5 };
      const queryString = apiClient.buildQueryString(data);

      expect(queryString).toBe('text=hello+world&threshold=0.5');
    });

    test('should handle empty object', () => {
      const queryString = apiClient.buildQueryString({});

      expect(queryString).toBe('');
    });

    test('should handle null/undefined values', () => {
      const data = { text: 'hello', nullValue: null, undefinedValue: undefined };
      const queryString = apiClient.buildQueryString(data);

      expect(queryString).toBe('text=hello');
    });

    test('should handle null/undefined data', () => {
      expect(apiClient.buildQueryString(null)).toBe('');
      expect(apiClient.buildQueryString(undefined)).toBe('');
    });

    test('should handle special characters', () => {
      const data = { text: 'hello & world!' };
      const queryString = apiClient.buildQueryString(data);

      expect(queryString).toBe('text=hello+%26+world%21');
    });
  });

  describe('Request Making - Success Cases', () => {
    test('should make successful GET request', async () => {
      const mockResponse = { success: true, data: 'test' };
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await apiClient.makeRequest('/test', null, 'GET');

      expect(result).toEqual(mockResponse);
      expect(fetch).toHaveBeenCalledWith(
        'https://test-api.com/test',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          })
        })
      );
    });

    test('should make successful POST request with JSON data', async () => {
      const mockResponse = { emotions: { joy: 0.8 } };
      const requestData = { text: 'I am happy!' };

      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await apiClient.makeRequest('/analyze/emotion', requestData, 'POST');

      expect(result).toEqual(mockResponse);
      expect(fetch).toHaveBeenCalledWith(
        'https://test-api.com/analyze/emotion?text=I+am+happy%21',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          })
        })
      );
    });

    test('should make successful POST request with FormData', async () => {
      const mockResponse = { transcription: 'Hello world' };
      const formData = new FormData();
      formData.append('audio', 'fake-audio-data');

      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse
      });

      const result = await apiClient.makeRequest('/transcribe', formData, 'POST', true);

      expect(result).toEqual(mockResponse);
      expect(fetch).toHaveBeenCalledWith(
        'https://test-api.com/transcribe',
        expect.objectContaining({
          method: 'POST',
          body: formData
        })
      );

      // Should not set Content-Type for FormData (browser sets it with boundary)
      expect(fetch.mock.calls[0][1].headers['Content-Type']).toBeUndefined();
    });

    test('should include API key in headers when available', async () => {
      if (localStorage.getItem && localStorage.getItem.mockReturnValue) {
        localStorage.getItem.mockReturnValue('test-api-key');
        fetch.mockResolvedValueOnce({
          ok: true,
          json: async () => ({ success: true })
        });

        await apiClient.makeRequest('/test', null, 'GET');

        expect(fetch).toHaveBeenCalledWith(
          'https://test-api.com/test',
          expect.objectContaining({
            headers: expect.objectContaining({
              'X-API-Key': 'test-api-key'
            })
          })
        );
      }
    });

    test('should use cold start timeout for first request', async () => {
      jest.useFakeTimers();

      // Mock fetch to never resolve (simulate timeout)
      fetch.mockImplementation(() => new Promise(() => {}));

      const requestPromise = apiClient.makeRequest('/test');

      // Fast-forward time to just before cold start timeout
      jest.advanceTimersByTime(44000);

      // Request should still be pending
      expect(fetch).toHaveBeenCalled();

      // Clean up
      jest.useRealTimers();
    });

    test('should mark cold start as complete after first successful request', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      });

      expect(apiClient.isColdStart).toBe(true);

      await apiClient.makeRequest('/test');

      expect(apiClient.isColdStart).toBe(false);
    });
  });

  describe('Error Handling', () => {
    test('should handle network errors', async () => {
      fetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('Network error');
    });

    test('should handle HTTP 400 errors', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ error: 'Bad request' })
      });

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('Bad request');
    });

    test('should handle HTTP 401 errors with custom message', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({})
      });

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('API key required.');
    });

    test('should handle HTTP 429 rate limit errors', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: async () => ({ message: 'Rate limited' })
      });

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('Rate limited');
    });

    test('should handle HTTP 503 service unavailable', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({})
      });

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('Service temporarily unavailable.');
    });

    test('should handle malformed JSON responses', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error('Invalid JSON');
        }
      });

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('HTTP 500');
    });

    test('should handle AbortError (timeout)', async () => {
      const abortError = new Error('Request timeout');
      abortError.name = 'AbortError';
      fetch.mockRejectedValueOnce(abortError);

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('Request timeout');
    });
  });

  describe('Retry Logic', () => {
    test('should retry on HTTP 500 errors', async () => {
      fetch
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: 'Internal server error' })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ success: true })
        });

      const result = await apiClient.makeRequest('/test');

      expect(result).toEqual({ success: true });
      expect(fetch).toHaveBeenCalledTimes(2);
    });

    test('should retry on HTTP 429 rate limit errors', async () => {
      fetch
        .mockResolvedValueOnce({
          ok: false,
          status: 429,
          json: async () => ({ error: 'Rate limited' })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ success: true })
        });

      const result = await apiClient.makeRequest('/test');

      expect(result).toEqual({ success: true });
      expect(fetch).toHaveBeenCalledTimes(2);
    });

    test('should retry on network errors', async () => {
      fetch
        .mockRejectedValueOnce(new Error('Network failure'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ success: true })
        });

      const result = await apiClient.makeRequest('/test');

      expect(result).toEqual({ success: true });
      expect(fetch).toHaveBeenCalledTimes(2);
    });

    test('should not retry on non-retryable errors (400)', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ error: 'Bad request' })
      });

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('Bad request');
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    test('should stop retrying after max attempts', async () => {
      // Mock 4 failures (initial + 3 retries)
      fetch
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: 'Server error' })
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: 'Server error' })
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: 'Server error' })
        })
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: 'Server error' })
        });

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('Server error');
      expect(fetch).toHaveBeenCalledTimes(4); // Initial + 3 retries
    });

    test('should implement exponential backoff between retries', async () => {
      jest.useFakeTimers();

      fetch
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: 'Server error' })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ success: true })
        });

      const requestPromise = apiClient.makeRequest('/test');

      // Wait for first request to complete
      await jest.runOnlyPendingTimersAsync();

      // Should wait 2^0 * 1000 = 1000ms before first retry
      expect(fetch).toHaveBeenCalledTimes(1);

      // Advance time for backoff delay
      jest.advanceTimersByTime(1001);
      await jest.runOnlyPendingTimersAsync();

      // Now second request should be made
      expect(fetch).toHaveBeenCalledTimes(2);

      const result = await requestPromise;
      expect(result).toEqual({ success: true });

      jest.useRealTimers();
    }, 10000);
  });

  describe('Request Tracking and Cancellation', () => {
    test('should track active requests with LayoutManager', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      });

      await apiClient.makeRequest('/test');

      expect(LayoutManager.addActiveRequest).toHaveBeenCalled();
      expect(LayoutManager.removeActiveRequest).toHaveBeenCalled();
    });

    test('should remove request from tracking on completion', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true })
      });

      await apiClient.makeRequest('/test');

      expect(LayoutManager.removeActiveRequest).toHaveBeenCalled();
    });

    test('should remove request from tracking on error', async () => {
      fetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(apiClient.makeRequest('/test')).rejects.toThrow('Network error');

      expect(LayoutManager.removeActiveRequest).toHaveBeenCalled();
    });
  });

  describe('Timeout Handling', () => {
    test('should use regular timeout for non-cold-start requests', async () => {
      jest.useFakeTimers();

      // Make cold start false
      apiClient.isColdStart = false;

      fetch.mockImplementation(() => new Promise(() => {})); // Never resolves

      const requestPromise = apiClient.makeRequest('/test');

      // Should timeout after regular timeout (15000ms)
      jest.advanceTimersByTime(15001);

      await expect(requestPromise).rejects.toThrow();

      jest.useRealTimers();
    }, 10000);

    test('should use custom timeout when specified', async () => {
      jest.useFakeTimers();

      fetch.mockImplementation(() => new Promise(() => {}));

      const requestPromise = apiClient.makeRequest('/test', null, 'POST', false, 5000);

      jest.advanceTimersByTime(5001);

      await expect(requestPromise).rejects.toThrow();

      jest.useRealTimers();
    }, 10000);
  });

  describe('Specific API Methods', () => {
    describe('transcribeAudio', () => {
      test('should transcribe audio file successfully', async () => {
        const mockFile = new File(['audio data'], 'test.wav', { type: 'audio/wav' });
        const mockResponse = { transcription: { text: 'Hello world' } };

        fetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse
        });

        const result = await apiClient.transcribeAudio(mockFile);

        expect(result).toEqual(mockResponse);
        expect(fetch).toHaveBeenCalledWith(
          expect.stringContaining('/analyze/voice-journal'),
          expect.objectContaining({
            method: 'POST',
            body: expect.any(FormData)
          })
        );
      });

      test('should handle transcription errors', async () => {
        const mockFile = new File(['audio data'], 'test.wav', { type: 'audio/wav' });

        fetch.mockRejectedValueOnce(new Error('Transcription failed'));

        await expect(apiClient.transcribeAudio(mockFile)).rejects.toThrow('Transcription failed');
      });
    });

    describe('summarizeText', () => {
      test('should summarize text successfully', async () => {
        const mockResponse = { summary: 'Short summary', original_length: 100 };

        fetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse
        });

        const result = await apiClient.summarizeText('This is a long text to summarize.');

        expect(result).toEqual(mockResponse);
        expect(fetch).toHaveBeenCalledWith(
          expect.stringContaining('/analyze/summarize'),
          expect.objectContaining({
            method: 'POST'
          })
        );
      });

      test('should return mock data when API is unavailable', async () => {
        fetch.mockRejectedValueOnce(new Error('Rate limit exceeded'));

        const result = await apiClient.summarizeText('Test text');

        expect(result.mock).toBe(true);
        expect(result.summary).toContain('Test text');
      });

      test('should return mock data on API key error', async () => {
        fetch.mockRejectedValueOnce(new Error('API key required'));

        const result = await apiClient.summarizeText('Test text');

        expect(result.mock).toBe(true);
      });
    });

    describe('detectEmotions', () => {
      test('should detect emotions successfully', async () => {
        const mockResponse = {
          emotions: { joy: 0.8, excitement: 0.6, optimism: 0.5 },
          predicted_emotion: 'joy'
        };

        fetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockResponse
        });

        const result = await apiClient.detectEmotions('I am very happy!');

        expect(result.emotions).toEqual(mockResponse.emotions);
        expect(result.top_emotions).toHaveLength(3);
        expect(result.top_emotions[0]).toEqual({ emotion: 'joy', confidence: 0.8 });
      });

      test('should return mock data when API is unavailable', async () => {
        fetch.mockRejectedValueOnce(new Error('Service temporarily unavailable'));

        const result = await apiClient.detectEmotions('Test text');

        expect(result.mock).toBe(true);
        expect(result.emotions).toBeDefined();
        expect(result.top_emotions).toBeDefined();
      });
    });

    describe('processCompleteWorkflow', () => {
      test('should process complete workflow with text only', async () => {
        const mockSummary = { summary: 'Summary text' };
        const mockEmotions = { emotions: { joy: 0.8 }, predicted_emotion: 'joy' };

        fetch
          .mockResolvedValueOnce({ ok: true, json: async () => mockSummary })
          .mockResolvedValueOnce({ ok: true, json: async () => mockEmotions });

        const result = await apiClient.processCompleteWorkflow(null, 'Happy test text');

        expect(result.summary).toEqual(mockSummary);
        expect(result.emotions).toEqual(expect.objectContaining(mockEmotions));
        expect(result.modelsUsed).toContain('SAMO T5');
        expect(result.modelsUsed).toContain('SAMO DeBERTa v3 Large');
        expect(result.processingTime).toBeGreaterThan(0);
      });

      test('should process complete workflow with audio', async () => {
        const mockFile = new File(['audio'], 'test.wav', { type: 'audio/wav' });
        const mockAudioResponse = {
          transcription: { text: 'Transcribed text' },
          summary: { summary: 'Audio summary' },
          emotion_analysis: { emotions: { joy: 0.9 } }
        };

        fetch.mockResolvedValueOnce({
          ok: true,
          json: async () => mockAudioResponse
        });

        const result = await apiClient.processCompleteWorkflow(mockFile, null);

        expect(result.transcription).toEqual(mockAudioResponse.transcription);
        expect(result.summary).toEqual(mockAudioResponse.summary);
        expect(result.emotions).toEqual(mockAudioResponse.emotion_analysis);
        expect(result.modelsUsed).toContain('SAMO Whisper');
      });

      test('should handle transcription failure', async () => {
        const mockFile = new File(['audio'], 'test.wav', { type: 'audio/wav' });

        fetch.mockRejectedValueOnce(new Error('Transcription failed'));

        await expect(
          apiClient.processCompleteWorkflow(mockFile, null)
        ).rejects.toThrow('Voice transcription failed. Please try again.');
      });

      test('should handle emotion detection failure', async () => {
        fetch.mockRejectedValueOnce(new Error('Emotion detection failed'));

        await expect(
          apiClient.processCompleteWorkflow(null, 'Test text')
        ).rejects.toThrow('Emotion detection failed. Please try again.');
      });

      test('should continue without summary if it fails', async () => {
        const mockEmotions = { emotions: { joy: 0.8 }, predicted_emotion: 'joy' };

        fetch
          .mockRejectedValueOnce(new Error('Summary failed'))
          .mockResolvedValueOnce({ ok: true, json: async () => mockEmotions });

        const result = await apiClient.processCompleteWorkflow(null, 'Test text');

        expect(result.summary).toBeNull();
        expect(result.emotions).toEqual(expect.objectContaining(mockEmotions));
      });
    });
  });

  describe('Mock Response Generation', () => {
    test('should generate realistic mock summary response', () => {
      const text = 'This is a long text that needs summarization to test the mock functionality.';
      const mockResponse = apiClient.getMockSummaryResponse(text);

      expect(mockResponse.mock).toBe(true);
      expect(mockResponse.summary).toContain('This is a long text');
      expect(mockResponse.original_length).toBe(text.length);
      expect(mockResponse.summary_length).toBeLessThan(text.length);
      expect(mockResponse.compression_ratio).toBeDefined();
      expect(mockResponse.request_id).toContain('demo-');
    });

    test('should generate realistic mock emotion response', () => {
      const text = 'I am feeling happy and excited today!';
      const mockResponse = apiClient.getMockEmotionResponse(text);

      expect(mockResponse.mock).toBe(true);
      expect(mockResponse.text).toBe(text);
      expect(mockResponse.emotions).toBeDefined();
      expect(mockResponse.predicted_emotion).toBeDefined();
      expect(mockResponse.top_emotions).toHaveLength(5);
      expect(mockResponse.top_emotions[0].confidence).toBeGreaterThanOrEqual(
        mockResponse.top_emotions[1].confidence
      );
    });
  });
});