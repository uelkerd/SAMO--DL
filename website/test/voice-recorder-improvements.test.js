/**
 * Tests for improved voice recorder initialization
 */

import { describe, test, beforeEach, afterEach, vi } from 'vitest';

// Mock the VoiceRecorder class
const MockVoiceRecorder = vi.fn().mockImplementation(() => ({
    init: vi.fn().mockResolvedValue(undefined)
}));

// Mock the entire module
vi.mock('../js/voice-recorder.js', async () => {
    const actual = await vi.importActual('../js/voice-recorder.js');
    return {
        ...actual,
        VoiceRecorder: MockVoiceRecorder
    };
});

describe('Voice Recorder Improvements', () => {
    let mockApiClient;
    let mockApiClientManager;

    beforeEach(() => {
        // Mock API client
        mockApiClient = {
            transcribe: vi.fn(),
            analyzeEmotion: vi.fn()
        };

        // Mock API client manager
        mockApiClientManager = {
            waitForApiClient: vi.fn(),
            isApiClientAvailable: vi.fn(),
            notifyApiClientReady: vi.fn()
        };

        // Reset global state
        window.apiClient = null;
        window.voiceRecorder = null;
        window.ApiClientManager = mockApiClientManager;
        window.SAMO_CONFIG = {
            API: {
                TIMEOUTS: {
                    API_CLIENT_INIT: 5000
                }
            }
        };
    });

    afterEach(() => {
        // Clean up
        delete window.apiClient;
        delete window.voiceRecorder;
        delete window.ApiClientManager;
        delete window.SAMO_CONFIG;
    });

    describe('Dependency Injection', () => {
        test('should accept API client via dependency injection', async () => {
            // Reset the mock before each test
            MockVoiceRecorder.mockClear();

            // Import the mocked initialization function
            const { initializeVoiceRecorder } = await import('../js/voice-recorder.js');

            // Test with injected API client
            await initializeVoiceRecorder({
                apiClient: mockApiClient,
                apiClientManager: mockApiClientManager
            });

            // Verify VoiceRecorder was called with the injected API client
            expect(MockVoiceRecorder).toHaveBeenCalledWith(mockApiClient);
            expect(window.voiceRecorder).toBeDefined();
        });

        test('should use API client manager when no client injected', async () => {
            // Mock successful API client resolution
            mockApiClientManager.waitForApiClient.mockResolvedValue(mockApiClient);

            // Reset the mock before each test
            MockVoiceRecorder.mockClear();

            const { initializeVoiceRecorder } = await import('../js/voice-recorder.js');

            await initializeVoiceRecorder({
                apiClientManager: mockApiClientManager
            });

            // Verify API client manager was used
            expect(mockApiClientManager.waitForApiClient).toHaveBeenCalledWith({
                timeoutMs: 5000,
                useEventBased: true
            });
            expect(MockVoiceRecorder).toHaveBeenCalledWith(mockApiClient);
        });
    });

    describe('Event-based Initialization', () => {
        test('should dispatch voiceRecorderReady event on success', async () => {
            const eventListener = vi.fn();
            window.addEventListener('voiceRecorderReady', eventListener);

            // Mock successful initialization
            mockApiClientManager.waitForApiClient.mockResolvedValue(mockApiClient);

            // Reset the mock before each test
            MockVoiceRecorder.mockClear();

            const { initializeVoiceRecorder } = await import('../js/voice-recorder.js');

            await initializeVoiceRecorder({
                apiClientManager: mockApiClientManager
            });

            // Verify event was dispatched
            expect(eventListener).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'voiceRecorderReady',
                    detail: expect.objectContaining({
                        voiceRecorder: expect.any(Object)
                    })
                })
            );
            
            window.removeEventListener('voiceRecorderReady', eventListener);
        });

        test('should dispatch voiceRecorderReady event with error details on failure', async () => {
            const eventListener = vi.fn();
            window.addEventListener('voiceRecorderReady', eventListener);

            // Mock API client manager failure
            mockApiClientManager.waitForApiClient.mockRejectedValue(new Error('API client timeout'));

            // Reset the mock before each test
            MockVoiceRecorder.mockClear();

            const { initializeVoiceRecorder } = await import('../js/voice-recorder.js');

            await initializeVoiceRecorder({
                apiClientManager: mockApiClientManager
            });

            // Verify event was dispatched with error details
            expect(eventListener).toHaveBeenCalledWith(
                expect.objectContaining({
                    type: 'voiceRecorderReady',
                    detail: expect.objectContaining({
                        voiceRecorder: expect.any(Object),
                        limitedFunctionality: true,
                        error: 'API client timeout'
                    })
                })
            );
            
            window.removeEventListener('voiceRecorderReady', eventListener);
        });
    });

    describe('Configuration Flexibility', () => {
        test('should use custom configuration when provided', async () => {
            const customConfig = {
                API: {
                    TIMEOUTS: {
                        API_CLIENT_INIT: 10000
                    }
                }
            };

            mockApiClientManager.waitForApiClient.mockResolvedValue(mockApiClient);

            // Reset the mock before each test
            MockVoiceRecorder.mockClear();

            const { initializeVoiceRecorder } = await import('../js/voice-recorder.js');

            await initializeVoiceRecorder({
                apiClientManager: mockApiClientManager,
                config: customConfig
            });

            // Verify custom timeout was used
            expect(mockApiClientManager.waitForApiClient).toHaveBeenCalledWith({
                timeoutMs: 10000,
                useEventBased: true
            });
        });
    });
});
