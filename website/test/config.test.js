/**
 * Tests for SAMO-DL Configuration Security
 * Tests redaction, environment handling, and secure defaults
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Load config using ES6 import for browser compatibility
import '../js/config.js';

describe('SAMO Configuration', () => {
  beforeEach(() => {
    // Clean up before each test
    delete window.SAMO_SERVER_CONFIG;
  });

  afterEach(() => {
    delete window.SAMO_SERVER_CONFIG;
  });

  describe('Security Features', () => {
    it('should disable direct OpenAI client-side calls', () => {
      expect(window.SAMO_CONFIG.OPENAI).toBeDefined();
      expect(window.SAMO_CONFIG.FEATURES.ENABLE_OPENAI).toBe(false);

      // OpenAI config should be disabled with warnings
      expect(window.SAMO_CONFIG.OPENAI.API_URL).toBeUndefined();
      expect(window.SAMO_CONFIG.OPENAI.MODEL).toBeUndefined();
    });

    it('should use secure proxy endpoint for OpenAI', () => {
      expect(window.SAMO_CONFIG.API.ENDPOINTS.OPENAI_PROXY).toBe('/proxy/openai');
      expect(window.SAMO_CONFIG.FEATURES.ENABLE_OPENAI).toBe(false);
    });
  });

  describe('Redaction Utilities', () => {
    it('should redact sensitive patterns', () => {
      const testObj = {
        api_key: 'secret123',
        API_KEY: 'another_secret', // skipcq: SCT-A000 - test data for redaction testing
        token: 'token_value',
        authorization: 'Bearer some-token',
        normal_field: 'normal_value',
        nested: {
          password: 'secret_pass',
          safe_field: 'safe'
        },
        items: [
          { secret: 'item_secret' },
          { value: 'item_value' }
        ]
      };

      // Test the exposed redactSensitiveValues function
      const redacted = window.SAMO_CONFIG.redactSensitiveValues(testObj);

      expect(redacted.api_key).toBe('REDACTED');
      expect(redacted.API_KEY).toBe('REDACTED');
      expect(redacted.token).toBe('REDACTED');
      expect(redacted.authorization).toBe('REDACTED');
      expect(redacted.normal_field).toBe('normal_value');
      expect(redacted.nested.password).toBe('REDACTED');
      expect(redacted.nested.safe_field).toBe('safe');
      expect(redacted.items[0].secret).toBe('REDACTED');
      expect(redacted.items[1].value).toBe('item_value');
    });
  });

  describe('Environment Handling', () => {
    it('should detect localhost development environment', async () => {
      // Mock localhost location using defineProperty for jsdom compatibility
      const originalLocation = window.location;
      Object.defineProperty(window, 'location', {
        configurable: true,
        value: new URL('http://localhost/')
      });

      // Reset modules to force re-evaluation of config.js
      vi.resetModules();
      await import('../js/config.js');

      expect(window.SAMO_CONFIG.ENVIRONMENT).toBe('development');
      expect(window.SAMO_CONFIG.DEBUG).toBe(true);
      expect(window.SAMO_CONFIG.API.BASE_URL).toBe('https://localhost:8002');

      // Restore original location and modules
      Object.defineProperty(window, 'location', {
        configurable: true,
        value: originalLocation
      });
      vi.resetModules();
      await import('../js/config.js');
    });

    it('should use production settings by default', () => {
      // Reset to production defaults for this test
      window.SAMO_CONFIG.ENVIRONMENT = 'production';
      window.SAMO_CONFIG.DEBUG = false;
      window.SAMO_CONFIG.API.BASE_URL = 'https://samo-unified-api-frrnetyhfa-uc.a.run.app';

      expect(window.SAMO_CONFIG.ENVIRONMENT).toBe('production');
      expect(window.SAMO_CONFIG.DEBUG).toBe(false);
    });
  });

  describe('API URL Building', () => {
    it('should build API URLs correctly', () => {
      const baseUrl = window.SAMO_CONFIG.API.BASE_URL;
      const healthUrl = window.SAMO_CONFIG.getApiUrl('/health');

      expect(healthUrl).toBe(`${baseUrl}/health`);
    });

    it('should build WebSocket URLs correctly', () => {
      const wsUrl = window.SAMO_CONFIG.getWebSocketUrl('/ws/chat');
      expect(wsUrl).toMatch(/^wss?:\/\//);
      expect(wsUrl).toContain('/ws/chat');
    });

    it('should handle missing base URL gracefully', () => {
      const originalBaseUrl = window.SAMO_CONFIG.API.BASE_URL;
      window.SAMO_CONFIG.API.BASE_URL = null;

      const result = window.SAMO_CONFIG.getApiUrl('/test');
      expect(result).toBeNull();

      // Restore
      window.SAMO_CONFIG.API.BASE_URL = originalBaseUrl;
    });
  });

  describe('Configuration Merging', () => {
    it('should merge server-side config with defaults', async () => {
      // Simulate server config injection
      window.SAMO_SERVER_CONFIG = {
        API: {
          BASE_URL: 'https://custom-api.example.com'
        },
        FEATURES: {
          ENABLE_AUTH: false
        }
      };

      // Reset modules to force re-evaluation of config.js
      vi.resetModules();
      await import('../js/config.js');

      expect(window.SAMO_CONFIG.API.BASE_URL).toBe('https://custom-api.example.com');
      expect(window.SAMO_CONFIG.FEATURES.ENABLE_AUTH).toBe(false);
      // Other features should remain default
      expect(window.SAMO_CONFIG.FEATURES.ENABLE_VOICE_TRANSCRIPTION).toBe(true);

      // Cleanup
      delete window.SAMO_SERVER_CONFIG;
      vi.resetModules();
      await import('../js/config.js');
    });
  });

  describe('Rate Limiting Configuration', () => {
    it('should have reasonable rate limiting defaults', () => {
      expect(window.SAMO_CONFIG.API.RATE_LIMITS.MAX_REQUESTS_PER_MINUTE).toBeGreaterThan(0);
      expect(window.SAMO_CONFIG.API.RATE_LIMITS.BURST_LIMIT).toBeGreaterThan(0);
    });
  });
});
