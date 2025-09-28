// Test setup for SAMO-DL website tests
// This file runs before each test suite

// Mock window.SAMO_CONFIG for tests
global.window = global.window || {};
window.SAMO_CONFIG = {
  API: {
    BASE_URL: 'http://localhost:3000',
    ENDPOINTS: {
      EMOTION: '/analyze/emotion',
      HEALTH: '/health'
    },
    TIMEOUTS: {
      DEFAULT: 5000
    }
  },
  UI: {
    DEMO: {
      MAX_TEXT_LENGTH: 5000
    }
  }
};

// Mock navigator.mediaDevices for voice recording tests (with guard)
if (typeof navigator !== 'undefined') {
  Object.defineProperty(navigator, 'mediaDevices', {
    value: {
      getUserMedia: vi.fn().mockResolvedValue({
        getTracks: () => [{ stop: vi.fn() }]
      })
    },
    writable: true
  });
}

// Optional: matchMedia stub
if (!window.matchMedia) {
  window.matchMedia = vi.fn().mockReturnValue({
    matches: false,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  });
}

// Mock MediaRecorder
global.MediaRecorder = vi.fn().mockImplementation(() => ({
  start: vi.fn(),
  stop: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn()
}));

// Mock fetch for API calls
global.fetch = vi.fn();

// Cleanup after each test
afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  document.body.innerHTML = '';
});
