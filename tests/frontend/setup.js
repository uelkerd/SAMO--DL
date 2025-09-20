/**
 * Jest Test Setup
 * Configures JSDOM environment and global mocks for browser APIs
 */

// Mock browser globals and APIs
global.fetch = jest.fn();
global.AbortController = jest.fn(() => ({
  signal: {},
  abort: jest.fn()
}));

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// Mock console methods to reduce test noise (can be overridden per test)
global.console = {
  ...console,
  log: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
  info: jest.fn(),
};

// Mock window.alert, confirm, prompt
global.alert = jest.fn();
global.confirm = jest.fn(() => true);
global.prompt = jest.fn(() => 'test-input');

// Mock performance API
global.performance = {
  now: jest.fn(() => Date.now()),
  mark: jest.fn(),
  measure: jest.fn(),
};

// Mock URL constructor
global.URL = jest.fn((url) => ({
  href: url,
  toString: () => url
}));

// Mock URLSearchParams with working implementation
global.URLSearchParams = jest.fn().mockImplementation(() => {
  const params = new Map();
  return {
    append: jest.fn((key, value) => {
      params.set(key, value);
    }),
    toString: jest.fn(() => {
      const entries = [];
      for (const [key, value] of params) {
        entries.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
      }
      return entries.join('&');
    })
  };
});

// Mock setTimeout/setInterval for deterministic testing
jest.useFakeTimers();

// Custom Jest matchers for DOM testing
expect.extend({
  toBeVisible(received) {
    const pass = received &&
                 received.style.display !== 'none' &&
                 !received.classList.contains('d-none') &&
                 received.style.visibility !== 'hidden';

    if (pass) {
      return {
        message: () => `expected element not to be visible`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected element to be visible`,
        pass: false,
      };
    }
  },

  toHaveText(received, expectedText) {
    const pass = received &&
                 (received.textContent === expectedText ||
                  received.innerText === expectedText ||
                  received.value === expectedText);

    if (pass) {
      return {
        message: () => `expected element not to have text "${expectedText}"`,
        pass: true,
      };
    } else {
      const actualText = received ?
        (received.textContent || received.innerText || received.value) :
        'null';
      return {
        message: () => `expected element to have text "${expectedText}" but got "${actualText}"`,
        pass: false,
      };
    }
  }
});

// Global test utilities
global.createMockElement = (tag = 'div', attributes = {}) => {
  const element = document.createElement(tag);

  // Mock classList methods as Jest functions
  element.classList.add = jest.fn();
  element.classList.remove = jest.fn();
  element.classList.toggle = jest.fn();
  element.classList.contains = jest.fn((className) => {
    return element.className.split(' ').includes(className);
  });

  Object.keys(attributes).forEach(key => {
    if (key === 'class') {
      element.className = attributes[key];
    } else if (key === 'style') {
      element.style.cssText = attributes[key];
    } else {
      element.setAttribute(key, attributes[key]);
    }
  });
  return element;
};

global.createMockDOMEnvironment = () => {
  // Clear document body
  document.body.innerHTML = '';

  // Create basic DOM structure that tests expect
  const mockElements = {
    textInput: createMockElement('textarea', {
      id: 'textInput',
      value: ''
    }),
    processBtn: createMockElement('button', {
      id: 'processBtn'
    }),
    generateBtn: createMockElement('button', {
      id: 'generateBtn'
    }),
    clearBtn: createMockElement('button', {
      id: 'clearBtn'
    }),
    emotionChart: createMockElement('div', {
      id: 'emotionChart'
    }),
    progressConsole: createMockElement('div', {
      id: 'progressConsole'
    }),
    progressConsoleRow: createMockElement('div', {
      id: 'progressConsoleRow',
      style: 'display: none;'
    }),
    resultsLayout: createMockElement('div', {
      id: 'resultsLayout',
      class: 'd-none'
    }),
    inputLayout: createMockElement('div', {
      id: 'inputLayout'
    }),
    emotionResults: createMockElement('div', {
      id: 'emotionResults',
      class: 'result-section-hidden'
    }),
    summarizationResults: createMockElement('div', {
      id: 'summarizationResults',
      class: 'result-section-hidden'
    }),
    primaryEmotion: createMockElement('span', {
      id: 'primaryEmotion'
    }),
    summaryText: createMockElement('div', {
      id: 'summaryText'
    }),
    processingStatusCompact: createMockElement('span', {
      id: 'processingStatusCompact'
    }),
    totalTimeCompact: createMockElement('span', {
      id: 'totalTimeCompact'
    })
  };

  // Add elements to DOM
  Object.values(mockElements).forEach(element => {
    document.body.appendChild(element);
  });

  return mockElements;
};

// Reset between tests
beforeEach(() => {
  // Clear all mocks
  jest.clearAllMocks();

  // Reset localStorage
  localStorageMock.getItem.mockClear();
  localStorageMock.setItem.mockClear();
  localStorageMock.removeItem.mockClear();
  localStorageMock.clear.mockClear();

  // Reset fetch mock
  fetch.mockClear();

  // Reset timers
  jest.clearAllTimers();

  // Clear console mocks
  console.log.mockClear();
  console.warn.mockClear();
  console.error.mockClear();
  console.info.mockClear();

  // Clear DOM
  document.body.innerHTML = '';
});

afterEach(() => {
  // Run pending timers
  jest.runOnlyPendingTimers();
});