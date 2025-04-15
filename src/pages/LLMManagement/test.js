import React from 'react';
import { render } from '@testing-library/react';
import UsageDashboard from './UsageDashboard';

// Mock the dependencies
jest.mock('../../context/NotificationContext', () => ({
  useNotification: () => ({
    showSuccess: jest.fn(),
    showError: jest.fn()
  })
}));

jest.mock('../../services/api', () => ({
  llm: {
    getUsageStatistics: jest.fn().mockResolvedValue({
      success: true,
      data: {
        total_requests: 100,
        components: {
          gateway: {
            total_requests: 50,
            total_tokens: 1000,
            average_latency_ms: 200,
            providers: {
              'openai': { requests: 30 },
              'anthropic': { requests: 20 }
            },
            models: {
              'gpt-4': { requests: 20, tokens: 500 },
              'claude-3': { requests: 30, tokens: 500 }
            }
          },
          dspy: {
            total_requests: 30,
            modules: {
              'rag': 20,
              'qa': 10
            }
          },
          biomedlm: {
            total_requests: 20,
            models: {
              'biomedlm-base': 10,
              'biomedlm-large': 10
            }
          }
        }
      }
    })
  }
}));

jest.mock('../../components/UI/LoadingIndicators', () => ({
  ContentLoader: () => <div>Loading...</div>
}));

describe('UsageDashboard', () => {
  it('renders without crashing', () => {
    render(<UsageDashboard />);
    // If the component renders without throwing an error, the test passes
  });
});

// Run the test
const test = () => {
  try {
    describe('UsageDashboard', () => {
      it('renders without crashing', () => {
        render(<UsageDashboard />);
        console.log('Test passed: UsageDashboard renders without crashing');
      });
    });
  } catch (error) {
    console.error('Test failed:', error);
  }
};

test();
