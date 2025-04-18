import { useFeatureFlags } from '../context/FeatureFlagContext';

/**
 * Service factory that decides whether to use mock data or real API data
 * based on feature flags.
 * 
 * @param realImplementation - The real API implementation
 * @param mockImplementation - The mock implementation
 * @returns The appropriate implementation based on feature flags
 */
export function useServiceFactory<T>(
  realImplementation: T,
  mockImplementation: T
): T {
  const { isEnabled } = useFeatureFlags();
  
  // Use mock data if the feature flag is enabled
  if (isEnabled('useMockData')) {
    return mockImplementation;
  }
  
  // Otherwise, use the real implementation
  return realImplementation;
}

/**
 * Create a service that can toggle between real and mock implementations
 * 
 * @param realImplementation - The real API implementation
 * @param mockImplementation - The mock implementation
 * @returns A function that returns the appropriate implementation based on the useMockData flag
 */
export function createToggleableService<T>(
  realImplementation: T,
  mockImplementation: T
): (useMockData: boolean) => T {
  return (useMockData: boolean) => {
    return useMockData ? mockImplementation : realImplementation;
  };
}
