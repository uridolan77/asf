import React, { createContext, useContext, useState, useEffect } from 'react';

// Define feature flags
interface FeatureFlags {
  useMockData: boolean;
  enableReactQuery: boolean;
  enableNewUI: boolean;
  enableBetaFeatures: boolean;
  enablePerformanceMonitoring: boolean;
  [key: string]: boolean;
}

// Default feature flags
const defaultFeatureFlags: FeatureFlags = {
  useMockData: false, // Set to false by default to use real API
  enableReactQuery: true, // Enable React Query by default
  enableNewUI: false, // Disable new UI by default
  enableBetaFeatures: false, // Disable beta features by default
  enablePerformanceMonitoring: true, // Enable performance monitoring by default
};

// Context type
interface FeatureFlagContextType {
  flags: FeatureFlags;
  setFlag: (flag: keyof FeatureFlags, value: boolean) => void;
  isEnabled: (flag: keyof FeatureFlags) => boolean;
}

// Create context
const FeatureFlagContext = createContext<FeatureFlagContextType | undefined>(undefined);

// Provider props
interface FeatureFlagProviderProps {
  children: React.ReactNode;
  initialFlags?: Partial<FeatureFlags>;
}

/**
 * Feature Flag Provider
 * 
 * Provides feature flags to the application
 */
export const FeatureFlagProvider: React.FC<FeatureFlagProviderProps> = ({ 
  children, 
  initialFlags = {} 
}) => {
  // Initialize flags with defaults and any overrides
  const [flags, setFlags] = useState<FeatureFlags>({
    ...defaultFeatureFlags,
    ...initialFlags,
    // Override with environment variables if available
    ...(import.meta.env.VITE_USE_MOCK_DATA === 'true' ? { useMockData: true } : {}),
    ...(import.meta.env.VITE_ENABLE_REACT_QUERY === 'false' ? { enableReactQuery: false } : {}),
    ...(import.meta.env.VITE_ENABLE_NEW_UI === 'true' ? { enableNewUI: true } : {}),
    ...(import.meta.env.VITE_ENABLE_BETA_FEATURES === 'true' ? { enableBetaFeatures: true } : {}),
  });

  // Load flags from localStorage on mount
  useEffect(() => {
    try {
      const storedFlags = localStorage.getItem('featureFlags');
      if (storedFlags) {
        setFlags(prevFlags => ({
          ...prevFlags,
          ...JSON.parse(storedFlags)
        }));
      }
    } catch (error) {
      console.error('Failed to load feature flags from localStorage:', error);
    }
  }, []);

  // Save flags to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem('featureFlags', JSON.stringify(flags));
    } catch (error) {
      console.error('Failed to save feature flags to localStorage:', error);
    }
  }, [flags]);

  // Set a single flag
  const setFlag = (flag: keyof FeatureFlags, value: boolean) => {
    setFlags(prevFlags => ({
      ...prevFlags,
      [flag]: value
    }));
  };

  // Check if a flag is enabled
  const isEnabled = (flag: keyof FeatureFlags) => {
    return flags[flag] === true;
  };

  return (
    <FeatureFlagContext.Provider value={{ flags, setFlag, isEnabled }}>
      {children}
    </FeatureFlagContext.Provider>
  );
};

/**
 * Hook to use feature flags
 */
export const useFeatureFlags = () => {
  const context = useContext(FeatureFlagContext);
  if (context === undefined) {
    throw new Error('useFeatureFlags must be used within a FeatureFlagProvider');
  }
  return context;
};

/**
 * Component to conditionally render based on feature flag
 */
interface FeatureFlagProps {
  flag: keyof FeatureFlags;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export const FeatureFlag: React.FC<FeatureFlagProps> = ({ 
  flag, 
  children, 
  fallback = null 
}) => {
  const { isEnabled } = useFeatureFlags();
  return isEnabled(flag) ? <>{children}</> : <>{fallback}</>;
};
