import React, { useEffect } from 'react';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { SnackbarProvider } from 'notistack';
import { theme } from '../theme';
import QueryProvider from './QueryProvider';
import { FeatureFlagProvider } from '../context/FeatureFlagContext';
import { NotificationProvider } from '../context/NotificationContext';

interface AppProvidersProps {
  children: React.ReactNode;
}

/**
 * AppProviders component
 * 
 * Wraps the application with all necessary providers:
 * - BrowserRouter for routing
 * - ThemeProvider for MUI theming
 * - FeatureFlagProvider for feature flags
 * - QueryProvider for React Query
 * - NotificationProvider for notifications
 * - SnackbarProvider for snackbar notifications
 */
const AppProviders: React.FC<AppProvidersProps> = ({ children }) => {
  // Initialize MSW if needed
  useEffect(() => {
    const initMocks = async () => {
      // Check if we should use mock data from localStorage
      const storedFlags = localStorage.getItem('featureFlags');
      const useMockData = storedFlags 
        ? JSON.parse(storedFlags).useMockData 
        : import.meta.env.VITE_USE_MOCK_DATA === 'true';
      
      if (useMockData && process.env.NODE_ENV !== 'production') {
        // Import and initialize MSW
        const initMocksFn = await import('../mocks').then(module => module.default);
        await initMocksFn();
      }
    };
    
    initMocks();
  }, []);
  
  return (
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <FeatureFlagProvider>
          <QueryProvider>
            <NotificationProvider>
              <SnackbarProvider 
                maxSnack={3}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                autoHideDuration={5000}
              >
                {children}
              </SnackbarProvider>
            </NotificationProvider>
          </QueryProvider>
        </FeatureFlagProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
};

export default AppProviders;
