# Standardized Error Handling Guide

This guide provides a comprehensive approach for implementing consistent error handling across the BO frontend application.

## Why Standardized Error Handling?

- **Consistent User Experience**: Users see consistent error messages and recovery options
- **Improved Debugging**: Structured error logging makes debugging easier
- **Better Error Recovery**: Standardized approaches for retrying operations
- **Reduced Code Duplication**: Reusable error handling components and utilities
- **Centralized Error Tracking**: Easier to implement error tracking and monitoring

## Implementation Steps

### 1. Create Error Types

First, define standard error types:

```tsx
// src/types/error.ts
export interface ApiError {
  message: string;
  status?: number;
  code?: string;
  details?: Record<string, any>;
}

export interface ValidationError {
  field: string;
  message: string;
}

export interface FormErrors {
  [field: string]: string;
}

export type ErrorSeverity = 'error' | 'warning' | 'info';

export interface ErrorOptions {
  severity?: ErrorSeverity;
  retry?: boolean;
  retryFn?: () => void;
  details?: Record<string, any>;
}
```

### 2. Create Error Utilities

Create utility functions for handling errors:

```tsx
// src/utils/errorHandler.ts
import { ApiError, ValidationError, FormErrors, ErrorOptions } from '../types/error';
import { logErrorToService } from '../services/errorLoggingService';

/**
 * Extract a user-friendly error message from various error types
 */
export function getErrorMessage(error: unknown): string {
  // Handle axios errors
  if (error && typeof error === 'object' && 'response' in error) {
    const axiosError = error as any;
    
    // Handle structured API errors
    if (axiosError.response?.data?.message) {
      return axiosError.response.data.message;
    }
    
    // Handle HTTP status errors
    if (axiosError.response?.status) {
      switch (axiosError.response.status) {
        case 400:
          return 'Invalid request. Please check your input.';
        case 401:
          return 'You are not authenticated. Please log in.';
        case 403:
          return 'You do not have permission to perform this action.';
        case 404:
          return 'The requested resource was not found.';
        case 500:
          return 'An internal server error occurred. Please try again later.';
        default:
          return `Request failed with status ${axiosError.response.status}`;
      }
    }
    
    // Handle network errors
    if (axiosError.message === 'Network Error') {
      return 'Unable to connect to the server. Please check your internet connection.';
    }
    
    return axiosError.message || 'An unknown error occurred';
  }
  
  // Handle ApiError objects
  if (error && typeof error === 'object' && 'message' in error) {
    return (error as ApiError).message;
  }
  
  // Handle Error objects
  if (error instanceof Error) {
    return error.message;
  }
  
  // Handle string errors
  if (typeof error === 'string') {
    return error;
  }
  
  // Default fallback
  return 'An unexpected error occurred';
}

/**
 * Handle authentication errors
 */
export function handleAuthError(error: unknown): boolean {
  if (
    error && 
    typeof error === 'object' && 
    'response' in error && 
    (error as any).response?.status === 401
  ) {
    // Clear token
    localStorage.removeItem('token');
    
    // Redirect to login
    window.location.href = '/login';
    
    return true;
  }
  
  return false;
}

/**
 * Log errors to console and error tracking service
 */
export function logError(error: unknown, context: string = ''): void {
  // Log to console in development
  if (process.env.NODE_ENV !== 'production') {
    console.error(`Error ${context ? `in ${context}` : ''}:`, error);
  }
  
  // Log to error tracking service
  logErrorToService(error, context);
}

/**
 * Convert API validation errors to form errors
 */
export function mapValidationErrorsToFormErrors(
  validationErrors: ValidationError[]
): FormErrors {
  return validationErrors.reduce((acc, { field, message }) => {
    acc[field] = message;
    return acc;
  }, {} as FormErrors);
}

/**
 * Create a standardized error object
 */
export function createErrorObject(
  message: string,
  options?: ErrorOptions
): ApiError {
  return {
    message,
    ...options?.details,
  };
}

/**
 * Check if an error is a specific type
 */
export function isApiError(error: unknown): error is ApiError {
  return (
    error !== null &&
    typeof error === 'object' &&
    'message' in error
  );
}

/**
 * Check if an error is a validation error
 */
export function isValidationError(error: unknown): boolean {
  if (
    error && 
    typeof error === 'object' && 
    'response' in error && 
    (error as any).response?.data?.errors
  ) {
    return Array.isArray((error as any).response.data.errors);
  }
  
  return false;
}
```

### 3. Create Error Components

Create reusable error components:

```tsx
// src/components/UI/ErrorDisplay.tsx
import React from 'react';
import { Alert, AlertTitle, Button, Box, Typography, Paper } from '@mui/material';
import { ErrorOutline, Refresh } from '@mui/icons-material';
import { ErrorSeverity } from '../../types/error';

interface ErrorDisplayProps {
  message: string;
  severity?: ErrorSeverity;
  title?: string;
  retry?: boolean;
  onRetry?: () => void;
  details?: React.ReactNode;
}

/**
 * Generic error display component
 */
export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  message,
  severity = 'error',
  title,
  retry = false,
  onRetry,
  details
}) => {
  return (
    <Alert
      severity={severity}
      action={
        retry && onRetry ? (
          <Button 
            color="inherit" 
            size="small" 
            onClick={onRetry}
            startIcon={<Refresh />}
          >
            Retry
          </Button>
        ) : undefined
      }
      sx={{ mb: 2 }}
    >
      {title && <AlertTitle>{title}</AlertTitle>}
      {message}
      {details && (
        <Box sx={{ mt: 1 }}>
          {details}
        </Box>
      )}
    </Alert>
  );
};

interface FullPageErrorProps {
  message: string;
  retry?: boolean;
  onRetry?: () => void;
}

/**
 * Full page error display
 */
export const FullPageError: React.FC<FullPageErrorProps> = ({
  message,
  retry = false,
  onRetry
}) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        p: 3
      }}
    >
      <Paper
        elevation={3}
        sx={{
          p: 4,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          maxWidth: 500
        }}
      >
        <ErrorOutline color="error" sx={{ fontSize: 64, mb: 2 }} />
        <Typography variant="h5" gutterBottom>
          Something went wrong
        </Typography>
        <Typography align="center" color="text.secondary" paragraph>
          {message}
        </Typography>
        {retry && onRetry && (
          <Button
            variant="contained"
            color="primary"
            startIcon={<Refresh />}
            onClick={onRetry}
            sx={{ mt: 2 }}
          >
            Try Again
          </Button>
        )}
      </Paper>
    </Box>
  );
};

interface FormErrorSummaryProps {
  errors: Record<string, string>;
}

/**
 * Form error summary component
 */
export const FormErrorSummary: React.FC<FormErrorSummaryProps> = ({ errors }) => {
  const errorCount = Object.keys(errors).length;
  
  if (errorCount === 0) {
    return null;
  }
  
  return (
    <Alert severity="error" sx={{ mb: 2 }}>
      <AlertTitle>
        {errorCount === 1
          ? '1 error needs to be fixed'
          : `${errorCount} errors need to be fixed`}
      </AlertTitle>
      <ul style={{ margin: 0, paddingLeft: 20 }}>
        {Object.entries(errors).map(([field, message]) => (
          <li key={field}>{message}</li>
        ))}
      </ul>
    </Alert>
  );
};
```

### 4. Create Error Boundary Component

Create a reusable error boundary component:

```tsx
// src/components/UI/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react';
import { FullPageError } from './ErrorDisplay';
import { logError } from '../../utils/errorHandler';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error boundary component to catch and handle errors in child components
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log the error
    logError(error, 'ErrorBoundary');
    
    // Call onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  handleRetry = (): void => {
    this.setState({
      hasError: false,
      error: null
    });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }
      
      // Use default error display
      return (
        <FullPageError
          message={this.state.error?.message || 'An unexpected error occurred'}
          retry={true}
          onRetry={this.handleRetry}
        />
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

### 5. Create Notification Context

Create a notification context for displaying error messages:

```tsx
// src/context/NotificationContext.tsx
import React, { createContext, useContext, useState, ReactNode } from 'react';
import { Snackbar, Alert, AlertProps } from '@mui/material';

interface Notification {
  id: string;
  message: string;
  severity: AlertProps['severity'];
  autoHideDuration?: number;
}

interface NotificationContextValue {
  showSuccess: (message: string, duration?: number) => void;
  showError: (message: string, duration?: number) => void;
  showWarning: (message: string, duration?: number) => void;
  showInfo: (message: string, duration?: number) => void;
  closeNotification: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextValue | undefined>(undefined);

interface NotificationProviderProps {
  children: ReactNode;
}

export const NotificationProvider: React.FC<NotificationProviderProps> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const showNotification = (
    message: string,
    severity: AlertProps['severity'],
    duration?: number
  ): void => {
    const id = Date.now().toString();
    setNotifications(prev => [...prev, {
      id,
      message,
      severity,
      autoHideDuration: duration || 6000
    }]);
  };

  const showSuccess = (message: string, duration?: number): void => {
    showNotification(message, 'success', duration);
  };

  const showError = (message: string, duration?: number): void => {
    showNotification(message, 'error', duration);
  };

  const showWarning = (message: string, duration?: number): void => {
    showNotification(message, 'warning', duration);
  };

  const showInfo = (message: string, duration?: number): void => {
    showNotification(message, 'info', duration);
  };

  const closeNotification = (id: string): void => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  };

  return (
    <NotificationContext.Provider
      value={{
        showSuccess,
        showError,
        showWarning,
        showInfo,
        closeNotification
      }}
    >
      {children}
      {notifications.map(notification => (
        <Snackbar
          key={notification.id}
          open={true}
          autoHideDuration={notification.autoHideDuration}
          onClose={() => closeNotification(notification.id)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={() => closeNotification(notification.id)}
            severity={notification.severity}
            variant="filled"
            elevation={6}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      ))}
    </NotificationContext.Provider>
  );
};

export const useNotification = (): NotificationContextValue => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};
```

### 6. Create Error Logging Service

Create a service for logging errors:

```tsx
// src/services/errorLoggingService.ts
interface ErrorLogData {
  message: string;
  stack?: string;
  context?: string;
  timestamp: string;
  userAgent: string;
  url: string;
  additionalData?: Record<string, any>;
}

/**
 * Log error to a service (e.g., Sentry, LogRocket)
 */
export function logErrorToService(
  error: unknown,
  context: string = '',
  additionalData: Record<string, any> = {}
): void {
  // Extract error message and stack
  const errorMessage = error instanceof Error ? error.message : String(error);
  const errorStack = error instanceof Error ? error.stack : undefined;
  
  // Create error log data
  const errorLogData: ErrorLogData = {
    message: errorMessage,
    stack: errorStack,
    context,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href,
    additionalData
  };
  
  // In a real application, you would send this to your error logging service
  // For example, with Sentry:
  // Sentry.captureException(error, {
  //   extra: {
  //     context,
  //     ...additionalData
  //   }
  // });
  
  // For now, just log to console in development
  if (process.env.NODE_ENV !== 'production') {
    console.group('Error Log');
    console.error(errorLogData);
    console.groupEnd();
  }
}

/**
 * Initialize error logging service
 */
export function initErrorLogging(): void {
  // Set up global error handler
  window.addEventListener('error', (event) => {
    logErrorToService(event.error, 'Unhandled Error');
    // Don't prevent default to allow browser to handle the error as well
  });
  
  // Set up unhandled promise rejection handler
  window.addEventListener('unhandledrejection', (event) => {
    logErrorToService(event.reason, 'Unhandled Promise Rejection');
    // Don't prevent default to allow browser to handle the rejection as well
  });
  
  // In a real application, you would initialize your error logging service here
  // For example, with Sentry:
  // Sentry.init({
  //   dsn: 'your-sentry-dsn',
  //   environment: process.env.NODE_ENV,
  //   release: 'your-app-version'
  // });
}
```

### 7. Implement API Error Handling

Implement error handling in the API client:

```tsx
// src/services/apiClient.ts
import axios, { AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import { getErrorMessage, handleAuthError, logError } from '../utils/errorHandler';

const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    logError(error, 'API Request');
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response.data,
  (error: AxiosError) => {
    // Log the error
    logError(error, 'API Response');
    
    // Handle authentication errors
    if (error.response?.status === 401) {
      handleAuthError(error);
    }
    
    // Transform error for consistent handling
    const errorMessage = getErrorMessage(error);
    const transformedError = {
      message: errorMessage,
      status: error.response?.status,
      code: error.response?.data?.code,
      details: error.response?.data?.details,
      originalError: error
    };
    
    return Promise.reject(transformedError);
  }
);

export default apiClient;
```

### 8. Using Error Handling in Components

```tsx
// src/components/LLM/MCP/MCPProviderManagement.tsx
import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import { useMCPProviders } from '../../../hooks/useMCPProviders';
import { MCPProvider } from '../../../types/api';
import MCPConfigDialog from './MCPConfigDialog';
import { ContentLoader } from '../../UI/LoadingIndicators';
import { ErrorDisplay } from '../../UI/ErrorDisplay';
import ErrorBoundary from '../../UI/ErrorBoundary';
import { useNotification } from '../../../context/NotificationContext';

const MCPProviderManagement: React.FC = () => {
  // State
  const [configDialogOpen, setConfigDialogOpen] = useState<boolean>(false);
  const [configDialogMode, setConfigDialogMode] = useState<'add' | 'edit'>('add');
  const [selectedProvider, setSelectedProvider] = useState<MCPProvider | null>(null);
  
  const { showError } = useNotification();
  
  // Fetch providers using React Query
  const {
    providers,
    isLoading,
    isError,
    error,
    refetch,
    createProvider,
    updateProvider,
    deleteProvider
  } = useMCPProviders();
  
  // Handle provider creation
  const handleCreateProvider = () => {
    setSelectedProvider(null);
    setConfigDialogMode('add');
    setConfigDialogOpen(true);
  };
  
  // Handle provider edit
  const handleEditProvider = (provider: MCPProvider) => {
    setSelectedProvider(provider);
    setConfigDialogMode('edit');
    setConfigDialogOpen(true);
  };
  
  // Handle provider save
  const handleSaveProvider = (provider: MCPProvider) => {
    try {
      if (configDialogMode === 'add') {
        createProvider(provider);
      } else {
        updateProvider(provider);
      }
      setConfigDialogOpen(false);
    } catch (error) {
      showError(`Failed to save provider: ${getErrorMessage(error)}`);
    }
  };
  
  // Handle provider delete
  const handleDeleteProvider = (id: string) => {
    try {
      deleteProvider({ id });
    } catch (error) {
      showError(`Failed to delete provider: ${getErrorMessage(error)}`);
    }
  };
  
  // Render loading state
  if (isLoading) {
    return <ContentLoader />;
  }
  
  // Render error state
  if (isError) {
    return (
      <ErrorDisplay
        message={`Failed to load providers: ${error?.message || 'Unknown error'}`}
        retry={true}
        onRetry={() => refetch()}
      />
    );
  }
  
  return (
    <ErrorBoundary>
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6">MCP Providers</Typography>
          <Box>
            <Button
              variant="contained"
              color="primary"
              onClick={handleCreateProvider}
            >
              Add Provider
            </Button>
            <IconButton onClick={() => refetch()} sx={{ ml: 1 }}>
              <RefreshIcon />
            </IconButton>
          </Box>
        </Box>
        
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Name</TableCell>
                <TableCell>Transport</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {providers.map((provider) => (
                <TableRow key={provider.id}>
                  <TableCell>{provider.id}</TableCell>
                  <TableCell>{provider.name}</TableCell>
                  <TableCell>{provider.transport_type}</TableCell>
                  <TableCell>
                    <Chip
                      label={provider.status}
                      color={provider.status === 'connected' ? 'success' : 'error'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <IconButton onClick={() => handleEditProvider(provider)}>
                      <EditIcon />
                    </IconButton>
                    <IconButton onClick={() => handleDeleteProvider(provider.id)}>
                      <DeleteIcon />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <MCPConfigDialog
          open={configDialogOpen}
          mode={configDialogMode}
          provider={selectedProvider}
          onClose={() => setConfigDialogOpen(false)}
          onSave={handleSaveProvider}
        />
      </Box>
    </ErrorBoundary>
  );
};

export default MCPProviderManagement;
```

## Best Practices

### 1. Use Error Boundaries

Wrap components in error boundaries to catch and handle errors:

```tsx
<ErrorBoundary>
  <MyComponent />
</ErrorBoundary>
```

### 2. Provide Retry Options

Always provide retry options for recoverable errors:

```tsx
<ErrorDisplay
  message="Failed to load data"
  retry={true}
  onRetry={() => refetch()}
/>
```

### 3. Use Consistent Error Messages

Use consistent error messages for similar errors:

```tsx
// Good
showError(`Failed to load providers: ${error.message}`);

// Bad
showError('Error occurred while fetching providers');
```

### 4. Log Errors

Always log errors for debugging:

```tsx
try {
  // Some operation
} catch (error) {
  logError(error, 'ComponentName');
  showError(`Operation failed: ${getErrorMessage(error)}`);
}
```

### 5. Handle Different Error Types

Handle different error types appropriately:

```tsx
try {
  // Some operation
} catch (error) {
  if (isValidationError(error)) {
    // Handle validation errors
    const formErrors = mapValidationErrorsToFormErrors(error.response.data.errors);
    setErrors(formErrors);
  } else if (error.response?.status === 404) {
    // Handle not found errors
    showError('The requested resource was not found');
  } else {
    // Handle other errors
    showError(`Operation failed: ${getErrorMessage(error)}`);
  }
}
```

### 6. Use Error Context

Use error context to provide error handling utilities:

```tsx
// In your app
<NotificationProvider>
  <App />
</NotificationProvider>

// In your component
const { showError } = useNotification();

// Later in your component
showError('Something went wrong');
```

## Migration Strategy

1. **Create error utilities**: Implement error handling utilities
2. **Create error components**: Create reusable error components
3. **Implement error boundaries**: Add error boundaries to key components
4. **Update API client**: Add error handling to API client
5. **Update components**: Update components to use standardized error handling
6. **Add error logging**: Implement error logging service

## Resources

- [React Error Boundaries](https://reactjs.org/docs/error-boundaries.html)
- [Error Handling in React 16+](https://blog.logrocket.com/error-handling-react-16-react-error-boundaries/)
- [Handling API Errors in React](https://kentcdodds.com/blog/handle-api-errors-in-react-with-the-error-boundary-component)
