/**
 * Centralized error handling utility
 */

// Parse error message from various error types
export const getErrorMessage = (error) => {
  // If the error is an Axios error with a response
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    const { data, status } = error.response;
    
    // Check for structured error message in response data
    if (data && data.message) {
      return `Error ${status}: ${data.message}`;
    }
    
    // Check for error details in response data
    if (data && data.error) {
      return `Error ${status}: ${data.error}`;
    }
    
    // Handle common HTTP status codes
    switch (status) {
      case 400:
        return 'Bad request. Please check your input.';
      case 401:
        return 'Unauthorized. Please log in again.';
      case 403:
        return 'Forbidden. You do not have permission to access this resource.';
      case 404:
        return 'Resource not found.';
      case 500:
        return 'Internal server error. Please try again later.';
      default:
        return `Request failed with status code ${status}`;
    }
  } 
  
  // The request was made but no response was received
  if (error.request) {
    return 'No response received from server. Please check your network connection.';
  }
  
  // Something happened in setting up the request that triggered an Error
  if (error.message) {
    return error.message;
  }
  
  // For plain string errors
  if (typeof error === 'string') {
    return error;
  }
  
  // Default fallback
  return 'An unexpected error occurred';
};

// Handle authentication errors
export const handleAuthError = (error, navigate) => {
  if (error.response && (error.response.status === 401 || error.response.status === 403)) {
    // Clear token
    localStorage.removeItem('token');
    
    // Redirect to login
    navigate('/');
    
    return true;
  }
  return false;
};

// Log errors to console (could be extended to log to a service)
export const logError = (error, context = '') => {
  if (process.env.NODE_ENV !== 'production') {
    console.error(`Error ${context ? `in ${context}` : ''}:`, error);
  }
  
  // Here you could add integration with error logging services
  // like Sentry, LogRocket, etc.
};

// Create a standardized error object
export const createErrorObject = (error, context = '') => {
  const message = getErrorMessage(error);
  logError(error, context);
  
  return {
    message,
    originalError: error,
    context,
    timestamp: new Date().toISOString(),
  };
};

export default {
  getErrorMessage,
  handleAuthError,
  logError,
  createErrorObject
};
