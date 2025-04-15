import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { handleAuthError } from '../utils/errorHandler';

/**
 * Custom hook for handling API calls with loading and error states
 * 
 * @param {Function} apiFunction - The API function to call
 * @param {Object} options - Options for the hook
 * @param {boolean} options.loadOnMount - Whether to load data on mount
 * @param {Array} options.dependencies - Dependencies for the useEffect hook
 * @param {Function} options.onSuccess - Callback for successful API calls
 * @param {Function} options.onError - Callback for failed API calls
 * @param {any} options.initialData - Initial data
 * @returns {Object} - { data, loading, error, execute, reset }
 */
const useApi = (apiFunction, options = {}) => {
  const {
    loadOnMount = false,
    dependencies = [],
    onSuccess,
    onError,
    initialData = null,
  } = options;

  const [data, setData] = useState(initialData);
  const [loading, setLoading] = useState(loadOnMount);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Execute the API call
  const execute = useCallback(async (...args) => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiFunction(...args);
      
      if (result.success) {
        setData(result.data);
        if (onSuccess) onSuccess(result.data);
        return { success: true, data: result.data };
      } else {
        // Handle authentication errors
        if (result.isAuthError) {
          handleAuthError(result.originalError, navigate);
        }
        
        setError(result.error);
        if (onError) onError(result.error);
        return { success: false, error: result.error };
      }
    } catch (err) {
      const errorMessage = err.message || 'An unexpected error occurred';
      setError(errorMessage);
      if (onError) onError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setLoading(false);
    }
  }, [apiFunction, navigate, onSuccess, onError]);

  // Reset the state
  const reset = useCallback(() => {
    setData(initialData);
    setLoading(false);
    setError(null);
  }, [initialData]);

  // Load data on mount if specified
  useEffect(() => {
    if (loadOnMount) {
      execute();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);

  return { data, loading, error, execute, reset };
};

export default useApi;
