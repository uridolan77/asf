import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useAuth } from '../context/AuthContext.jsx';
import { useNotification } from '../context/NotificationContext.jsx';

/**
 * Custom hook for making API calls with caching and memoization
 * @param {string} endpoint - API endpoint
 * @param {Object} options - Options for the API call
 * @param {boolean} options.skip - Skip the API call
 * @param {Object} options.params - Query parameters
 * @param {boolean} options.loadOnMount - Execute API call on component mount
 * @param {Object} options.initialData - Initial data
 * @param {boolean} options.cacheEnabled - Enable request caching
 * @param {number} options.cacheDuration - Cache duration in milliseconds
 * @param {string} options.cacheKey - Custom cache key
 * @param {function} options.onSuccess - Success callback
 * @param {function} options.onError - Error callback
 * @returns {Object} - API call state and methods
 */
const useApi = (
  endpoint,
  {
    skip = false,
    params = {},
    loadOnMount = false,
    initialData = null,
    cacheEnabled = false,
    cacheDuration = 5 * 60 * 1000, // 5 minutes default
    cacheKey = '',
    onSuccess = null,
    onError = null
  } = {}
) => {
  const [data, setData] = useState(initialData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const { api } = useAuth();
  const { showError } = useNotification();

  // Cache storage
  const cache = useMemo(() => {
    return new Map();
  }, []);

  // Generate a unique cache key
  const generatedCacheKey = useMemo(() => {
    const baseKey = cacheKey || endpoint;
    const paramsKey = Object.keys(params).length
      ? `-${JSON.stringify(params)}`
      : '';
    return `${baseKey}${paramsKey}`;
  }, [endpoint, params, cacheKey]);

  // Check if the cache is valid
  const isCacheValid = useCallback((cachedData) => {
    if (!cachedData) return false;
    const now = Date.now();
    return now - cachedData.timestamp < cacheDuration;
  }, [cacheDuration]);

  // Execute API function with caching
  const execute = useCallback(async (...args) => {
    try {
      setLoading(true);
      setError(null);

      // Generate a cache key based on the endpoint and arguments
      const argsKey = args.length > 0 ? `-${JSON.stringify(args)}` : '';
      const queryKey = cacheKey || `${endpoint}${argsKey}`;

      // Check cache first if enabled
      if (cacheEnabled) {
        const cachedData = cache.get(queryKey);
        if (cachedData && isCacheValid(cachedData)) {
          setData(cachedData.data);
          setLoading(false);
          return cachedData.data;
        }
      }

      // Execute the API function (endpoint is the actual function)
      const response = await endpoint(...args);

      // Handle both direct responses and responses wrapped in {success, data} format
      const result = response.success !== undefined ? response.data : response;

      // Store in cache if enabled
      if (cacheEnabled) {
        cache.set(queryKey, {
          data: result,
          timestamp: Date.now()
        });
      }

      setData(result);
      if (onSuccess) onSuccess(result);
      return result;
    } catch (err) {
      setError(err);
      if (onError) {
        onError(err);
      } else {
        const errorMessage = err.response?.data?.message || err.message || 'An error occurred';
        showError(errorMessage);
      }
      return null;
    } finally {
      setLoading(false);
    }
  }, [endpoint, cacheEnabled, cache, isCacheValid, cacheKey, onSuccess, onError, showError]);

  // Execute on mount if not skipped and loadOnMount is true
  // Using a ref to track if we've already executed on mount
  const hasExecutedOnMount = useRef(false);

  useEffect(() => {
    // Only execute if loadOnMount is true, not skipped, and we haven't executed yet
    if (!skip && loadOnMount && !hasExecutedOnMount.current) {
      hasExecutedOnMount.current = true;
      execute();
    }

    // Reset the ref when skip or loadOnMount changes
    if (skip || !loadOnMount) {
      hasExecutedOnMount.current = false;
    }
  }, [skip, loadOnMount, execute]);

  // Clear cache for this endpoint
  const clearCache = useCallback(() => {
    if (cacheKey) {
      cache.delete(cacheKey);
    } else {
      // Clear all caches for this endpoint
      for (const key of cache.keys()) {
        if (key.startsWith(endpoint)) {
          cache.delete(key);
        }
      }
    }
  }, [cache, endpoint, cacheKey]);

  // Update data
  const updateData = useCallback((newData) => {
    setData(newData);

    if (cacheEnabled) {
      const cachedData = cache.get(generatedCacheKey);
      if (cachedData) {
        cache.set(generatedCacheKey, {
          ...cachedData,
          data: newData
        });
      }
    }
  }, [cache, cacheEnabled, generatedCacheKey]);

  // Post data
  const postData = useCallback(async (postEndpoint, payload, config = {}) => {
    try {
      setLoading(true);
      const response = await api.post(postEndpoint || endpoint, payload, config);

      if (cacheEnabled) {
        clearCache();
      }

      if (config.updateLocalData !== false) {
        setData(prevData => {
          // If previous data is an array, add the new item
          if (Array.isArray(prevData)) {
            return [...prevData, response.data];
          }
          return response.data;
        });
      }

      if (config.onSuccess) {
        config.onSuccess(response.data);
      }

      return response.data;
    } catch (err) {
      if (config.onError) {
        config.onError(err);
      } else {
        const errorMessage = err.response?.data?.message || 'An error occurred';
        showError(errorMessage);
      }
      setError(err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [api, endpoint, cacheEnabled, clearCache, showError]);

  // Put data
  const putData = useCallback(async (id, payload, config = {}) => {
    try {
      setLoading(true);
      const putEndpoint = `${endpoint}/${id}`;
      const response = await api.put(putEndpoint, payload);

      if (cacheEnabled) {
        clearCache();
      }

      if (config.updateLocalData !== false) {
        setData(prevData => {
          // If previous data is an array, update the matching item
          if (Array.isArray(prevData)) {
            return prevData.map(item =>
              item.id === id ? { ...item, ...response.data } : item
            );
          }
          return response.data;
        });
      }

      if (config.onSuccess) {
        config.onSuccess(response.data);
      }

      return response.data;
    } catch (err) {
      if (config.onError) {
        config.onError(err);
      } else {
        const errorMessage = err.response?.data?.message || 'An error occurred';
        showError(errorMessage);
      }
      setError(err);
      return null;
    } finally {
      setLoading(false);
    }
  }, [api, endpoint, cacheEnabled, clearCache, showError]);

  // Delete data
  const deleteData = useCallback(async (id, config = {}) => {
    try {
      setLoading(true);
      const deleteEndpoint = `${endpoint}/${id}`;
      await api.delete(deleteEndpoint);

      if (cacheEnabled) {
        clearCache();
      }

      if (config.updateLocalData !== false) {
        setData(prevData => {
          // If previous data is an array, remove the deleted item
          if (Array.isArray(prevData)) {
            return prevData.filter(item => item.id !== id);
          }
          return null;
        });
      }

      if (config.onSuccess) {
        config.onSuccess();
      }

      return true;
    } catch (err) {
      if (config.onError) {
        config.onError(err);
      } else {
        const errorMessage = err.response?.data?.message || 'An error occurred';
        showError(errorMessage);
      }
      setError(err);
      return false;
    } finally {
      setLoading(false);
    }
  }, [api, endpoint, cacheEnabled, clearCache, showError]);

  return {
    data,
    loading,
    error,
    execute,
    postData,
    putData,
    deleteData,
    clearCache,
    updateData
  };
};

export default useApi;
