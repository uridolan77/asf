import { useState, useEffect, useMemo, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNotification } from '../context/NotificationContext';

/**
 * Custom hook for making API calls with caching and memoization
 * @param {string} endpoint - API endpoint
 * @param {Object} options - Options for the API call
 * @param {boolean} options.skip - Skip the API call
 * @param {Object} options.params - Query parameters
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

  // Fetch data function with caching
  const fetchData = useCallback(async (overrideParams = {}) => {
    try {
      setLoading(true);
      setError(null);

      const queryParams = { ...params, ...overrideParams };
      const queryKey = cacheKey || `${endpoint}-${JSON.stringify(queryParams)}`;

      // Check cache first if enabled
      if (cacheEnabled) {
        const cachedData = cache.get(queryKey);
        if (cachedData && isCacheValid(cachedData)) {
          setData(cachedData.data);
          setLoading(false);
          return cachedData.data;
        }
      }

      // Make the API call
      const response = await api.get(endpoint, { params: queryParams });
      const result = response.data;

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
        const errorMessage = err.response?.data?.message || 'An error occurred while fetching data';
        showError(errorMessage);
      }
      return null;
    } finally {
      setLoading(false);
    }
  }, [endpoint, params, api, cacheEnabled, cache, isCacheValid, cacheKey, onSuccess, onError, showError]);

  // Fetch on mount if not skipped
  useEffect(() => {
    if (!skip) {
      fetchData();
    }
  }, [skip, fetchData]);

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
    fetchData,
    postData,
    putData,
    deleteData,
    clearCache,
    updateData
  };
};

export default useApi;
