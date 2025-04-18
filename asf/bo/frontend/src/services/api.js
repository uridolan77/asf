import axios from 'axios';
import { getErrorMessage, logError } from '../utils/errorHandler';

// Create axios instance with default config
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  withCredentials: false,
});

// Log requests
api.interceptors.request.use(request => {
  console.log('Starting Request', request);
  return request;
});

// Log responses
api.interceptors.response.use(response => {
  console.log('Response:', response);
  return response;
}, error => {
  console.error('Response Error:', error);
  return Promise.reject(error);
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    logError(error, 'API Request');
    return Promise.reject(error);
  }
);

/**
 * Wrapper for API calls with consistent error handling
 * @param {string} method - HTTP method (get, post, put, delete)
 * @param {string} url - API endpoint URL
 * @param {object} data - Request data or query parameters
 * @param {object} options - Additional axios options
 * @param {AbortSignal} [signal] - AbortController signal for cancellation
 * @returns {object} - Response object with success flag and data/error
 */
export const apiCall = async (method, url, data = null, options = {}, signal = null) => {
  try {
    const response = await api({
      method,
      url,
      data: method !== 'get' ? data : null,
      params: method === 'get' ? data : null,
      signal: signal, // Pass the AbortController signal
      ...options,
    });

    return {
      success: true,
      data: response.data,
      status: response.status,
    };
  } catch (error) {
    // Check if the request was cancelled
    if (error.name === 'CanceledError' || error.name === 'AbortError') {
      console.log('Request was cancelled', { method, url });
      return {
        success: false,
        error: 'Request was cancelled',
        status: 'cancelled',
        isCancelled: true,
      };
    }

    // Check if it's an auth error and handle navigation in the component
    const isAuthError = error.response && (error.response.status === 401 || error.response.status === 403);

    return {
      success: false,
      error: getErrorMessage(error),
      status: error.response?.status,
      isAuthError,
      originalError: error,
    };
  }
};

// LLM API service is integrated directly

// API service methods
const apiService = {
  // Auth endpoints
  auth: {
    login: (credentials) => apiCall('post', '/api/login', credentials),
    logout: () => apiCall('post', '/api/logout'),
    me: () => apiCall('get', '/api/me'),
  },

  // User endpoints
  users: {
    getAll: (params) => apiCall('get', '/api/users', params),
    getById: (id) => apiCall('get', `/api/users/${id}`),
    create: (userData) => apiCall('post', '/api/users', userData),
    update: (id, userData) => apiCall('put', `/api/users/${id}`, userData),
    delete: (id) => apiCall('delete', `/api/users/${id}`),
  },

  // Medical search endpoints
  medical: {
    picoSearch: (params) => apiCall('post', '/api/medical/search/pico', params),
    advancedSearch: (params) => apiCall('post', '/api/medical/search', params),
    terminologySearch: (query) => apiCall('get', '/api/medical/terminology/search', { query }),
  },

  // Analysis endpoints
  analysis: {
    contradictions: (params) => {
      // Try both endpoints with fallback
      return apiCall('post', '/api/medical/analysis/contradictions', params)
        .catch(error => {
          console.warn('Primary analysis endpoint failed, trying fallback', error);
          return apiCall('post', '/v1/analysis/contradictions', params);
        });
    },
    cap: () => {
      // Try both endpoints with fallback
      return apiCall('get', '/api/medical/analysis/cap')
        .catch(error => {
          console.warn('Primary CAP endpoint failed, trying fallback', error);
          return apiCall('get', '/v1/analysis/cap');
        });
    },
    getById: (id) => {
      // Try both endpoints with fallback
      return apiCall('get', `/api/medical/analysis/${id}`)
        .catch(error => {
          console.warn('Primary analysis retrieval endpoint failed, trying fallback', error);
          return apiCall('get', `/v1/analysis/${id}`);
        });
    },
    getHistory: (params) => {
      // Try both endpoints with fallback
      return apiCall('get', '/api/medical/analysis/history', params)
        .catch(error => {
          console.warn('Primary analysis history endpoint failed, trying fallback', error);
          return apiCall('get', '/v1/analysis/history', params);
        });
    },
  },

  // Export endpoints
  export: {
    toFormat: (format, params) => apiCall('post', `/api/medical/export/${format}`, params),
    getStatus: (taskId) => apiCall('get', `/api/medical/export/status/${taskId}`),
    download: (taskId) => apiCall('get', `/api/medical/export/download/${taskId}`),
  },

  // ML service endpoints
  ml: {
    // Get ML services status with better error handling and fallbacks
    getServicesStatus: async () => {
      try {
        // First try the ML API endpoint
        const mlResponse = await apiCall('get', '/api/ml/services/status');
        if (mlResponse.success) {
          return mlResponse;
        }

        // If that fails, try the medical ML endpoint
        const medicalMlResponse = await apiCall('get', '/api/medical/ml/services/status');
        if (medicalMlResponse.success) {
          return medicalMlResponse;
        }

        // If both fail, return a default response to prevent UI from getting stuck
        console.warn("Both ML service status endpoints failed, returning default response");
        return {
          success: true,
          data: {
            services: [
              {
                name: "Contradiction Detector",
                status: "unknown",
                version: "unknown",
                description: "Service status could not be determined",
                last_updated: new Date().toISOString().split('T')[0],
                health: "unknown"
              }
            ]
          }
        };
      } catch (error) {
        // Return a default response even if everything fails
        console.error("Error fetching ML services status:", error);
        return {
          success: true,
          data: {
            services: [
              {
                name: "ML Services",
                status: "error",
                version: "unknown",
                description: "Failed to connect to ML services",
                last_updated: new Date().toISOString().split('T')[0],
                health: "error"
              }
            ]
          }
        };
      }
    },

    // Get ML services metrics with better error handling and fallbacks
    getServicesMetrics: async () => {
      try {
        // First try the ML API endpoint
        const mlResponse = await apiCall('get', '/api/ml/services/metrics');
        if (mlResponse.success) {
          return mlResponse;
        }

        // If that fails, try the medical ML endpoint
        const medicalMlResponse = await apiCall('get', '/api/medical/ml/services/metrics');
        if (medicalMlResponse.success) {
          return medicalMlResponse;
        }

        // If both fail, return a default response to prevent UI from getting stuck
        console.warn("Both ML service metrics endpoints failed, returning default response");
        return {
          success: true,
          data: {
            status: "unavailable",
            period: "current",
            services: {}
          }
        };
      } catch (error) {
        // Return a default response even if everything fails
        console.error("Error fetching ML services metrics:", error);
        return {
          success: true,
          data: {
            status: "error",
            period: "unavailable",
            services: {}
          }
        };
      }
    },

    // Contradiction detection with signal support for cancellation
    detectContradiction: (params, signal = null) =>
      apiCall('post', '/api/medical/ml/contradiction', params, {}, signal),

    // Batch contradiction detection with signal support
    detectContradictionsBatch: (params, signal = null) =>
      apiCall('post', '/api/medical/ml/contradiction/batch', params, {}, signal),

    // Temporal analysis with signal support
    calculateTemporalConfidence: (params, signal = null) =>
      apiCall('post', '/api/medical/ml/temporal/confidence', params, {}, signal),

    detectTemporalContradiction: (params, signal = null) =>
      apiCall('post', '/api/medical/ml/temporal/contradiction', params, {}, signal),

    // Bias assessment with signal support
    assessBias: (params, signal = null) =>
      apiCall('post', '/api/medical/ml/bias/assess', params, {}, signal),

    getBiasAssessmentTools: () =>
      apiCall('get', '/api/medical/ml/bias/tools'),
  },

  // Document processing endpoints
  documentProcessing: {
    // Process a single document
    processSingle: (formData) => apiCall('post', '/api/document-processing/process', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }),

    // Process multiple documents in batch
    processBatch: (formData) => apiCall('post', '/api/document-processing/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }),

    // Get all processing tasks
    getTasks: (status, limit, offset) => apiCall('get', '/api/document-processing/tasks', { status, limit, offset }),

    // Get a specific processing task
    getTask: (taskId) => apiCall('get', `/api/document-processing/tasks/${taskId}`),

    // Get processing results
    getResults: (taskId) => apiCall('get', `/api/document-processing/results/${taskId}`),

    // Get processing progress
    getProgress: (taskId) => apiCall('get', `/api/document-processing/tasks/${taskId}/progress`),

    // Get default processing settings
    getSettings: () => apiCall('get', '/api/document-processing/settings'),

    // Poll for processing progress
    pollProgress: async (taskId, callback, interval = 1000, maxAttempts = 300) => {
      let attempts = 0;
      const poll = async () => {
        attempts++;
        try {
          const response = await apiService.documentProcessing.getProgress(taskId);

          if (response.success) {
            // Call the callback with the progress data
            callback(response.data);

            // If processing is complete or failed, stop polling
            if (['completed', 'failed'].includes(response.data.status)) {
              return response.data;
            }

            // Continue polling if we haven't reached max attempts
            if (attempts < maxAttempts) {
              setTimeout(poll, interval);
            }
          } else {
            console.error('Error polling progress:', response.error);
            return null;
          }
        } catch (error) {
          console.error('Error polling progress:', error);
          return null;
        }
      };

      // Start polling
      return poll();
    },
  },

  // Medical clients endpoints
  clients: {
    // Get all clients
    getAll: () => apiCall('get', '/api/medical/clients'),

    // Get specific client
    getClient: (clientId) => apiCall('get', `/api/medical/clients/${clientId}`),

    // Update client configuration
    updateClient: (clientId, config) => apiCall('put', `/api/medical/clients/${clientId}`, { config }),

    // Get client usage statistics
    getClientUsage: (clientId) => apiCall('get', `/api/medical/clients/${clientId}/usage`),

    // Test client connection
    testConnection: (clientId) => apiCall('post', `/api/medical/clients/${clientId}/test`),

    // NCBI specific endpoints
    ncbi: {
      // Search PubMed
      searchPubMed: (query, maxResults = 10) => apiCall('post', '/api/medical/clients/ncbi/search', { query, max_results: maxResults }),

      // Get article details
      getArticle: (pmid) => apiCall('get', `/api/medical/clients/ncbi/article/${pmid}`),

      // Get citation data
      getCitations: (pmid) => apiCall('get', `/api/medical/clients/ncbi/citations/${pmid}`),
    },
  },

  // Knowledge base endpoints
  knowledgeBase: {
    getAll: () => apiCall('get', '/api/medical/knowledge-base'),
    getById: (id) => apiCall('get', `/api/medical/knowledge-base/${id}`),
    create: (data) => apiCall('post', '/api/medical/knowledge-base', data),
    update: (id) => apiCall('post', `/api/medical/knowledge-base/${id}/update`),
    delete: (id) => apiCall('delete', `/api/medical/knowledge-base/${id}`),
  },

  // LLM API endpoints
  llm: {
    // Main LLM endpoints
    getStatus: () => apiCall('get', '/api/llm/status'),
    getAvailableModels: () => apiCall('get', '/api/llm/models'),
    generateText: (requestData) => apiCall('post', '/api/llm/generate', requestData),
    getUsageStatistics: (startDate, endDate) => apiCall('get', '/api/llm/usage', { start_date: startDate, end_date: endDate }),

    // Gateway endpoints
    getProviders: () => apiCall('get', '/api/llm/gateway/providers'),
    getProvider: (providerId) => apiCall('get', `/api/llm/gateway/providers/${providerId}`),
    updateProvider: (providerId, updateData) => apiCall('put', `/api/llm/gateway/providers/${providerId}`, updateData),
    registerProvider: (providerData) => apiCall('post', '/api/llm/gateway/providers', providerData),
    deleteProvider: (providerId) => apiCall('delete', `/api/llm/gateway/providers/${providerId}`),
    testProvider: (providerId) => apiCall('post', `/api/llm/gateway/providers/${providerId}/test`),

    // Model management endpoints
    getModels: (providerId = null, modelType = null) => apiCall('get', '/api/llm/gateway/models', { provider_id: providerId, model_type: modelType }),
    getModel: (modelId, providerId) => apiCall('get', `/api/llm/gateway/models/${modelId}`, { provider_id: providerId }),
    createModel: (modelData) => apiCall('post', '/api/llm/gateway/models', modelData),
    updateModel: (modelId, providerId, updateData) => apiCall('put', `/api/llm/gateway/models/${modelId}`, { ...updateData, provider_id: providerId }),
    deleteModel: (modelId, providerId) => apiCall('delete', `/api/llm/gateway/models/${modelId}`, { provider_id: providerId }),

    generateLLMResponse: (requestData) => apiCall('post', '/api/llm/gateway/generate', requestData),

    // DSPy endpoints
    getDspyModules: () => apiCall('get', '/api/dspy/modules'),
    getDspyModule: (moduleName) => apiCall('get', `/api/dspy/modules/${moduleName}`),
    registerDspyModule: (moduleName, moduleType, parameters, description) => apiCall('post', '/api/dspy/modules', {
      module_name: moduleName,
      module_type: moduleType,
      parameters,
      description
    }),
    unregisterDspyModule: (moduleName) => apiCall('delete', `/api/dspy/modules/${moduleName}`),
    executeDspyModule: (moduleName, inputs, config) => apiCall('post', '/api/dspy/execute', {
      module_name: moduleName,
      inputs,
      config
    }),
    optimizeDspyModule: (moduleName, metric, numTrials, examples, config) => apiCall('post', '/api/dspy/optimize', {
      module_name: moduleName,
      metric,
      num_trials: numTrials,
      examples,
      config
    }),
    getDspyConfig: () => apiCall('get', '/api/dspy/config'),
    updateDspyConfig: (config) => apiCall('put', '/api/dspy/config', config),

    // BiomedLM endpoints
    getBiomedLMModels: () => apiCall('get', '/api/llm/biomedlm/models'),
    getBiomedLMModel: (modelId) => apiCall('get', `/api/llm/biomedlm/models/${modelId}`),
    generateBiomedLMText: (modelId, prompt, params = {}) => apiCall('post', '/api/llm/biomedlm/generate', {
      model_id: modelId,
      prompt,
      ...params
    }),
    finetuneBiomedLMModel: (modelId, adapterName, task, dataset, params = {}) => apiCall('post', '/api/llm/biomedlm/finetune', {
      model_id: modelId,
      adapter_name: adapterName,
      task,
      dataset,
      ...params
    }),
    getBiomedLMConfig: () => apiCall('get', '/api/llm/biomedlm/config'),
    updateBiomedLMConfig: (config) => apiCall('put', '/api/llm/biomedlm/config', config),

    // MCP (Model Context Protocol) endpoints
    getMCPInfo: () => apiCall('get', '/api/llm/mcp/info'),
    getMCPProviders: () => apiCall('get', '/api/llm/mcp/providers'),
    getMCPProviderStatus: (providerId) => apiCall('get', `/api/llm/mcp/providers/${providerId}/status`),
    getMCPProviderUsage: (providerId, period = 'day') => apiCall('get', `/api/llm/mcp/providers/${providerId}/usage`, { period }),
    testMCPProvider: (providerId) => apiCall('post', `/api/llm/mcp/providers/${providerId}/test`),
    registerMCPProvider: (config) => apiCall('post', '/api/llm/mcp/providers', config),
    updateMCPProvider: (providerId, config) => apiCall('put', `/api/llm/mcp/providers/${providerId}`, config),
    deleteMCPProvider: (providerId) => apiCall('delete', `/api/llm/mcp/providers/${providerId}`),
    generateWithMCP: (request) => apiCall('post', '/api/llm/mcp/generate', request),
  },
};

export default apiService;
