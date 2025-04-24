// API service for making requests to the backend

import axios from 'axios';

// Create axios instance with base URL
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle session expiration
    if (error.response && error.response.status === 401) {
      // Redirect to login page if not already there
      if (!window.location.pathname.includes('/login')) {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Helper function to handle API responses
const handleResponse = (response) => {
  return {
    success: true,
    data: response.data,
  };
};

// Helper function to handle API errors
const handleError = (error) => {
  let errorMessage = 'An unexpected error occurred';
  
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    if (error.response.data && error.response.data.detail) {
      errorMessage = error.response.data.detail;
    } else if (error.response.data && error.response.data.message) {
      errorMessage = error.response.data.message;
    } else {
      errorMessage = `Server error: ${error.response.status}`;
    }
  } else if (error.request) {
    // The request was made but no response was received
    errorMessage = 'No response received from server';
  } else {
    // Something happened in setting up the request that triggered an Error
    errorMessage = error.message;
  }
  
  return {
    success: false,
    error: errorMessage,
  };
};

// API service object
const apiService = {
  // Auth endpoints
  auth: {
    login: async (username, password) => {
      try {
        const response = await api.post('/auth/token', {
          username,
          password,
        });
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    register: async (userData) => {
      try {
        const response = await api.post('/auth/register', userData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    getCurrentUser: async () => {
      try {
        const response = await api.get('/auth/me');
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    updateProfile: async (userData) => {
      try {
        const response = await api.put('/auth/me', userData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    changePassword: async (oldPassword, newPassword) => {
      try {
        const response = await api.post('/auth/change-password', {
          old_password: oldPassword,
          new_password: newPassword,
        });
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
  },
  
  // LLM endpoints
  llm: {
    // Provider management
    getProviders: async () => {
      try {
        const response = await api.get('/api/llm/gateway/providers');
        const result = handleResponse(response);
        
        // Ensure the data is an array
        if (result.success && !Array.isArray(result.data)) {
          result.data = [];
        }
        
        return result;
      } catch (error) {
        return handleError(error);
      }
    },
    
    getProvider: async (providerId) => {
      try {
        const response = await api.get(`/api/llm/gateway/providers/${providerId}`);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    createProvider: async (providerData) => {
      try {
        const response = await api.post('/api/llm/gateway/providers', providerData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    updateProvider: async (providerId, providerData) => {
      try {
        const response = await api.put(`/api/llm/gateway/providers/${providerId}`, providerData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    deleteProvider: async (providerId) => {
      try {
        const response = await api.delete(`/api/llm/gateway/providers/${providerId}`);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    // Model management
    getModels: async (providerId) => {
      try {
        const response = await api.get('/api/llm/gateway/models', {
          params: { provider_id: providerId }
        });
        const result = handleResponse(response);
        
        // Ensure the data is an array
        if (result.success && !Array.isArray(result.data)) {
          result.data = [];
        }
        
        return result;
      } catch (error) {
        return handleError(error);
      }
    },
    
    getModel: async (modelId, providerId) => {
      try {
        const response = await api.get(`/api/llm/gateway/models/${providerId}/${modelId}`);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    createModel: async (modelData) => {
      try {
        const response = await api.post('/api/llm/gateway/models', modelData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    updateModel: async (modelId, providerId, modelData) => {
      try {
        const response = await api.put(`/api/llm/gateway/models/${providerId}/${modelId}`, modelData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    deleteModel: async (modelId, providerId) => {
      try {
        const response = await api.delete(`/api/llm/gateway/models/${providerId}/${modelId}`);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    // LLM operations
    generateText: async (prompt, model, params = {}) => {
      try {
        const response = await api.post('/api/llm/gateway/generate', {
          prompt,
          model,
          ...params,
        });
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    generateLLMResponse: async (requestData) => {
      try {
        const response = await api.post('/api/llm/gateway/generate', requestData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    chat: async (messages, model, params = {}) => {
      try {
        const response = await api.post('/api/llm/gateway/chat', {
          messages,
          model,
          ...params,
        });
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    // Service Abstraction Layer management
    getServiceConfig: async () => {
      try {
        const response = await api.get('/api/llm/service/config');
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    updateServiceConfig: async (config) => {
      try {
        const response = await api.put('/api/llm/service/config', config);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    getServiceHealth: async () => {
      try {
        const response = await api.get('/api/llm/service/health');
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    getServiceStats: async () => {
      try {
        const response = await api.get('/api/llm/service/stats');
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    clearCache: async () => {
      try {
        const response = await api.post('/api/llm/service/cache/clear');
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    resetCircuitBreakers: async () => {
      try {
        const response = await api.post('/api/llm/service/resilience/reset-circuit-breakers');
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    // Service Configuration Management
    getServiceConfigurations: async (includePublic = true, skip = 0, limit = 100) => {
      try {
        const response = await api.get('/api/llm/service/configurations', {
          params: { include_public: includePublic, skip, limit }
        });
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    getServiceConfigurationById: async (configId) => {
      try {
        const response = await api.get(`/api/llm/service/configurations/${configId}`);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    createServiceConfiguration: async (configData) => {
      try {
        const response = await api.post('/api/llm/service/configurations', configData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    updateServiceConfiguration: async (configId, configData) => {
      try {
        const response = await api.put(`/api/llm/service/configurations/${configId}`, configData);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    deleteServiceConfiguration: async (configId) => {
      try {
        const response = await api.delete(`/api/llm/service/configurations/${configId}`);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
    
    applyServiceConfiguration: async (configId) => {
      try {
        const response = await api.post(`/api/llm/service/configurations/${configId}/apply`);
        return handleResponse(response);
      } catch (error) {
        return handleError(error);
      }
    },
  },
  
  // Other API endpoints can be added here
};

export default apiService;
