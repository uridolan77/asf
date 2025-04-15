import axios from 'axios';
import { getErrorMessage, logError, handleAuthError } from '../utils/errorHandler';

// Create axios instance with default config
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
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
 */
export const apiCall = async (method, url, data = null, options = {}) => {
  try {
    const response = await api({
      method,
      url,
      data: method !== 'get' ? data : null,
      params: method === 'get' ? data : null,
      ...options,
    });

    return {
      success: true,
      data: response.data,
      status: response.status,
    };
  } catch (error) {
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
    contradictions: (params) => apiCall('post', '/api/medical/analysis/contradictions', params),
    cap: () => apiCall('get', '/api/medical/analysis/cap'),
    getById: (id) => apiCall('get', `/api/medical/analysis/${id}`),
    getHistory: (params) => apiCall('get', '/api/medical/analysis/history', params),
  },

  // Export endpoints
  export: {
    toFormat: (format, params) => apiCall('post', `/api/medical/export/${format}`, params),
    getStatus: (taskId) => apiCall('get', `/api/medical/export/status/${taskId}`),
    download: (taskId) => apiCall('get', `/api/medical/export/download/${taskId}`),
  },

  // ML service endpoints
  ml: {
    // Contradiction detection
    detectContradiction: (params) => apiCall('post', '/api/medical/ml/contradiction', params),
    detectContradictionsBatch: (params) => apiCall('post', '/api/medical/ml/contradiction/batch', params),

    // Temporal analysis
    calculateTemporalConfidence: (params) => apiCall('post', '/api/medical/ml/temporal/confidence', params),
    detectTemporalContradiction: (params) => apiCall('post', '/api/medical/ml/temporal/contradiction', params),

    // Bias assessment
    assessBias: (params) => apiCall('post', '/api/medical/ml/bias/assess', params),
    getBiasAssessmentTools: () => apiCall('get', '/api/medical/ml/bias/tools'),
  },

  // Knowledge base endpoints
  knowledgeBase: {
    getAll: () => apiCall('get', '/api/medical/knowledge-base'),
    getById: (id) => apiCall('get', `/api/medical/knowledge-base/${id}`),
    create: (data) => apiCall('post', '/api/medical/knowledge-base', data),
    update: (id) => apiCall('post', `/api/medical/knowledge-base/${id}/update`),
    delete: (id) => apiCall('delete', `/api/medical/knowledge-base/${id}`),
  },
};

export default apiService;
