import axios from 'axios';

// Create axios instance
const axiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication
axiosInstance.interceptors.request.use(
  (config) => {
    // Get token from localStorage
    const token = localStorage.getItem('token');
    
    // If token exists, add to headers
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
axiosInstance.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle authentication errors
    if (error.response && error.response.status === 401) {
      // Clear token and redirect to login
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    
    return Promise.reject(error);
  }
);

// API client
export const apiClient = {
  // Basic HTTP methods
  get: (url, config) => axiosInstance.get(url, config),
  post: (url, data, config) => axiosInstance.post(url, data, config),
  put: (url, data, config) => axiosInstance.put(url, data, config),
  patch: (url, data, config) => axiosInstance.patch(url, data, config),
  delete: (url, config) => axiosInstance.delete(url, config),
  
  // Auth endpoints
  auth: {
    login: (credentials) => axiosInstance.post('/api/login', credentials),
    logout: () => axiosInstance.post('/api/logout'),
    me: () => axiosInstance.get('/api/me'),
  },
  
  // LLM Gateway endpoints
  llm: {
    // Standard Gateway
    getProviders: () => axiosInstance.get('/api/llm/gateway/providers'),
    getModels: (providerId) => axiosInstance.get('/api/llm/gateway/models', { params: { provider_id: providerId } }),
    generateResponse: (requestData) => axiosInstance.post('/api/llm/gateway/generate', requestData),
    
    // Enhanced Gateway with Progress Tracking
    getEnhancedProviders: () => axiosInstance.get('/api/llm/enhanced-gateway/providers'),
    getEnhancedModels: (providerId) => axiosInstance.get('/api/llm/enhanced-gateway/models', { params: { provider_id: providerId } }),
    generateEnhancedResponse: (requestData) => axiosInstance.post('/api/llm/enhanced-gateway/generate', requestData),
    
    // Progress Tracking
    getActiveOperations: (operationType) => axiosInstance.get('/api/llm/progress/active', { params: { operation_type: operationType } }),
    getAllOperations: (params) => axiosInstance.get('/api/llm/progress/operations', { params }),
    getOperationDetails: (operationId) => axiosInstance.get(`/api/llm/progress/operations/${operationId}`),
    getProgressSummary: () => axiosInstance.get('/api/llm/progress/summary'),
    deleteOperation: (operationId) => axiosInstance.delete(`/api/llm/progress/operations/${operationId}`),
    cleanupOperations: (maxAgeSeconds = 3600) => axiosInstance.post('/api/llm/progress/cleanup', { max_age_seconds: maxAgeSeconds }),
  },
  
  // DSPy endpoints
  dspy: {
    getModules: () => axiosInstance.get('/api/llm/dspy/modules'),
    executeModule: (moduleData) => axiosInstance.post('/api/llm/dspy/execute', moduleData),
    optimizeModule: (moduleData) => axiosInstance.post('/api/llm/dspy/optimize', moduleData),
  },
  
  // BiomedLM endpoints
  biomedlm: {
    getModels: () => axiosInstance.get('/api/llm/biomedlm/models'),
    generateResponse: (requestData) => axiosInstance.post('/api/llm/biomedlm/generate', requestData),
    finetuneModel: (finetuneData) => axiosInstance.post('/api/llm/biomedlm/finetune', finetuneData),
  },
  
  // CL-PEFT endpoints
  clPeft: {
    getAdapters: () => axiosInstance.get('/api/llm/cl-peft/adapters'),
    getAdapter: (adapterId) => axiosInstance.get(`/api/llm/cl-peft/adapters/${adapterId}`),
    trainAdapter: (adapterId, trainingData) => axiosInstance.post(`/api/llm/cl-peft/adapters/${adapterId}/train`, trainingData),
    evaluateAdapter: (adapterId, evaluationData) => axiosInstance.post(`/api/llm/cl-peft/adapters/${adapterId}/evaluate`, evaluationData),
    generateWithAdapter: (adapterId, requestData) => axiosInstance.post(`/api/llm/cl-peft/adapters/${adapterId}/generate`, requestData),
    getStrategies: () => axiosInstance.get('/api/llm/cl-peft/strategies'),
    getPeftMethods: () => axiosInstance.get('/api/llm/cl-peft/peft-methods'),
    getBaseModels: () => axiosInstance.get('/api/llm/cl-peft/base-models'),
  },
  
  // MCP endpoints
  mcp: {
    getProviders: () => axiosInstance.get('/api/llm/mcp/providers'),
    getProvider: (providerId) => axiosInstance.get(`/api/llm/mcp/providers/${providerId}`),
    testProvider: (providerId) => axiosInstance.post(`/api/llm/mcp/providers/${providerId}/test`),
    getModels: (providerId) => axiosInstance.get(`/api/llm/mcp/providers/${providerId}/models`),
    generateResponse: (requestData) => axiosInstance.post('/api/llm/mcp/generate', requestData),
  },
};

export default apiClient;
