import axios from 'axios';
import { handleApiResponse } from './apiUtils';

// Replace process.env with import.meta.env for Vite compatibility
const API_URL = import.meta.env.VITE_API_URL || '';

/**
 * LLM API service for interacting with the LLM API endpoints
 */
const llmApi = {
  /**
   * Get LLM status
   * @returns {Promise<Object>} LLM status
   */
  getStatus: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/status`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Get available models
   * @returns {Promise<Object>} Available models
   */
  getAvailableModels: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/models`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Generate text
   * @param {Object} requestData Request data
   * @returns {Promise<Object>} Generated text
   */
  generateText: async (requestData) => {
    try {
      const response = await axios.post(`${API_URL}/api/llm/generate`, requestData, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Get usage statistics
   * @param {string} startDate Start date (ISO format)
   * @param {string} endDate End date (ISO format)
   * @returns {Promise<Object>} Usage statistics
   */
  getUsageStatistics: async (startDate, endDate) => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/usage`, {
        params: {
          start_date: startDate,
          end_date: endDate
        },
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  // Gateway API
  /**
   * Get LLM Gateway providers
   * @returns {Promise<Object>} LLM Gateway providers
   */
  getProviders: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/gateway/providers`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Get LLM Gateway provider
   * @param {string} providerId Provider ID
   * @returns {Promise<Object>} LLM Gateway provider
   */
  getProvider: async (providerId) => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/gateway/providers/${providerId}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Update LLM Gateway provider
   * @param {string} providerId Provider ID
   * @param {Object} updateData Update data
   * @returns {Promise<Object>} Updated provider
   */
  updateProvider: async (providerId, updateData) => {
    try {
      const response = await axios.put(`${API_URL}/api/llm/gateway/providers/${providerId}`, updateData, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Test LLM Gateway provider
   * @param {string} providerId Provider ID
   * @returns {Promise<Object>} Test results
   */
  testProvider: async (providerId) => {
    try {
      const response = await axios.post(`${API_URL}/api/llm/gateway/providers/${providerId}/test`, {}, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Generate LLM response
   * @param {Object} requestData Request data
   * @returns {Promise<Object>} Generated response
   */
  generateLLMResponse: async (requestData) => {
    try {
      const response = await axios.post(`${API_URL}/api/llm/gateway/generate`, requestData, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Get LLM Gateway configuration
   * @returns {Promise<Object>} LLM Gateway configuration
   */
  getGatewayConfig: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/gateway/config`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Update LLM Gateway configuration
   * @param {Object} config Configuration
   * @returns {Promise<Object>} Updated configuration
   */
  updateGatewayConfig: async (config) => {
    try {
      const response = await axios.put(`${API_URL}/api/llm/gateway/config`, config, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  // DSPy API
  /**
   * Get DSPy modules
   * @returns {Promise<Object>} DSPy modules
   */
  getDspyModules: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/dspy/modules`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Get DSPy module
   * @param {string} moduleName Module name
   * @returns {Promise<Object>} DSPy module
   */
  getDspyModule: async (moduleName) => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/dspy/modules/${moduleName}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Register DSPy module
   * @param {string} moduleName Module name
   * @param {string} moduleType Module type
   * @param {Object} parameters Module parameters
   * @param {string} description Module description
   * @returns {Promise<Object>} Registered module
   */
  registerDspyModule: async (moduleName, moduleType, parameters, description) => {
    try {
      const response = await axios.post(`${API_URL}/api/llm/dspy/modules`, {
        module_name: moduleName,
        module_type: moduleType,
        parameters,
        description
      }, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Unregister DSPy module
   * @param {string} moduleName Module name
   * @returns {Promise<Object>} Success message
   */
  unregisterDspyModule: async (moduleName) => {
    try {
      const response = await axios.delete(`${API_URL}/api/llm/dspy/modules/${moduleName}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Execute DSPy module
   * @param {string} moduleName Module name
   * @param {Object} inputs Module inputs
   * @param {Object} config Module configuration
   * @returns {Promise<Object>} Execution results
   */
  executeDspyModule: async (moduleName, inputs, config) => {
    try {
      const response = await axios.post(`${API_URL}/api/llm/dspy/execute`, {
        module_name: moduleName,
        inputs,
        config
      }, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Optimize DSPy module
   * @param {string} moduleName Module name
   * @param {string} metric Optimization metric
   * @param {number} numTrials Number of optimization trials
   * @param {Array} examples Training examples
   * @param {Object} config Optimization configuration
   * @returns {Promise<Object>} Optimization results
   */
  optimizeDspyModule: async (moduleName, metric, numTrials, examples, config) => {
    try {
      const response = await axios.post(`${API_URL}/api/llm/dspy/optimize`, {
        module_name: moduleName,
        metric,
        num_trials: numTrials,
        examples,
        config
      }, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Get DSPy configuration
   * @returns {Promise<Object>} DSPy configuration
   */
  getDspyConfig: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/dspy/config`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Update DSPy configuration
   * @param {Object} config Configuration
   * @returns {Promise<Object>} Updated configuration
   */
  updateDspyConfig: async (config) => {
    try {
      const response = await axios.put(`${API_URL}/api/llm/dspy/config`, config, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  // BiomedLM API
  /**
   * Get BiomedLM models
   * @returns {Promise<Object>} BiomedLM models
   */
  getBiomedLMModels: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/biomedlm/models`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Get BiomedLM model
   * @param {string} modelId Model ID
   * @returns {Promise<Object>} BiomedLM model
   */
  getBiomedLMModel: async (modelId) => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/biomedlm/models/${modelId}`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Generate text with BiomedLM
   * @param {string} modelId Model ID
   * @param {string} prompt Prompt
   * @param {Object} params Generation parameters
   * @returns {Promise<Object>} Generated text
   */
  generateBiomedLMText: async (modelId, prompt, params = {}) => {
    try {
      const response = await axios.post(`${API_URL}/api/llm/biomedlm/generate`, {
        model_id: modelId,
        prompt,
        ...params
      }, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Fine-tune BiomedLM model
   * @param {string} modelId Model ID
   * @param {string} adapterName Adapter name
   * @param {string} task Task description
   * @param {Array|string} dataset Training dataset
   * @param {Object} params Fine-tuning parameters
   * @returns {Promise<Object>} Fine-tuning results
   */
  finetuneBiomedLMModel: async (modelId, adapterName, task, dataset, params = {}) => {
    try {
      const response = await axios.post(`${API_URL}/api/llm/biomedlm/finetune`, {
        model_id: modelId,
        adapter_name: adapterName,
        task,
        dataset,
        ...params
      }, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Get BiomedLM configuration
   * @returns {Promise<Object>} BiomedLM configuration
   */
  getBiomedLMConfig: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/llm/biomedlm/config`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  },

  /**
   * Update BiomedLM configuration
   * @param {Object} config Configuration
   * @returns {Promise<Object>} Updated configuration
   */
  updateBiomedLMConfig: async (config) => {
    try {
      const response = await axios.put(`${API_URL}/api/llm/biomedlm/config`, config, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`
        }
      });
      return handleApiResponse(response);
    } catch (error) {
      return handleApiResponse(error.response);
    }
  }
};

export default llmApi;
