import axios from 'axios';
import { API_BASE_URL } from '../config';
import {
  generateMockAdapters,
  generateMockAdapter,
  generateMockEvaluationResults,
  generateMockForgettingResults
} from '../components/llm/cl_peft/mockData';

// Set to true to use mock data instead of API calls
const USE_MOCK_DATA = true;

const API_URL = `${API_BASE_URL}/llm/cl-peft`;

/**
 * Fetch all adapters, optionally filtered
 * @param {Object} filters - Optional filters
 * @returns {Promise<Array>} - List of adapters
 */
export const fetchAdapters = async (filters = {}) => {
  try {
    if (USE_MOCK_DATA) {
      // Generate mock data
      console.log('Using mock data for fetchAdapters');
      const mockAdapters = generateMockAdapters(8);

      // Apply filters
      let filteredAdapters = [...mockAdapters];
      if (filters.cl_strategy) {
        filteredAdapters = filteredAdapters.filter(a => a.cl_strategy === filters.cl_strategy);
      }
      if (filters.peft_method) {
        filteredAdapters = filteredAdapters.filter(a => a.peft_method === filters.peft_method);
      }
      if (filters.status) {
        filteredAdapters = filteredAdapters.filter(a => a.status === filters.status);
      }
      if (filters.tag) {
        filteredAdapters = filteredAdapters.filter(a => a.tags.includes(filters.tag));
      }

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 500));

      return filteredAdapters;
    }

    // Real API call
    const params = {};
    if (filters.cl_strategy) params.cl_strategy = filters.cl_strategy;
    if (filters.peft_method) params.peft_method = filters.peft_method;
    if (filters.status) params.status = filters.status;
    if (filters.tag) params.tag = filters.tag;

    const response = await axios.get(API_URL + '/adapters', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching adapters:', error);
    throw error;
  }
};

/**
 * Fetch a specific adapter by ID
 * @param {string} adapterId - Adapter ID
 * @returns {Promise<Object>} - Adapter details
 */
export const fetchAdapter = async (adapterId) => {
  try {
    if (USE_MOCK_DATA) {
      // Generate mock data
      console.log(`Using mock data for fetchAdapter: ${adapterId}`);
      const mockAdapter = generateMockAdapter({ adapter_id: adapterId });

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 300));

      return mockAdapter;
    }

    // Real API call
    const response = await axios.get(`${API_URL}/adapters/${adapterId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching adapter ${adapterId}:`, error);
    throw error;
  }
};

/**
 * Create a new adapter
 * @param {Object} adapterData - Adapter configuration
 * @returns {Promise<Object>} - Created adapter
 */
export const createAdapter = async (adapterData) => {
  try {
    if (USE_MOCK_DATA) {
      // Generate mock data
      console.log('Using mock data for createAdapter');
      const mockAdapter = generateMockAdapter({
        adapter_name: adapterData.adapter_name,
        base_model_name: adapterData.base_model_name,
        description: adapterData.description,
        cl_strategy: adapterData.cl_strategy,
        peft_method: adapterData.peft_method,
        status: 'initializing',
        tags: adapterData.tags || []
      });

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      return mockAdapter;
    }

    // Real API call
    const response = await axios.post(`${API_URL}/adapters`, adapterData);
    return response.data;
  } catch (error) {
    console.error('Error creating adapter:', error);
    throw error;
  }
};

/**
 * Delete an adapter
 * @param {string} adapterId - Adapter ID
 * @returns {Promise<Object>} - Response
 */
export const deleteAdapter = async (adapterId) => {
  try {
    if (USE_MOCK_DATA) {
      // Generate mock data
      console.log(`Using mock data for deleteAdapter: ${adapterId}`);

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 500));

      return { success: true };
    }

    // Real API call
    const response = await axios.delete(`${API_URL}/adapters/${adapterId}`);
    return response.data;
  } catch (error) {
    console.error(`Error deleting adapter ${adapterId}:`, error);
    throw error;
  }
};

/**
 * Train an adapter on a task
 * @param {string} adapterId - Adapter ID
 * @param {Object} trainingData - Training configuration
 * @returns {Promise<Object>} - Training response
 */
export const trainAdapter = async (adapterId, trainingData) => {
  try {
    if (USE_MOCK_DATA) {
      // Generate mock data
      console.log(`Using mock data for trainAdapter: ${adapterId}`);

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 800));

      return {
        adapter_id: adapterId,
        task_id: trainingData.task_id,
        status: 'training',
        results: null
      };
    }

    // Real API call
    const response = await axios.post(`${API_URL}/adapters/${adapterId}/train`, trainingData);
    return response.data;
  } catch (error) {
    console.error(`Error training adapter ${adapterId}:`, error);
    throw error;
  }
};

/**
 * Evaluate an adapter on a task
 * @param {string} adapterId - Adapter ID
 * @param {Object} evaluationData - Evaluation configuration
 * @returns {Promise<Object>} - Evaluation results
 */
export const evaluateAdapter = async (adapterId, evaluationData) => {
  try {
    if (USE_MOCK_DATA) {
      // Generate mock data
      console.log(`Using mock data for evaluateAdapter: ${adapterId}`);
      const mockResults = generateMockEvaluationResults(adapterId, evaluationData.task_id);

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      return mockResults;
    }

    // Real API call
    const response = await axios.post(`${API_URL}/adapters/${adapterId}/evaluate`, evaluationData);
    return response.data;
  } catch (error) {
    console.error(`Error evaluating adapter ${adapterId}:`, error);
    throw error;
  }
};

/**
 * Compute forgetting for a task
 * @param {string} adapterId - Adapter ID
 * @param {Object} forgettingData - Forgetting computation configuration
 * @returns {Promise<Object>} - Forgetting results
 */
export const computeForgetting = async (adapterId, forgettingData) => {
  try {
    if (USE_MOCK_DATA) {
      // Generate mock data
      console.log(`Using mock data for computeForgetting: ${adapterId}`);
      const mockResults = generateMockForgettingResults(
        adapterId,
        forgettingData.task_id,
        forgettingData.metric_key
      );

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1200));

      return mockResults;
    }

    // Real API call
    const response = await axios.post(`${API_URL}/adapters/${adapterId}/forgetting`, forgettingData);
    return response.data;
  } catch (error) {
    console.error(`Error computing forgetting for adapter ${adapterId}:`, error);
    throw error;
  }
};

/**
 * Generate text using an adapter
 * @param {string} adapterId - Adapter ID
 * @param {Object} generationData - Generation configuration
 * @returns {Promise<Object>} - Generated text
 */
export const generateText = async (adapterId, generationData) => {
  try {
    if (USE_MOCK_DATA) {
      // Generate mock data
      console.log(`Using mock data for generateText: ${adapterId}`);

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      return {
        adapter_id: adapterId,
        prompt: generationData.prompt,
        generated_text: `This is a mock generated text for the prompt: "${generationData.prompt}". The adapter ${adapterId} would generate a more meaningful response in a real environment.`
      };
    }

    // Real API call
    const response = await axios.post(`${API_URL}/adapters/${adapterId}/generate`, generationData);
    return response.data;
  } catch (error) {
    console.error(`Error generating text with adapter ${adapterId}:`, error);
    throw error;
  }
};
