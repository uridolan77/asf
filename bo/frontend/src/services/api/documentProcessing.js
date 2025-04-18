/**
 * Document Processing API service
 * 
 * This service provides methods for interacting with the document processing API.
 */

import axios from 'axios';
import { getAuthHeader } from './auth';

const API_URL = '/api/document-processing';

/**
 * Upload a document for processing
 * 
 * @param {File} file - Document file
 * @param {Object} options - Processing options
 * @returns {Promise<Object>} Upload response with task ID
 */
export const uploadDocument = async (file, options = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  if (options.extractEntities) {
    formData.append('extract_entities', options.extractEntities);
  }
  
  if (options.extractRelations) {
    formData.append('extract_relations', options.extractRelations);
  }
  
  if (options.generateSummary) {
    formData.append('generate_summary', options.generateSummary);
  }
  
  const response = await axios.post(`${API_URL}/upload`, formData, {
    headers: {
      ...(await getAuthHeader()),
      'Content-Type': 'multipart/form-data'
    }
  });
  
  return response.data;
};

/**
 * Get the status of a document processing task
 * 
 * @param {string} taskId - Task ID
 * @returns {Promise<Object>} Task status
 */
export const getTaskStatus = async (taskId) => {
  const response = await axios.get(`${API_URL}/tasks/${taskId}/status`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get the result of a document processing task
 * 
 * @param {string} taskId - Task ID
 * @returns {Promise<Object>} Task result
 */
export const getTaskResult = async (taskId) => {
  const response = await axios.get(`${API_URL}/tasks/${taskId}/result`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get all document processing tasks
 * 
 * @param {number} page - Page number
 * @param {number} pageSize - Page size
 * @returns {Promise<Object>} Tasks list with pagination
 */
export const getTasks = async (page = 1, pageSize = 10) => {
  const response = await axios.get(`${API_URL}/tasks`, {
    params: { page, page_size: pageSize },
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Cancel a document processing task
 * 
 * @param {string} taskId - Task ID
 * @returns {Promise<Object>} Success message
 */
export const cancelTask = async (taskId) => {
  const response = await axios.post(`${API_URL}/tasks/${taskId}/cancel`, {}, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Delete a document processing task
 * 
 * @param {string} taskId - Task ID
 * @returns {Promise<Object>} Success message
 */
export const deleteTask = async (taskId) => {
  const response = await axios.delete(`${API_URL}/tasks/${taskId}`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

// Export all functions as an object
const documentProcessingApi = {
  uploadDocument,
  getTaskStatus,
  getTaskResult,
  getTasks,
  cancelTask,
  deleteTask
};

export default documentProcessingApi;
