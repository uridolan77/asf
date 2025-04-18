/**
 * Authentication API service
 * 
 * This service provides methods for authentication and authorization.
 */

import axios from 'axios';

const API_URL = '/api/auth';

/**
 * Login with username and password
 * 
 * @param {string} username - Username
 * @param {string} password - Password
 * @returns {Promise<Object>} Authentication response with token
 */
export const login = async (username, password) => {
  const response = await axios.post(`${API_URL}/login`, {
    username,
    password
  });
  return response.data;
};

/**
 * Logout the current user
 * 
 * @returns {Promise<Object>} Success message
 */
export const logout = async () => {
  const response = await axios.post(`${API_URL}/logout`, {}, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get the current user profile
 * 
 * @returns {Promise<Object>} User profile
 */
export const getProfile = async () => {
  const response = await axios.get(`${API_URL}/profile`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Refresh the authentication token
 * 
 * @returns {Promise<Object>} New token
 */
export const refreshToken = async () => {
  const response = await axios.post(`${API_URL}/refresh-token`, {}, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get authentication header with token
 * 
 * @returns {Promise<Object>} Authentication header
 */
export const getAuthHeader = async () => {
  const token = localStorage.getItem('token');
  
  if (!token) {
    return {};
  }
  
  return {
    Authorization: `Bearer ${token}`
  };
};

// Export all functions as an object
const authApi = {
  login,
  logout,
  getProfile,
  refreshToken,
  getAuthHeader
};

export default authApi;
export { getAuthHeader };
