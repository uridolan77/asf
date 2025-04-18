/**
 * LLM Gateway API service
 *
 * This service provides methods for interacting with the LLM Gateway API,
 * including provider management, metrics, and monitoring.
 */

import axios from 'axios';
import { getAuthHeader } from './auth';

const API_URL = '/api/llm';

/**
 * Get all available LLM providers
 *
 * @returns {Promise<Array>} List of providers
 */
export const getProviders = async () => {
  const response = await axios.get(`${API_URL}/providers`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get information about a specific LLM provider
 *
 * @param {string} providerId - Provider ID
 * @returns {Promise<Object>} Provider information
 */
export const getProvider = async (providerId) => {
  const response = await axios.get(`${API_URL}/providers/${providerId}`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get the current status of an LLM provider
 *
 * @param {string} providerId - Provider ID
 * @returns {Promise<Object>} Provider status
 */
export const getProviderStatus = async (providerId) => {
  const response = await axios.get(`${API_URL}/providers/${providerId}/status`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get usage metrics for an LLM provider
 *
 * @param {string} providerId - Provider ID
 * @param {string} period - Time period (hour, day, week, month)
 * @returns {Promise<Object>} Provider metrics
 */
export const getMCPProviderUsage = async (providerId, period = 'day') => {
  const response = await axios.get(`${API_URL}/providers/${providerId}/metrics`, {
    params: { period },
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get circuit breaker history for an LLM provider
 *
 * @param {string} providerId - Provider ID
 * @param {string} period - Time period (hour, day, week, month)
 * @returns {Promise<Array>} Circuit breaker history
 */
export const getCircuitBreakerHistory = async (providerId, period = 'day') => {
  const response = await axios.get(`${API_URL}/providers/${providerId}/circuit-breaker/history`, {
    params: { period },
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Reset the circuit breaker for an LLM provider
 *
 * @param {string} providerId - Provider ID
 * @returns {Promise<Object>} Success message
 */
export const resetCircuitBreaker = async (providerId) => {
  const response = await axios.post(`${API_URL}/providers/${providerId}/reset-circuit-breaker`, {}, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Refresh the session pool for an LLM provider
 *
 * @param {string} providerId - Provider ID
 * @returns {Promise<Object>} Success message
 */
export const refreshSessions = async (providerId) => {
  const response = await axios.post(`${API_URL}/providers/${providerId}/refresh-sessions`, {}, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get connection pool statistics for an LLM provider
 *
 * @param {string} providerId - Provider ID
 * @returns {Promise<Object>} Connection pool statistics
 */
export const getConnectionPoolStats = async (providerId) => {
  const response = await axios.get(`${API_URL}/providers/${providerId}/connection-pool`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get a summary of LLM Gateway metrics
 *
 * @param {string} period - Time period (hour, day, week, month)
 * @returns {Promise<Object>} Metrics summary
 */
export const getMetricsSummary = async (period = 'day') => {
  const response = await axios.get(`${API_URL}/metrics/summary`, {
    params: { period },
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get error metrics for the LLM Gateway
 *
 * @param {string} period - Time period (hour, day, week, month)
 * @returns {Promise<Object>} Error metrics
 */
export const getErrorMetrics = async (period = 'day') => {
  const response = await axios.get(`${API_URL}/metrics/errors`, {
    params: { period },
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get latency metrics for the LLM Gateway
 *
 * @param {string} period - Time period (hour, day, week, month)
 * @returns {Promise<Object>} Latency metrics
 */
export const getLatencyMetrics = async (period = 'day') => {
  const response = await axios.get(`${API_URL}/metrics/latency`, {
    params: { period },
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get token usage metrics for the LLM Gateway
 *
 * @param {string} period - Time period (hour, day, week, month)
 * @returns {Promise<Object>} Token usage metrics
 */
export const getTokenMetrics = async (period = 'day') => {
  const response = await axios.get(`${API_URL}/metrics/tokens`, {
    params: { period },
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get WebSocket connection statistics
 *
 * @returns {Promise<Object>} WebSocket statistics
 */
export const getWebSocketStats = async () => {
  const response = await axios.get(`${API_URL}/websocket/stats`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get Grafana dashboards
 *
 * @returns {Promise<Object>} Grafana dashboards
 */
export const getGrafanaDashboards = async () => {
  const response = await axios.get(`${API_URL}/grafana/dashboards`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Get Grafana dashboard URL for a provider
 *
 * @param {string} providerId - Provider ID
 * @returns {Promise<Object>} Dashboard URL
 */
export const getProviderDashboardUrl = async (providerId) => {
  const response = await axios.get(`${API_URL}/grafana/dashboards/provider/${providerId}`, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Provision a Grafana dashboard for a provider
 *
 * @param {string} providerId - Provider ID
 * @returns {Promise<Object>} Provision result
 */
export const provisionProviderDashboard = async (providerId) => {
  const response = await axios.post(`${API_URL}/grafana/dashboards/provider/${providerId}`, {}, {
    headers: await getAuthHeader()
  });
  return response.data;
};

/**
 * Set up Grafana with datasources, dashboards, and alert rules
 *
 * @returns {Promise<Object>} Setup result
 */
export const setupGrafana = async () => {
  const response = await axios.post(`${API_URL}/grafana/setup`, {}, {
    headers: await getAuthHeader()
  });
  return response.data;
};

// Export all functions as an object
const llmApi = {
  getProviders,
  getProvider,
  getProviderStatus,
  getMCPProviderUsage,
  getCircuitBreakerHistory,
  resetCircuitBreaker,
  refreshSessions,
  getConnectionPoolStats,
  getMetricsSummary,
  getErrorMetrics,
  getLatencyMetrics,
  getTokenMetrics,
  getWebSocketStats,
  getGrafanaDashboards,
  getProviderDashboardUrl,
  provisionProviderDashboard,
  setupGrafana
};

export default llmApi;
