/**
 * Clinical Data Service
 * 
 * This service provides methods for interacting with the clinical data API endpoints.
 * It handles communication with the backend for clinical data operations.
 */
import axios from 'axios';

// Base URL for API requests
const API_BASE_URL = 'http://localhost:8000/api/medical/clinical-data';

/**
 * Get the authorization header with the current token
 * @returns {Object} Headers object with Authorization
 */
const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  };
};

/**
 * Search for a medical term and find related SNOMED CT concepts and clinical trials
 * @param {string} term - The medical term to search for
 * @param {number} maxTrials - Maximum number of trials to return
 * @returns {Promise} Promise resolving to search results
 */
export const searchConceptAndTrials = async (term, maxTrials = 10) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/search?term=${encodeURIComponent(term)}&max_trials=${maxTrials}`,
      getAuthHeaders()
    );
    return response.data;
  } catch (error) {
    console.error('Error searching for concept and trials:', error);
    throw error;
  }
};

/**
 * Find clinical trials related to a specific medical concept
 * @param {string} conceptId - The concept identifier (e.g., SNOMED CT concept ID)
 * @param {string} terminology - The terminology system (default: "SNOMEDCT")
 * @param {number} maxTrials - Maximum number of trials to return
 * @returns {Promise} Promise resolving to search results
 */
export const getTrialsByConceptId = async (conceptId, terminology = 'SNOMEDCT', maxTrials = 10) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/concept/${conceptId}/trials?terminology=${terminology}&max_trials=${maxTrials}`,
      getAuthHeaders()
    );
    return response.data;
  } catch (error) {
    console.error('Error getting trials by concept ID:', error);
    throw error;
  }
};

/**
 * Map all conditions in a clinical trial to SNOMED CT concepts
 * @param {string} nctId - The ClinicalTrials.gov identifier (NCT number)
 * @returns {Promise} Promise resolving to mapping results
 */
export const mapTrialConditions = async (nctId) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/trial/${nctId}/mapping`,
      getAuthHeaders()
    );
    return response.data;
  } catch (error) {
    console.error('Error mapping trial conditions:', error);
    throw error;
  }
};

/**
 * Find clinical trials with semantic expansion of the search term
 * @param {string} term - The medical term to search for
 * @param {boolean} includeSimilar - Whether to include similar concepts
 * @param {number} maxTrials - Maximum number of trials to return
 * @returns {Promise} Promise resolving to search results
 */
export const findTrialsWithSemanticExpansion = async (term, includeSimilar = true, maxTrials = 20) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/semantic-search?term=${encodeURIComponent(term)}&include_similar=${includeSimilar}&max_trials=${maxTrials}`,
      getAuthHeaders()
    );
    return response.data;
  } catch (error) {
    console.error('Error finding trials with semantic expansion:', error);
    throw error;
  }
};

/**
 * Get semantic context for a clinical trial
 * @param {string} nctId - The ClinicalTrials.gov identifier (NCT number)
 * @returns {Promise} Promise resolving to semantic context
 */
export const getTrialSemanticContext = async (nctId) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/trial/${nctId}/semantic-context`,
      getAuthHeaders()
    );
    return response.data;
  } catch (error) {
    console.error('Error getting trial semantic context:', error);
    throw error;
  }
};

/**
 * Analyze clinical trial phases for a medical concept
 * @param {string} conceptId - The concept identifier (e.g., SNOMED CT concept ID)
 * @param {string} terminology - The terminology system (default: "SNOMEDCT")
 * @param {boolean} includeDescendants - Whether to include descendant concepts
 * @param {number} maxResults - Maximum number of trials to analyze
 * @returns {Promise} Promise resolving to analysis results
 */
export const analyzeTrialPhasesByConceptId = async (
  conceptId,
  terminology = 'SNOMEDCT',
  includeDescendants = true,
  maxResults = 500
) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/concept/${conceptId}/phase-analysis?terminology=${terminology}&include_descendants=${includeDescendants}&max_results=${maxResults}`,
      getAuthHeaders()
    );
    return response.data;
  } catch (error) {
    console.error('Error analyzing trial phases by concept ID:', error);
    throw error;
  }
};

// Export all functions as a service object
const clinicalDataService = {
  searchConceptAndTrials,
  getTrialsByConceptId,
  mapTrialConditions,
  findTrialsWithSemanticExpansion,
  getTrialSemanticContext,
  analyzeTrialPhasesByConceptId
};

export default clinicalDataService;
