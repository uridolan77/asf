/**
 * API service index
 * 
 * This file exports all API services as a single object.
 */

import llm from './llm';
import auth from './auth';
import documentProcessing from './documentProcessing';

const apiService = {
  llm,
  auth,
  documentProcessing
};

export default apiService;
