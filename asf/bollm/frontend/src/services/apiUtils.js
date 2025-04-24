/**
 * Handle API response in a consistent way
 * 
 * @param {Object} response - Axios response object
 * @returns {Object} - Standardized response object
 */
export const handleApiResponse = (response) => {
  // If no response, return error
  if (!response) {
    return {
      success: false,
      error: 'No response received',
      data: null
    };
  }
  
  // If response has status in 2xx range, return success
  if (response.status >= 200 && response.status < 300) {
    return {
      success: true,
      data: response.data,
      status: response.status
    };
  }
  
  // Otherwise, return error
  let errorMessage = 'An error occurred';
  
  // Try to extract error message from response
  if (response.data) {
    if (typeof response.data === 'string') {
      errorMessage = response.data;
    } else if (response.data.message) {
      errorMessage = response.data.message;
    } else if (response.data.error) {
      errorMessage = response.data.error;
    } else if (response.data.detail) {
      errorMessage = response.data.detail;
    }
  }
  
  return {
    success: false,
    error: errorMessage,
    status: response.status,
    data: response.data
  };
};

/**
 * Format query parameters for API requests
 * 
 * @param {Object} params - Query parameters
 * @returns {string} - Formatted query string
 */
export const formatQueryParams = (params) => {
  if (!params || Object.keys(params).length === 0) {
    return '';
  }
  
  const queryParams = new URLSearchParams();
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      if (Array.isArray(value)) {
        value.forEach(item => queryParams.append(`${key}[]`, item));
      } else {
        queryParams.append(key, value);
      }
    }
  });
  
  return queryParams.toString();
};

/**
 * Format form data for API requests
 * 
 * @param {Object} data - Form data
 * @returns {FormData} - Formatted form data
 */
export const formatFormData = (data) => {
  if (!data) {
    return null;
  }
  
  const formData = new FormData();
  
  Object.entries(data).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      if (Array.isArray(value)) {
        value.forEach(item => {
          if (item instanceof File) {
            formData.append(key, item, item.name);
          } else {
            formData.append(key, item);
          }
        });
      } else if (value instanceof File) {
        formData.append(key, value, value.name);
      } else {
        formData.append(key, value);
      }
    }
  });
  
  return formData;
};
