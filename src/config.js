/**
 * Application configuration
 */

// API base URL
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

// Authentication settings
export const AUTH_CONFIG = {
  tokenKey: 'token',
  refreshTokenKey: 'refreshToken',
  tokenExpiryKey: 'tokenExpiry',
  loginPath: '/login',
  homePath: '/dashboard'
};

// Default pagination settings
export const PAGINATION_CONFIG = {
  defaultPageSize: 10,
  pageSizeOptions: [5, 10, 25, 50, 100]
};

// Theme settings
export const THEME_CONFIG = {
  defaultTheme: 'light',
  themeStorageKey: 'theme'
};

// Date format settings
export const DATE_FORMAT = 'yyyy-MM-dd';
export const DATE_TIME_FORMAT = 'yyyy-MM-dd HH:mm:ss';
export const TIME_FORMAT = 'HH:mm:ss';

// File upload settings
export const UPLOAD_CONFIG = {
  maxFileSize: 10 * 1024 * 1024, // 10MB
  acceptedFileTypes: {
    documents: ['.pdf', '.doc', '.docx', '.txt'],
    images: ['.jpg', '.jpeg', '.png', '.gif'],
    data: ['.csv', '.xls', '.xlsx', '.json']
  }
};

// LLM settings
export const LLM_CONFIG = {
  defaultProvider: 'openai',
  defaultModel: 'gpt-3.5-turbo',
  maxTokens: 2048,
  temperature: 0.7
};

// CL-PEFT settings
export const CL_PEFT_CONFIG = {
  defaultStrategy: 'ewc',
  defaultPeftMethod: 'lora',
  defaultLoraRank: 8,
  maxMemoryUsage: 16 // GB
};
