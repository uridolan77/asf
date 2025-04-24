// Type definitions for API responses and data models
export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  created_at: string;
  last_login?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterData extends LoginCredentials {
  email: string;
  role?: string;
}

export interface ApiError {
  status: number;
  message: string;
  detail?: string;
}

// Add more interfaces as needed for your specific API responses