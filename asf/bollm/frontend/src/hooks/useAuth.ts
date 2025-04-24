// React Query hooks for authentication
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { AuthResponse, LoginCredentials, RegisterData, User } from '../types/api';
import api from '../services/apiClient';

// Login mutation hook
export const useLogin = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (credentials: LoginCredentials) => 
      api.post<AuthResponse>('/auth/login', credentials),
    
    onSuccess: (data) => {
      // Store the token
      localStorage.setItem('token', data.access_token);
      
      // Set user data in the query cache
      queryClient.setQueryData(['user'], data.user);
    },
  });
};

// Register mutation hook
export const useRegister = () => {
  return useMutation({
    mutationFn: (userData: RegisterData) => 
      api.post<AuthResponse>('/auth/register', userData),
  });
};

// Logout mutation hook
export const useLogout = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: () => api.post<void>('/auth/logout'),
    
    onSuccess: () => {
      // Remove token from localStorage
      localStorage.removeItem('token');
      
      // Clear the user from the query cache
      queryClient.removeQueries();
    },
  });
};