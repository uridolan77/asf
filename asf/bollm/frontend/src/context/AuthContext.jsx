// frontend/src/context/AuthContext.jsx
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useNotification } from './NotificationContext.jsx';

// Create the AuthContext
const AuthContext = createContext();

// Base API URL
const API_URL = 'http://localhost:8000';  // Hardcoded for now

// Create the AuthProvider component
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const navigate = useNavigate();
  const { showError, showSuccess } = useNotification();

  // Create API instance with interceptors
  const api = axios.create({
    baseURL: API_URL,
  });

  // Add token to requests
  api.interceptors.request.use(
    (config) => {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => Promise.reject(error)
  );

  // Handle response errors (token refresh logic)
  api.interceptors.response.use(
    (response) => response,
    async (error) => {
      const originalRequest = error.config;

      // If the error is 401 and we haven't already tried to refresh the token
      if (error.response?.status === 401 && !originalRequest._retry) {
        originalRequest._retry = true;

        try {
          // Attempt to refresh the token
          const refreshToken = localStorage.getItem('refreshToken');
          if (!refreshToken) {
            throw new Error('No refresh token available');
          }

          const res = await axios.post(`${API_URL}/api/refresh-token`, {
            refresh_token: refreshToken
          });

          // Store the new tokens
          localStorage.setItem('token', res.data.access_token);
          localStorage.setItem('refreshToken', res.data.refresh_token);

          // Retry the original request with the new token
          originalRequest.headers.Authorization = `Bearer ${res.data.access_token}`;
          return api(originalRequest);
        } catch (refreshError) {
          // If refresh token is invalid, logout
          logout();
          return Promise.reject(refreshError);
        }
      }

      return Promise.reject(error);
    }
  );

  // Initialize auth state from localStorage
  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('token');

      if (token) {
        try {
          // Verify token by fetching user data
          const response = await axios.get(`${API_URL}/api/me`, {
            headers: { Authorization: `Bearer ${token}` },
            withCredentials: false, // Set to false for CORS
          });
          setUser(response.data);
          setIsAuthenticated(true);
        } catch (error) {
          console.error('Failed to fetch user profile:', error);
          localStorage.removeItem('token');
          localStorage.removeItem('refreshToken');
        }
      }

      setLoading(false);
    };

    initAuth();
  }, []);

  // Login function
  const login = useCallback(async (credentials) => {
    try {
      setLoading(true);

      const response = await axios.post(`${API_URL}/api/login`,
        new URLSearchParams(credentials).toString(),
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          withCredentials: false, // Set to false for CORS
        }
      );

      const { access_token } = response.data;
      localStorage.setItem('token', access_token);

      // Fetch user data
      try {
        const userResponse = await axios.get(`${API_URL}/api/me`, {
          headers: { Authorization: `Bearer ${access_token}` },
          withCredentials: false,
        });
        setUser(userResponse.data);
      } catch (userError) {
        console.warn('Could not fetch user data:', userError);
        // Continue login process even if user data fetch fails
      }

      setIsAuthenticated(true);
      showSuccess('Successfully logged in');
      navigate('/dashboard');
      return true;
    } catch (error) {
      console.error('Login error:', error);
      const message = error.response?.data?.detail || error.message || 'Failed to login';
      showError(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [navigate, showSuccess, showError]);

  // Logout function
  const logout = useCallback(() => {
    localStorage.removeItem('token');
    localStorage.removeItem('refreshToken');
    setUser(null);
    setIsAuthenticated(false);
    navigate('/');
  }, [navigate]);

  // Register function
  const register = useCallback(async (userData) => {
    try {
      setLoading(true);
      await axios.post(`${API_URL}/api/register`, userData);
      showSuccess('Registration successful! You can now login.');
      navigate('/');
      return true;
    } catch (error) {
      const message = error.response?.data?.message || 'Registration failed';
      showError(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [navigate, showSuccess, showError]);

  // Update user profile
  const updateProfile = useCallback(async (profileData) => {
    try {
      setLoading(true);
      const response = await api.put('/api/profile', profileData);
      setUser(response.data);
      showSuccess('Profile updated successfully');
      return true;
    } catch (error) {
      const message = error.response?.data?.message || 'Failed to update profile';
      showError(message);
      return false;
    } finally {
      setLoading(false);
    }
  }, [showSuccess, showError]);

  // Check if user has a specific role
  const hasRole = useCallback((role) => {
    return user?.roles?.includes(role) || false;
  }, [user]);

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        isAuthenticated,
        login,
        logout,
        register,
        updateProfile,
        hasRole,
        api,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use the auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;
