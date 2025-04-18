import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import apiService from '../services/api';

// Create context
const AuthContext = createContext();

// Token refresh interval (15 minutes)
const TOKEN_REFRESH_INTERVAL = 15 * 60 * 1000;

/**
 * Authentication provider component
 */
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  
  // Load user from local storage on mount
  useEffect(() => {
    const loadUser = async () => {
      try {
        const token = localStorage.getItem('token');
        
        if (token) {
          // Validate token by fetching user profile
          const userProfile = await apiService.auth.getProfile();
          setUser(userProfile);
        }
      } catch (error) {
        console.error('Error loading user:', error);
        // Clear invalid token
        localStorage.removeItem('token');
        setUser(null);
      } finally {
        setLoading(false);
      }
    };
    
    loadUser();
  }, []);
  
  // Set up token refresh interval
  useEffect(() => {
    if (!user) return;
    
    const refreshToken = async () => {
      try {
        const response = await apiService.auth.refreshToken();
        localStorage.setItem('token', response.token);
      } catch (error) {
        console.error('Error refreshing token:', error);
        // Handle token refresh failure
        if (error.response && error.response.status === 401) {
          logout();
        }
      }
    };
    
    const intervalId = setInterval(refreshToken, TOKEN_REFRESH_INTERVAL);
    
    return () => clearInterval(intervalId);
  }, [user]);
  
  /**
   * Login with username and password
   */
  const login = async (username, password) => {
    try {
      setError(null);
      setLoading(true);
      
      const response = await apiService.auth.login(username, password);
      
      // Store token in local storage
      localStorage.setItem('token', response.token);
      
      // Set user state
      setUser(response.user);
      
      return response.user;
    } catch (error) {
      console.error('Login error:', error);
      setError(error.response?.data?.detail || 'Login failed');
      throw error;
    } finally {
      setLoading(false);
    }
  };
  
  /**
   * Logout the current user
   */
  const logout = useCallback(async () => {
    try {
      // Call logout API if user is logged in
      if (user) {
        await apiService.auth.logout();
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear token and user state
      localStorage.removeItem('token');
      setUser(null);
      
      // Redirect to login page
      navigate('/login');
    }
  }, [user, navigate]);
  
  /**
   * Get access token
   */
  const getAccessToken = useCallback(() => {
    return localStorage.getItem('token');
  }, []);
  
  // Context value
  const value = {
    user,
    loading,
    error,
    login,
    logout,
    getAccessToken,
    isAuthenticated: !!user
  };
  
  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

/**
 * Hook to use the auth context
 */
export const useAuth = () => {
  const context = useContext(AuthContext);
  
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
};

export default AuthContext;
