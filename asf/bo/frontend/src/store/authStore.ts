// Global state management with Zustand
import { create } from 'zustand';
import { User } from '../types/api';

// Define the store state interface
interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  setUser: (user: User | null) => void;
  setLoading: (isLoading: boolean) => void;
  logout: () => void;
}

// Create the store
export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  
  setUser: (user) => set({
    user,
    isAuthenticated: !!user,
    isLoading: false
  }),
  
  setLoading: (isLoading) => set({ isLoading }),
  
  logout: () => set({
    user: null,
    isAuthenticated: false,
    isLoading: false
  })
}));