import { useState, useEffect } from 'react';
import { useApiPost, useApiQuery } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface SearchResult {
  id: string;
  title: string;
  authors: string[];
  journal: string;
  year: number;
  abstract: string;
  relevance_score: number;
  source: string;
  pmid?: string;
  doi?: string;
  url?: string;
}

export interface PICOSearchParams {
  condition: string;
  interventions: string[];
  outcomes: string[];
  population?: string;
  study_design?: string;
  years?: number;
  max_results?: number;
  page?: number;
  page_size?: number;
}

export interface PICOSearchResponse {
  total_count: number;
  page: number;
  page_size: number;
  articles: SearchResult[];
}

export interface SearchHistoryItem {
  id: string;
  timestamp: string;
  condition: string;
  interventions: string[];
  outcomes: string[];
  population?: string;
  studyDesign?: string;
  years?: number;
  resultCount: number;
}

export interface SuggestionResponse {
  suggestions: string[];
}

/**
 * Hook for PICO search operations
 */
export function usePICOSearch() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');
  
  // State for search history
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([]);
  
  // Load search history from localStorage
  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem('picoSearchHistory');
      if (savedHistory) {
        setSearchHistory(JSON.parse(savedHistory));
      }
    } catch (error) {
      console.error('Error loading PICO search history:', error);
    }
  }, []);
  
  // Save search to history
  const saveToHistory = (search: Omit<SearchHistoryItem, 'id'>) => {
    try {
      const newSearch: SearchHistoryItem = {
        ...search,
        id: Date.now().toString()
      };
      
      const updatedHistory = [newSearch, ...searchHistory.slice(0, 9)];
      setSearchHistory(updatedHistory);
      localStorage.setItem('picoSearchHistory', JSON.stringify(updatedHistory));
    } catch (error) {
      console.error('Error saving PICO search to history:', error);
    }
  };
  
  // Clear search history
  const clearHistory = () => {
    setSearchHistory([]);
    localStorage.removeItem('picoSearchHistory');
  };
  
  // PICO search
  const picoSearch = (params: PICOSearchParams) => {
    return useApiPost<PICOSearchResponse, PICOSearchParams>(
      '/api/medical/search/pico',
      {
        onSuccess: (data) => {
          // Save search to history
          saveToHistory({
            timestamp: new Date().toISOString(),
            condition: params.condition,
            interventions: params.interventions,
            outcomes: params.outcomes,
            population: params.population,
            studyDesign: params.study_design,
            years: params.years,
            resultCount: data.total_count
          });
        },
        onError: (error) => {
          showError(`PICO search failed: ${error.message}`);
        }
      }
    );
  };
  
  // Get condition suggestions
  const getConditionSuggestions = (query: string) => {
    return useApiQuery<SuggestionResponse>(
      `/api/medical/suggestions/conditions?query=${encodeURIComponent(query)}`,
      ['medical', 'suggestions', 'conditions', query],
      {
        enabled: !!query && query.length >= 2,
        staleTime: 60 * 60 * 1000, // 1 hour
        onError: (error) => {
          console.error('Failed to fetch condition suggestions:', error);
        }
      }
    );
  };
  
  // Get intervention suggestions
  const getInterventionSuggestions = (query: string) => {
    return useApiQuery<SuggestionResponse>(
      `/api/medical/suggestions/interventions?query=${encodeURIComponent(query)}`,
      ['medical', 'suggestions', 'interventions', query],
      {
        enabled: !!query && query.length >= 2,
        staleTime: 60 * 60 * 1000, // 1 hour
        onError: (error) => {
          console.error('Failed to fetch intervention suggestions:', error);
        }
      }
    );
  };
  
  // Get outcome suggestions
  const getOutcomeSuggestions = (query: string) => {
    return useApiQuery<SuggestionResponse>(
      `/api/medical/suggestions/outcomes?query=${encodeURIComponent(query)}`,
      ['medical', 'suggestions', 'outcomes', query],
      {
        enabled: !!query && query.length >= 2,
        staleTime: 60 * 60 * 1000, // 1 hour
        onError: (error) => {
          console.error('Failed to fetch outcome suggestions:', error);
        }
      }
    );
  };
  
  return {
    // Search
    picoSearch,
    
    // Suggestions
    getConditionSuggestions,
    getInterventionSuggestions,
    getOutcomeSuggestions,
    
    // History
    searchHistory,
    saveToHistory,
    clearHistory
  };
}
