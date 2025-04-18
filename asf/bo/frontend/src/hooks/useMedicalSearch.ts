import { useApiQuery, useApiPost } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';

// Types
interface SearchResult {
  id: string;
  title: string;
  authors: string[];
  journal: string;
  year: number;
  abstract: string;
  relevance_score: number;
  source: string;
}

interface SearchResponse {
  articles: SearchResult[];
  query: string;
  total_results: number;
}

interface PICOSearchParams {
  condition: string;
  interventions: string[];
  outcomes: string[];
  population?: string;
  study_design?: string;
  years?: number;
  max_results?: number;
}

interface PICOSearchResponse {
  total_count: number;
  page: number;
  page_size: number;
  results: SearchResult[];
}

interface SearchHistoryItem {
  id: string;
  query: string;
  type: 'standard' | 'pico';
  timestamp: string;
  result_count: number;
}

/**
 * Hook for medical search operations
 */
export function useMedicalSearch() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();

  // Standard search
  const search = (query: string, maxResults: number = 20) => {
    return useApiPost<SearchResponse, { query: string; max_results: number }>(
      '/api/medical/search',
      {
        onSuccess: (data) => {
          // Save search to history
          saveSearchToHistory({
            query,
            type: 'standard',
            timestamp: new Date().toISOString(),
            result_count: data.total_results
          });
        },
        onError: (error) => {
          showError(`Search failed: ${error.message}`);
        }
      }
    );
  };

  // PICO search
  const picoSearch = (params: PICOSearchParams) => {
    return useApiPost<PICOSearchResponse, PICOSearchParams>(
      '/api/medical/search/pico',
      {
        onSuccess: (data) => {
          // Save search to history
          const picoDescription = `P: ${params.population || params.condition}, ` +
                                 `I: ${params.interventions.join(', ')}, ` +
                                 `O: ${params.outcomes.join(', ')}`;
          
          saveSearchToHistory({
            query: picoDescription,
            type: 'pico',
            timestamp: new Date().toISOString(),
            result_count: data.total_count
          });
        },
        onError: (error) => {
          showError(`PICO search failed: ${error.message}`);
        }
      }
    );
  };

  // Get search history
  const {
    data: searchHistory = { searches: [] },
    isLoading: isLoadingHistory,
    isError: isErrorHistory,
    error: errorHistory,
    refetch: refetchHistory
  } = useApiQuery<{ searches: SearchHistoryItem[] }>(
    '/api/medical/search/history',
    ['medical', 'search', 'history'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        console.error('Failed to fetch search history:', error);
      }
    }
  );

  // Save search to history
  const {
    mutate: saveSearchToHistory,
    isPending: isSavingHistory,
    isError: isErrorSavingHistory,
    error: errorSavingHistory
  } = useApiPost<any, Omit<SearchHistoryItem, 'id'>>(
    '/api/medical/search/history',
    {
      onSuccess: () => {
        // Invalidate search history cache
        queryClient.invalidateQueries({ queryKey: ['medical', 'search', 'history'] });
      },
      onError: (error) => {
        console.error('Failed to save search to history:', error);
      }
    }
  );

  return {
    // Search functions
    search,
    picoSearch,
    
    // Search history
    searchHistory: searchHistory.searches,
    isLoadingHistory,
    isErrorHistory,
    errorHistory,
    refetchHistory,
    saveSearchToHistory,
    isSavingHistory,
    isErrorSavingHistory,
    errorSavingHistory
  };
}
