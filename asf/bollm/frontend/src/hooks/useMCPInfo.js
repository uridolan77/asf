import { useQuery } from '@tanstack/react-query';
import apiService from '../services/api';
import { useNotification } from '../context/NotificationContext';

/**
 * Custom hook for fetching MCP information
 *
 * This hook provides methods for fetching MCP information
 * with automatic caching and refetching.
 */
export const useMCPInfo = () => {
  const { showError } = useNotification();

  // Fetch MCP information
  const {
    data: mcpInfo,
    isLoading,
    isError,
    error,
    refetch
  } = useQuery({
    queryKey: ['mcpInfo'],
    queryFn: () => apiService.llm.getMCPInfo(),
    staleTime: 3600000, // 1 hour
    refetchOnWindowFocus: false,
    onError: (err) => {
      showError(`Failed to fetch MCP information: ${err.message}`);
    }
  });

  return {
    mcpInfo,
    isLoading,
    isError,
    error,
    refetch
  };
};
