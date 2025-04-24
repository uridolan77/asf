import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiService from '../services/api';
import { useNotification } from '../context/NotificationContext';

/**
 * Custom hook for managing MCP providers using React Query
 *
 * This hook provides methods for fetching, creating, updating, and deleting
 * MCP providers with automatic cache invalidation and optimistic updates.
 */
export const useMCPProviders = () => {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();

  // Fetch all providers
  const {
    data: providers = [],
    isLoading,
    isError,
    error,
    refetch
  } = useQuery({
    queryKey: ['mcpProviders'],
    queryFn: () => apiService.llm.getMCPProviders(),
    staleTime: 60000, // 1 minute
    refetchOnWindowFocus: true,
    onError: (err) => {
      showError(`Failed to fetch MCP providers: ${err.message}`);
    }
  });

  // Add a new provider
  const addProvider = useMutation({
    mutationFn: (config) => apiService.llm.registerMCPProvider(config),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['mcpProviders'] });
      showSuccess(`Provider ${data.provider.provider_id} added successfully`);
    },
    onError: (err) => {
      showError(`Failed to add provider: ${err.message}`);
    }
  });

  // Update an existing provider
  const updateProvider = useMutation({
    mutationFn: ({ providerId, config }) => apiService.llm.updateMCPProvider(providerId, config),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['mcpProviders'] });
      showSuccess(`Provider ${data.provider.provider_id} updated successfully`);
    },
    onError: (err) => {
      showError(`Failed to update provider: ${err.message}`);
    }
  });

  // Delete a provider
  const deleteProvider = useMutation({
    mutationFn: (providerId) => apiService.llm.deleteMCPProvider(providerId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['mcpProviders'] });
      showSuccess(`Provider ${variables} deleted successfully`);
    },
    onError: (err) => {
      showError(`Failed to delete provider: ${err.message}`);
    }
  });

  // Test a provider connection
  const testProvider = useMutation({
    mutationFn: (providerId) => apiService.llm.testMCPProvider(providerId),
    onSuccess: (data, variables) => {
      showSuccess(`Connection test for ${variables} successful`);
      return data;
    },
    onError: (err) => {
      showError(`Connection test failed: ${err.message}`);
    }
  });

  // Get provider status
  const getProviderStatus = async (providerId) => {
    try {
      return await apiService.llm.getMCPProviderStatus(providerId);
    } catch (err) {
      showError(`Failed to get provider status: ${err.message}`);
      throw err;
    }
  };

  // Get provider usage statistics
  const getProviderUsage = async (providerId, period = 'day') => {
    try {
      return await apiService.llm.getMCPProviderUsage(providerId, period);
    } catch (err) {
      showError(`Failed to get provider usage: ${err.message}`);
      throw err;
    }
  };

  return {
    providers,
    isLoading,
    isError,
    error,
    refetch,
    addProvider: addProvider.mutate,
    updateProvider: updateProvider.mutate,
    deleteProvider: deleteProvider.mutate,
    testProvider: testProvider.mutate,
    getProviderStatus,
    getProviderUsage,
    addProviderLoading: addProvider.isLoading,
    updateProviderLoading: updateProvider.isLoading,
    deleteProviderLoading: deleteProvider.isLoading,
    testProviderLoading: testProvider.isLoading
  };
};
