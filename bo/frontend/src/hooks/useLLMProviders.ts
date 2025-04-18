import { useApiQuery, useApiPost, useApiDelete } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface LLMProvider {
  id: string;
  name: string;
  provider_type: string;
  display_name: string;
  enabled: boolean;
  models: string[];
  connection_params: Record<string, any>;
  created_at: string;
  updated_at: string;
  status?: string;
  description?: string;
}

export interface LLMModel {
  id: string;
  name: string;
  provider_id: string;
  model_type: string;
  parameters: Record<string, any>;
  capabilities: string[];
  created_at: string;
  updated_at: string;
  status?: string;
  description?: string;
}

export interface CreateProviderParams {
  provider_type: string;
  display_name: string;
  connection_params: Record<string, any>;
  enabled?: boolean;
}

export interface UpdateProviderParams {
  display_name?: string;
  connection_params?: Record<string, any>;
  enabled?: boolean;
}

export interface TestProviderParams {
  provider_id: string;
  test_type?: string;
}

export interface ProviderTestResult {
  success: boolean;
  message: string;
  details?: Record<string, any>;
}

/**
 * Hook for LLM providers operations
 */
export function useLLMProviders() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Get all providers
  const {
    data: providers = [],
    isLoading: isLoadingProviders,
    isError: isErrorProviders,
    error: errorProviders,
    refetch: refetchProviders
  } = useApiQuery<LLMProvider[]>(
    '/api/llm/providers',
    ['llm', 'providers'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch LLM providers: ${error.message}`);
      }
    }
  );

  // Get provider details
  const getProviderDetails = (providerId: string) => {
    return useApiQuery<LLMProvider>(
      `/api/llm/providers/${providerId}`,
      ['llm', 'providers', providerId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!providerId,
        onError: (error) => {
          showError(`Failed to fetch provider details: ${error.message}`);
        }
      }
    );
  };

  // Create provider
  const createProvider = (params: CreateProviderParams) => {
    return useApiPost<LLMProvider, CreateProviderParams>(
      '/api/llm/providers',
      {
        onSuccess: (data) => {
          showSuccess(`Provider "${data.display_name}" created successfully`);
          queryClient.invalidateQueries({ queryKey: ['llm', 'providers'] });
        },
        onError: (error) => {
          showError(`Failed to create provider: ${error.message}`);
        }
      }
    );
  };

  // Update provider
  const updateProvider = (providerId: string, params: UpdateProviderParams) => {
    return useApiPost<LLMProvider, UpdateProviderParams>(
      `/api/llm/providers/${providerId}`,
      {
        onSuccess: (data) => {
          showSuccess(`Provider "${data.display_name}" updated successfully`);
          queryClient.invalidateQueries({ queryKey: ['llm', 'providers'] });
          queryClient.invalidateQueries({ queryKey: ['llm', 'providers', providerId] });
        },
        onError: (error) => {
          showError(`Failed to update provider: ${error.message}`);
        }
      }
    );
  };

  // Delete provider
  const deleteProvider = (providerId: string) => {
    return useApiDelete<{ success: boolean }, void>(
      `/api/llm/providers/${providerId}`,
      {
        onSuccess: () => {
          showSuccess('Provider deleted successfully');
          queryClient.invalidateQueries({ queryKey: ['llm', 'providers'] });
        },
        onError: (error) => {
          showError(`Failed to delete provider: ${error.message}`);
        }
      }
    );
  };

  // Test provider
  const testProvider = (params: TestProviderParams) => {
    return useApiPost<ProviderTestResult, TestProviderParams>(
      '/api/llm/providers/test',
      {
        onSuccess: (data) => {
          if (data.success) {
            showSuccess(`Provider test successful: ${data.message}`);
          } else {
            showError(`Provider test failed: ${data.message}`);
          }
        },
        onError: (error) => {
          showError(`Provider test failed: ${error.message}`);
        }
      }
    );
  };

  // Get provider types
  const {
    data: providerTypes = [],
    isLoading: isLoadingProviderTypes,
    isError: isErrorProviderTypes,
    error: errorProviderTypes
  } = useApiQuery<Array<{ id: string; name: string; description: string }>>(
    '/api/llm/provider-types',
    ['llm', 'provider-types'],
    {
      staleTime: 3600000, // 1 hour
      onError: (error) => {
        showError(`Failed to fetch provider types: ${error.message}`);
      }
    }
  );

  // Get all models
  const {
    data: models = [],
    isLoading: isLoadingModels,
    isError: isErrorModels,
    error: errorModels,
    refetch: refetchModels
  } = useApiQuery<LLMModel[]>(
    '/api/llm/models',
    ['llm', 'models'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch LLM models: ${error.message}`);
      }
    }
  );

  // Get models by provider
  const getModelsByProvider = (providerId: string) => {
    return useApiQuery<LLMModel[]>(
      `/api/llm/providers/${providerId}/models`,
      ['llm', 'providers', providerId, 'models'],
      {
        staleTime: 60000, // 1 minute
        enabled: !!providerId,
        onError: (error) => {
          showError(`Failed to fetch provider models: ${error.message}`);
        }
      }
    );
  };

  return {
    // Providers
    providers,
    isLoadingProviders,
    isErrorProviders,
    errorProviders,
    refetchProviders,
    getProviderDetails,
    createProvider,
    updateProvider,
    deleteProvider,
    testProvider,
    
    // Provider types
    providerTypes,
    isLoadingProviderTypes,
    isErrorProviderTypes,
    errorProviderTypes,
    
    // Models
    models,
    isLoadingModels,
    isErrorModels,
    errorModels,
    refetchModels,
    getModelsByProvider
  };
}
