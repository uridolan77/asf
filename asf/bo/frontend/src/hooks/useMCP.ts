import { useApiQuery, useApiPost, useApiDelete } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface MCPProvider {
  id: string;
  name: string;
  display_name: string;
  transport_type: string;
  status: string;
  enabled: boolean;
  models: string[];
  config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface MCPModel {
  id: string;
  name: string;
  provider_id: string;
  model_type: string;
  capabilities: string[];
  config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface MCPStatus {
  status: string;
  components: Record<string, any>;
  version: string;
  uptime: number;
  active_connections: number;
  request_count: number;
  error_count: number;
}

export interface MCPProviderStatus {
  id: string;
  name: string;
  status: string;
  health_check: {
    status: string;
    last_check: string;
    error?: string;
  };
  circuit_breaker: {
    state: string;
    failure_count: number;
    last_failure: string;
    reset_timeout: number;
  };
  metrics: {
    request_count: number;
    error_count: number;
    latency: {
      avg: number;
      p50: number;
      p95: number;
      p99: number;
    };
  };
}

export interface MCPUsageStats {
  provider_id: string;
  period: string;
  request_count: number;
  error_count: number;
  token_usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  latency: {
    avg: number;
    p50: number;
    p95: number;
    p99: number;
  };
  requests_by_hour: Array<{
    hour: string;
    count: number;
  }>;
  tokens_by_hour: Array<{
    hour: string;
    prompt_tokens: number;
    completion_tokens: number;
  }>;
  errors_by_type: Record<string, number>;
}

export interface CreateMCPProviderParams {
  name: string;
  display_name: string;
  transport_type: string;
  config: Record<string, any>;
  enabled?: boolean;
}

export interface UpdateMCPProviderParams {
  display_name?: string;
  config?: Record<string, any>;
  enabled?: boolean;
}

export interface TestMCPProviderParams {
  provider_id: string;
  test_type?: string;
}

export interface MCPProviderTestResult {
  success: boolean;
  message: string;
  details?: Record<string, any>;
}

/**
 * Hook for MCP operations
 */
export function useMCP() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Get MCP status
  const {
    data: mcpStatus,
    isLoading: isLoadingMCPStatus,
    isError: isErrorMCPStatus,
    error: errorMCPStatus,
    refetch: refetchMCPStatus
  } = useApiQuery<MCPStatus>(
    '/api/llm/mcp/status',
    ['llm', 'mcp', 'status'],
    {
      staleTime: 30000, // 30 seconds
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch MCP status: ${error.message}`);
      }
    }
  );

  // Get all providers
  const {
    data: providers = [],
    isLoading: isLoadingProviders,
    isError: isErrorProviders,
    error: errorProviders,
    refetch: refetchProviders
  } = useApiQuery<MCPProvider[]>(
    '/api/llm/mcp/providers',
    ['llm', 'mcp', 'providers'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch MCP providers: ${error.message}`);
      }
    }
  );

  // Get provider details
  const getProviderDetails = (providerId: string) => {
    return useApiQuery<MCPProvider>(
      `/api/llm/mcp/providers/${providerId}`,
      ['llm', 'mcp', 'providers', providerId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!providerId,
        onError: (error) => {
          showError(`Failed to fetch provider details: ${error.message}`);
        }
      }
    );
  };

  // Get provider status
  const getProviderStatus = (providerId: string) => {
    return useApiQuery<MCPProviderStatus>(
      `/api/llm/mcp/providers/${providerId}/status`,
      ['llm', 'mcp', 'providers', providerId, 'status'],
      {
        staleTime: 30000, // 30 seconds
        enabled: !!providerId,
        refetchInterval: 30000, // Refetch every 30 seconds
        onError: (error) => {
          showError(`Failed to fetch provider status: ${error.message}`);
        }
      }
    );
  };

  // Get provider usage stats
  const getProviderUsage = (providerId: string, period: string = 'day') => {
    return useApiQuery<MCPUsageStats>(
      `/api/llm/mcp/providers/${providerId}/usage?period=${period}`,
      ['llm', 'mcp', 'providers', providerId, 'usage', period],
      {
        staleTime: 60000, // 1 minute
        enabled: !!providerId,
        refetchInterval: 60000, // Refetch every minute
        onError: (error) => {
          showError(`Failed to fetch provider usage stats: ${error.message}`);
        }
      }
    );
  };

  // Create provider
  const createProvider = () => {
    return useApiPost<MCPProvider, CreateMCPProviderParams>(
      '/api/llm/mcp/providers',
      {
        onSuccess: (data) => {
          showSuccess(`Provider "${data.display_name}" created successfully`);
          queryClient.invalidateQueries({ queryKey: ['llm', 'mcp', 'providers'] });
        },
        onError: (error) => {
          showError(`Failed to create provider: ${error.message}`);
        }
      }
    );
  };

  // Update provider
  const updateProvider = (providerId: string) => {
    return useApiPost<MCPProvider, UpdateMCPProviderParams>(
      `/api/llm/mcp/providers/${providerId}`,
      {
        onSuccess: (data) => {
          showSuccess(`Provider "${data.display_name}" updated successfully`);
          queryClient.invalidateQueries({ queryKey: ['llm', 'mcp', 'providers'] });
          queryClient.invalidateQueries({ queryKey: ['llm', 'mcp', 'providers', providerId] });
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
      `/api/llm/mcp/providers/${providerId}`,
      {
        onSuccess: () => {
          showSuccess('Provider deleted successfully');
          queryClient.invalidateQueries({ queryKey: ['llm', 'mcp', 'providers'] });
        },
        onError: (error) => {
          showError(`Failed to delete provider: ${error.message}`);
        }
      }
    );
  };

  // Test provider
  const testProvider = () => {
    return useApiPost<MCPProviderTestResult, TestMCPProviderParams>(
      '/api/llm/mcp/providers/test',
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

  // Get all models
  const {
    data: models = [],
    isLoading: isLoadingModels,
    isError: isErrorModels,
    error: errorModels,
    refetch: refetchModels
  } = useApiQuery<MCPModel[]>(
    '/api/llm/mcp/models',
    ['llm', 'mcp', 'models'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch MCP models: ${error.message}`);
      }
    }
  );

  // Get models by provider
  const getModelsByProvider = (providerId: string) => {
    return useApiQuery<MCPModel[]>(
      `/api/llm/mcp/providers/${providerId}/models`,
      ['llm', 'mcp', 'providers', providerId, 'models'],
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
    // MCP Status
    mcpStatus,
    isLoadingMCPStatus,
    isErrorMCPStatus,
    errorMCPStatus,
    refetchMCPStatus,
    
    // Providers
    providers,
    isLoadingProviders,
    isErrorProviders,
    errorProviders,
    refetchProviders,
    getProviderDetails,
    getProviderStatus,
    getProviderUsage,
    createProvider,
    updateProvider,
    deleteProvider,
    testProvider,
    
    // Models
    models,
    isLoadingModels,
    isErrorModels,
    errorModels,
    refetchModels,
    getModelsByProvider
  };
}
