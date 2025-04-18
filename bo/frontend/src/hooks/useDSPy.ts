import { useApiQuery, useApiPost, useApiDelete } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface DSPyClient {
  client_id: string;
  name: string;
  description: string;
  status: string;
  llm_provider: string;
  cache_backend: string;
  created_at: string;
  updated_at: string;
  config: Record<string, any>;
}

export interface DSPyModule {
  module_id: string;
  client_id: string;
  name: string;
  type: string;
  description: string;
  status: string;
  created_at: string;
  updated_at: string;
  parameters: Record<string, any>;
  metrics: Record<string, number>;
}

export interface DSPyCircuitBreaker {
  name: string;
  state: 'CLOSED' | 'OPEN' | 'HALF_OPEN';
  failure_count: number;
  last_failure_time: string;
  reset_timeout: number;
  failure_threshold: number;
  recovery_timeout: number;
  module_name: string;
}

export interface DSPyAuditLog {
  id: string;
  timestamp: string;
  client_id: string;
  module_id: string;
  action: string;
  status: string;
  user_id?: string;
  input?: string;
  output?: string;
  error?: string;
  latency: number;
  metadata?: Record<string, any>;
}

export interface DSPyMetrics {
  client_id: string;
  period: string;
  request_count: number;
  error_count: number;
  latency: {
    avg: number;
    p50: number;
    p95: number;
    p99: number;
  };
  requests_by_module: Record<string, number>;
  errors_by_module: Record<string, number>;
  latency_by_module: Record<string, number>;
  requests_by_hour: Array<{
    hour: string;
    count: number;
  }>;
  errors_by_type: Record<string, number>;
}

export interface CreateDSPyClientParams {
  name: string;
  description?: string;
  llm_provider: string;
  cache_backend: string;
  config: Record<string, any>;
}

export interface UpdateDSPyClientParams {
  name?: string;
  description?: string;
  llm_provider?: string;
  cache_backend?: string;
  config?: Record<string, any>;
}

export interface ExecuteModuleParams {
  module_id: string;
  input: string | Record<string, any>;
  parameters?: Record<string, any>;
}

export interface OptimizeModuleParams {
  module_id: string;
  config: Record<string, any>;
}

/**
 * Hook for DSPy operations
 */
export function useDSPy() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Get all clients
  const {
    data: clients = [],
    isLoading: isLoadingClients,
    isError: isErrorClients,
    error: errorClients,
    refetch: refetchClients
  } = useApiQuery<DSPyClient[]>(
    '/api/llm/dspy/clients',
    ['dspy', 'clients'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch DSPy clients: ${error.message}`);
      }
    }
  );

  // Get client details
  const getClientDetails = (clientId: string) => {
    return useApiQuery<DSPyClient>(
      `/api/llm/dspy/clients/${clientId}`,
      ['dspy', 'clients', clientId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!clientId,
        onError: (error) => {
          showError(`Failed to fetch client details: ${error.message}`);
        }
      }
    );
  };

  // Create client
  const createClient = () => {
    return useApiPost<DSPyClient, CreateDSPyClientParams>(
      '/api/llm/dspy/clients',
      {
        onSuccess: (data) => {
          showSuccess(`Client "${data.name}" created successfully`);
          queryClient.invalidateQueries({ queryKey: ['dspy', 'clients'] });
        },
        onError: (error) => {
          showError(`Failed to create client: ${error.message}`);
        }
      }
    );
  };

  // Update client
  const updateClient = (clientId: string) => {
    return useApiPost<DSPyClient, UpdateDSPyClientParams>(
      `/api/llm/dspy/clients/${clientId}`,
      {
        onSuccess: (data) => {
          showSuccess(`Client "${data.name}" updated successfully`);
          queryClient.invalidateQueries({ queryKey: ['dspy', 'clients'] });
          queryClient.invalidateQueries({ queryKey: ['dspy', 'clients', clientId] });
        },
        onError: (error) => {
          showError(`Failed to update client: ${error.message}`);
        }
      }
    );
  };

  // Delete client
  const deleteClient = (clientId: string) => {
    return useApiDelete<{ success: boolean }, void>(
      `/api/llm/dspy/clients/${clientId}`,
      {
        onSuccess: () => {
          showSuccess('Client deleted successfully');
          queryClient.invalidateQueries({ queryKey: ['dspy', 'clients'] });
        },
        onError: (error) => {
          showError(`Failed to delete client: ${error.message}`);
        }
      }
    );
  };

  // Get modules by client
  const getModulesByClient = (clientId: string) => {
    return useApiQuery<DSPyModule[]>(
      `/api/llm/dspy/clients/${clientId}/modules`,
      ['dspy', 'clients', clientId, 'modules'],
      {
        staleTime: 60000, // 1 minute
        enabled: !!clientId,
        onError: (error) => {
          showError(`Failed to fetch client modules: ${error.message}`);
        }
      }
    );
  };

  // Get module details
  const getModuleDetails = (moduleId: string) => {
    return useApiQuery<DSPyModule>(
      `/api/llm/dspy/modules/${moduleId}`,
      ['dspy', 'modules', moduleId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!moduleId,
        onError: (error) => {
          showError(`Failed to fetch module details: ${error.message}`);
        }
      }
    );
  };

  // Execute module
  const executeModule = () => {
    return useApiPost<any, ExecuteModuleParams>(
      '/api/llm/dspy/execute',
      {
        onSuccess: (data) => {
          showSuccess('Module executed successfully');
        },
        onError: (error) => {
          showError(`Failed to execute module: ${error.message}`);
        }
      }
    );
  };

  // Optimize module
  const optimizeModule = () => {
    return useApiPost<any, OptimizeModuleParams>(
      '/api/llm/dspy/optimize',
      {
        onSuccess: (data) => {
          showSuccess('Module optimized successfully');
          queryClient.invalidateQueries({ queryKey: ['dspy', 'modules'] });
        },
        onError: (error) => {
          showError(`Failed to optimize module: ${error.message}`);
        }
      }
    );
  };

  // Get circuit breakers by client
  const getCircuitBreakersByClient = (clientId: string) => {
    return useApiQuery<DSPyCircuitBreaker[]>(
      `/api/llm/dspy/clients/${clientId}/circuit-breakers`,
      ['dspy', 'clients', clientId, 'circuit-breakers'],
      {
        staleTime: 30000, // 30 seconds
        enabled: !!clientId,
        refetchInterval: 30000, // Refetch every 30 seconds
        onError: (error) => {
          showError(`Failed to fetch circuit breakers: ${error.message}`);
        }
      }
    );
  };

  // Reset circuit breaker
  const resetCircuitBreaker = (clientId: string, breakerName: string) => {
    return useApiPost<{ success: boolean }, { breaker_name: string }>(
      `/api/llm/dspy/clients/${clientId}/circuit-breakers/reset`,
      {
        onSuccess: () => {
          showSuccess('Circuit breaker reset successfully');
          queryClient.invalidateQueries({ queryKey: ['dspy', 'clients', clientId, 'circuit-breakers'] });
        },
        onError: (error) => {
          showError(`Failed to reset circuit breaker: ${error.message}`);
        }
      }
    );
  };

  // Get audit logs by client
  const getAuditLogsByClient = (clientId: string, page: number = 1, pageSize: number = 20) => {
    return useApiQuery<{ total_count: number; page: number; page_size: number; logs: DSPyAuditLog[] }>(
      `/api/llm/dspy/clients/${clientId}/audit-logs?page=${page}&page_size=${pageSize}`,
      ['dspy', 'clients', clientId, 'audit-logs', page, pageSize],
      {
        staleTime: 60000, // 1 minute
        enabled: !!clientId,
        keepPreviousData: true,
        onError: (error) => {
          showError(`Failed to fetch audit logs: ${error.message}`);
        }
      }
    );
  };

  // Get metrics by client
  const getMetricsByClient = (clientId: string, period: string = 'day') => {
    return useApiQuery<DSPyMetrics>(
      `/api/llm/dspy/clients/${clientId}/metrics?period=${period}`,
      ['dspy', 'clients', clientId, 'metrics', period],
      {
        staleTime: 60000, // 1 minute
        enabled: !!clientId,
        refetchInterval: 60000, // Refetch every minute
        onError: (error) => {
          showError(`Failed to fetch metrics: ${error.message}`);
        }
      }
    );
  };

  return {
    // Clients
    clients,
    isLoadingClients,
    isErrorClients,
    errorClients,
    refetchClients,
    getClientDetails,
    createClient,
    updateClient,
    deleteClient,
    
    // Modules
    getModulesByClient,
    getModuleDetails,
    executeModule,
    optimizeModule,
    
    // Circuit Breakers
    getCircuitBreakersByClient,
    resetCircuitBreaker,
    
    // Audit Logs
    getAuditLogsByClient,
    
    // Metrics
    getMetricsByClient
  };
}
