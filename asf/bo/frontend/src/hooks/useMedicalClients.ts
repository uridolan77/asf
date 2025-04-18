import { useApiQuery, useApiPost, useApiPut } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';

// Types
interface MedicalClient {
  id: string;
  name: string;
  description: string;
  status: string;
  last_checked: string;
  api_version: string;
  endpoints: string[];
  config?: Record<string, any>;
}

interface ClientStatus {
  status: string;
  last_checked: string;
  response_time?: number;
  error_message?: string;
}

interface ClientUsage {
  client_id: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  average_response_time: number;
  usage_by_day: Array<{
    date: string;
    requests: number;
  }>;
}

interface ClientConfig {
  api_key?: string;
  base_url?: string;
  timeout?: number;
  max_retries?: number;
  cache_ttl?: number;
  [key: string]: any;
}

/**
 * Hook for medical clients operations
 */
export function useMedicalClients() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();

  // Get all medical clients
  const {
    data: clients = [],
    isLoading: isLoadingClients,
    isError: isErrorClients,
    error: errorClients,
    refetch: refetchClients
  } = useApiQuery<MedicalClient[]>(
    '/api/medical/clients',
    ['medical', 'clients'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch medical clients: ${error.message}`);
      }
    }
  );

  // Get client details
  const getClientDetails = (clientId: string) => {
    return useApiQuery<MedicalClient>(
      `/api/medical/clients/${clientId}`,
      ['medical', 'clients', clientId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!clientId,
        onError: (error) => {
          showError(`Failed to fetch client details: ${error.message}`);
        }
      }
    );
  };

  // Get client status
  const getClientStatus = (clientId: string) => {
    return useApiQuery<ClientStatus>(
      `/api/medical/clients/${clientId}/status`,
      ['medical', 'clients', clientId, 'status'],
      {
        staleTime: 30000, // 30 seconds
        enabled: !!clientId,
        onError: (error) => {
          console.error(`Failed to fetch client status: ${error.message}`);
        }
      }
    );
  };

  // Get client usage
  const getClientUsage = (clientId: string, days: number = 30) => {
    return useApiQuery<ClientUsage>(
      `/api/medical/clients/${clientId}/usage?days=${days}`,
      ['medical', 'clients', clientId, 'usage', days],
      {
        staleTime: 5 * 60 * 1000, // 5 minutes
        enabled: !!clientId,
        onError: (error) => {
          showError(`Failed to fetch client usage: ${error.message}`);
        }
      }
    );
  };

  // Update client configuration
  const updateClientConfig = (clientId: string) => {
    return useApiPut<MedicalClient, ClientConfig>(
      `/api/medical/clients/${clientId}/config`,
      {
        onSuccess: (data) => {
          showSuccess('Client configuration updated successfully');
          queryClient.invalidateQueries({ queryKey: ['medical', 'clients'] });
          queryClient.invalidateQueries({ queryKey: ['medical', 'clients', clientId] });
        },
        onError: (error) => {
          showError(`Failed to update client configuration: ${error.message}`);
        }
      }
    );
  };

  // Test client connection
  const testClientConnection = (clientId: string) => {
    return useApiPost<ClientStatus, void>(
      `/api/medical/clients/${clientId}/test`,
      {
        onSuccess: (data) => {
          if (data.status === 'connected') {
            showSuccess('Client connection test successful');
          } else {
            showError(`Client connection test failed: ${data.error_message}`);
          }
          queryClient.invalidateQueries({ queryKey: ['medical', 'clients', clientId, 'status'] });
        },
        onError: (error) => {
          showError(`Client connection test failed: ${error.message}`);
        }
      }
    );
  };

  // Get client endpoints
  const getClientEndpoints = (clientId: string) => {
    return useApiQuery<string[]>(
      `/api/medical/clients/${clientId}/endpoints`,
      ['medical', 'clients', clientId, 'endpoints'],
      {
        staleTime: 60 * 60 * 1000, // 1 hour
        enabled: !!clientId,
        onError: (error) => {
          showError(`Failed to fetch client endpoints: ${error.message}`);
        }
      }
    );
  };

  // Test endpoint
  const testEndpoint = (clientId: string, endpoint: string) => {
    return useApiPost<any, void>(
      `/api/medical/clients/${clientId}/endpoints/${endpoint}/test`,
      {
        onSuccess: (data) => {
          showSuccess(`Endpoint ${endpoint} test successful`);
        },
        onError: (error) => {
          showError(`Endpoint ${endpoint} test failed: ${error.message}`);
        }
      }
    );
  };

  return {
    // Clients list
    clients,
    isLoadingClients,
    isErrorClients,
    errorClients,
    refetchClients,

    // Client details
    getClientDetails,

    // Client status
    getClientStatus,

    // Client usage
    getClientUsage,

    // Client configuration
    updateClientConfig,

    // Client testing
    testClientConnection,
    
    // Client endpoints
    getClientEndpoints,
    testEndpoint
  };
}
