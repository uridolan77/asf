import { useApiQuery, useApiPost, useApiPut, useApiDelete } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';

// Types
interface LLMStatus {
  status: string;
  components: Record<string, any>;
}

interface MCPProvider {
  id: string;
  name: string;
  transport_type: string;
  status: string;
  config: Record<string, any>;
}

interface Model {
  id: string;
  name: string;
  provider_id: string;
  model_type: string;
  capabilities: string[];
  config: Record<string, any>;
}

interface GenerateRequest {
  model_id: string;
  provider_id: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  stop_sequences?: string[];
}

interface GenerateResponse {
  model_id: string;
  provider_id: string;
  prompt: string;
  completion: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  finish_reason: string;
}

/**
 * Hook for LLM-related operations
 */
export function useLLM() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();

  // Get LLM status
  const {
    data: llmStatus,
    isLoading: isLoadingStatus,
    isError: isErrorStatus,
    error: errorStatus,
    refetch: refetchStatus
  } = useApiQuery<LLMStatus>(
    '/api/llm/status',
    ['llm', 'status'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        console.error('Failed to fetch LLM status:', error);
      }
    }
  );

  // Get MCP providers
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
        console.error('Failed to fetch MCP providers:', error);
      }
    }
  );

  // Create MCP provider
  const {
    mutate: createProvider,
    isPending: isCreatingProvider,
    isError: isErrorCreatingProvider,
    error: errorCreatingProvider
  } = useApiPost<MCPProvider, Omit<MCPProvider, 'id'>>(
    '/api/llm/mcp/providers',
    {
      onSuccess: (data) => {
        showSuccess('Provider created successfully');
        queryClient.invalidateQueries({ queryKey: ['llm', 'mcp', 'providers'] });
      },
      onError: (error) => {
        showError(`Failed to create provider: ${error.message}`);
      }
    }
  );

  // Update MCP provider
  const {
    mutate: updateProvider,
    isPending: isUpdatingProvider,
    isError: isErrorUpdatingProvider,
    error: errorUpdatingProvider
  } = useApiPut<MCPProvider, MCPProvider>(
    '/api/llm/mcp/providers',
    {
      onSuccess: (data) => {
        showSuccess('Provider updated successfully');
        queryClient.invalidateQueries({ queryKey: ['llm', 'mcp', 'providers'] });
      },
      onError: (error) => {
        showError(`Failed to update provider: ${error.message}`);
      }
    }
  );

  // Delete MCP provider
  const {
    mutate: deleteProvider,
    isPending: isDeletingProvider,
    isError: isErrorDeletingProvider,
    error: errorDeletingProvider
  } = useApiDelete<void, { id: string }>(
    '/api/llm/mcp/providers',
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

  // Get models
  const getModels = (providerId?: string) => {
    return useApiQuery<Model[]>(
      `/api/llm/gateway/models${providerId ? `?provider_id=${providerId}` : ''}`,
      ['llm', 'models', providerId || 'all'],
      {
        staleTime: 60000, // 1 minute
        refetchOnWindowFocus: true,
        onError: (error) => {
          console.error('Failed to fetch models:', error);
        }
      }
    );
  };

  // Generate text
  const {
    mutate: generateText,
    isPending: isGenerating,
    isError: isErrorGenerating,
    error: errorGenerating
  } = useApiPost<GenerateResponse, GenerateRequest>(
    '/api/llm/gateway/generate',
    {
      onError: (error) => {
        showError(`Failed to generate text: ${error.message}`);
      }
    }
  );

  return {
    // Status
    llmStatus,
    isLoadingStatus,
    isErrorStatus,
    errorStatus,
    refetchStatus,

    // Providers
    providers,
    isLoadingProviders,
    isErrorProviders,
    errorProviders,
    refetchProviders,
    createProvider,
    isCreatingProvider,
    isErrorCreatingProvider,
    errorCreatingProvider,
    updateProvider,
    isUpdatingProvider,
    isErrorUpdatingProvider,
    errorUpdatingProvider,
    deleteProvider,
    isDeletingProvider,
    isErrorDeletingProvider,
    errorDeletingProvider,

    // Models
    getModels,

    // Generation
    generateText,
    isGenerating,
    isErrorGenerating,
    errorGenerating
  };
}
