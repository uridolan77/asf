import { useApiQuery, useApiPost, useApiDelete } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface Adapter {
  id: string;
  name: string;
  base_model: string;
  adapter_type: string;
  cl_strategy: string;
  task: string;
  rank: number;
  alpha: number;
  dropout: number;
  status: string;
  created_at: string;
  updated_at: string;
  metrics?: Record<string, number>;
  tasks?: string[];
  datasets?: string[];
}

export interface Task {
  id: string;
  adapter_id: string;
  task_type: string;
  status: string;
  progress: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  error?: string;
  parameters?: Record<string, any>;
  metrics?: Record<string, number>;
}

export interface Dataset {
  id: string;
  name: string;
  task: string;
  size: number;
  format: string;
  created_at: string;
  metrics?: Record<string, any>;
}

export interface CLStrategy {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface PEFTMethod {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface BaseModel {
  id: string;
  name: string;
  size: string;
  provider: string;
  description: string;
}

export interface CreateAdapterParams {
  name: string;
  base_model: string;
  adapter_type: string;
  cl_strategy: string;
  task: string;
  rank?: number;
  alpha?: number;
  dropout?: number;
  parameters?: Record<string, any>;
}

export interface TrainingParams {
  adapter_id: string;
  dataset_id: string;
  learning_rate?: number;
  batch_size?: number;
  epochs?: number;
  max_steps?: number;
  warmup_steps?: number;
  weight_decay?: number;
  use_8bit?: boolean;
  use_4bit?: boolean;
  gradient_checkpointing?: boolean;
  gradient_accumulation_steps?: number;
  parameters?: Record<string, any>;
}

export interface EvaluationParams {
  adapter_id: string;
  dataset_id: string;
  metrics: string[];
  batch_size?: number;
  max_samples?: number;
  parameters?: Record<string, any>;
}

export interface GenerationParams {
  adapter_id: string;
  prompt: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  parameters?: Record<string, any>;
}

/**
 * Hook for CL-PEFT operations
 */
export function useCLPEFT() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Get all adapters
  const {
    data: adapters = [],
    isLoading: isLoadingAdapters,
    isError: isErrorAdapters,
    error: errorAdapters,
    refetch: refetchAdapters
  } = useApiQuery<Adapter[]>(
    '/api/llm/cl-peft/adapters',
    ['cl-peft', 'adapters'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch adapters: ${error.message}`);
      }
    }
  );

  // Get adapter details
  const getAdapterDetails = (adapterId: string) => {
    return useApiQuery<Adapter>(
      `/api/llm/cl-peft/adapters/${adapterId}`,
      ['cl-peft', 'adapters', adapterId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!adapterId,
        onError: (error) => {
          showError(`Failed to fetch adapter details: ${error.message}`);
        }
      }
    );
  };

  // Create adapter
  const createAdapter = (params: CreateAdapterParams) => {
    return useApiPost<Adapter, CreateAdapterParams>(
      '/api/llm/cl-peft/adapters',
      {
        onSuccess: (data) => {
          showSuccess(`Adapter "${data.name}" created successfully`);
          queryClient.invalidateQueries({ queryKey: ['cl-peft', 'adapters'] });
        },
        onError: (error) => {
          showError(`Failed to create adapter: ${error.message}`);
        }
      }
    );
  };

  // Delete adapter
  const deleteAdapter = (adapterId: string) => {
    return useApiDelete<{ success: boolean }, void>(
      `/api/llm/cl-peft/adapters/${adapterId}`,
      {
        onSuccess: () => {
          showSuccess('Adapter deleted successfully');
          queryClient.invalidateQueries({ queryKey: ['cl-peft', 'adapters'] });
        },
        onError: (error) => {
          showError(`Failed to delete adapter: ${error.message}`);
        }
      }
    );
  };

  // Get all tasks
  const getTasks = (adapterId?: string) => {
    const url = adapterId 
      ? `/api/llm/cl-peft/adapters/${adapterId}/tasks` 
      : '/api/llm/cl-peft/tasks';
    
    return useApiQuery<Task[]>(
      url,
      adapterId ? ['cl-peft', 'adapters', adapterId, 'tasks'] : ['cl-peft', 'tasks'],
      {
        staleTime: 30000, // 30 seconds
        refetchOnWindowFocus: true,
        enabled: adapterId ? !!adapterId : true,
        onError: (error) => {
          showError(`Failed to fetch tasks: ${error.message}`);
        }
      }
    );
  };

  // Get task details
  const getTaskDetails = (taskId: string) => {
    return useApiQuery<Task>(
      `/api/llm/cl-peft/tasks/${taskId}`,
      ['cl-peft', 'tasks', taskId],
      {
        staleTime: 30000, // 30 seconds
        enabled: !!taskId,
        onError: (error) => {
          showError(`Failed to fetch task details: ${error.message}`);
        }
      }
    );
  };

  // Get all datasets
  const {
    data: datasets = [],
    isLoading: isLoadingDatasets,
    isError: isErrorDatasets,
    error: errorDatasets,
    refetch: refetchDatasets
  } = useApiQuery<Dataset[]>(
    '/api/llm/cl-peft/datasets',
    ['cl-peft', 'datasets'],
    {
      staleTime: 300000, // 5 minutes
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch datasets: ${error.message}`);
      }
    }
  );

  // Get all CL strategies
  const {
    data: clStrategies = [],
    isLoading: isLoadingCLStrategies,
    isError: isErrorCLStrategies,
    error: errorCLStrategies
  } = useApiQuery<CLStrategy[]>(
    '/api/llm/cl-peft/strategies',
    ['cl-peft', 'strategies'],
    {
      staleTime: 3600000, // 1 hour
      onError: (error) => {
        showError(`Failed to fetch CL strategies: ${error.message}`);
      }
    }
  );

  // Get all PEFT methods
  const {
    data: peftMethods = [],
    isLoading: isLoadingPEFTMethods,
    isError: isErrorPEFTMethods,
    error: errorPEFTMethods
  } = useApiQuery<PEFTMethod[]>(
    '/api/llm/cl-peft/peft-methods',
    ['cl-peft', 'peft-methods'],
    {
      staleTime: 3600000, // 1 hour
      onError: (error) => {
        showError(`Failed to fetch PEFT methods: ${error.message}`);
      }
    }
  );

  // Get all base models
  const {
    data: baseModels = [],
    isLoading: isLoadingBaseModels,
    isError: isErrorBaseModels,
    error: errorBaseModels
  } = useApiQuery<BaseModel[]>(
    '/api/llm/cl-peft/base-models',
    ['cl-peft', 'base-models'],
    {
      staleTime: 3600000, // 1 hour
      onError: (error) => {
        showError(`Failed to fetch base models: ${error.message}`);
      }
    }
  );

  // Train adapter
  const trainAdapter = (params: TrainingParams) => {
    return useApiPost<Task, TrainingParams>(
      '/api/llm/cl-peft/train',
      {
        onSuccess: (data) => {
          showSuccess('Training started successfully');
          queryClient.invalidateQueries({ queryKey: ['cl-peft', 'adapters', params.adapter_id] });
          queryClient.invalidateQueries({ queryKey: ['cl-peft', 'adapters', params.adapter_id, 'tasks'] });
          queryClient.invalidateQueries({ queryKey: ['cl-peft', 'tasks'] });
        },
        onError: (error) => {
          showError(`Failed to start training: ${error.message}`);
        }
      }
    );
  };

  // Evaluate adapter
  const evaluateAdapter = (params: EvaluationParams) => {
    return useApiPost<Task, EvaluationParams>(
      '/api/llm/cl-peft/evaluate',
      {
        onSuccess: (data) => {
          showSuccess('Evaluation started successfully');
          queryClient.invalidateQueries({ queryKey: ['cl-peft', 'adapters', params.adapter_id] });
          queryClient.invalidateQueries({ queryKey: ['cl-peft', 'adapters', params.adapter_id, 'tasks'] });
          queryClient.invalidateQueries({ queryKey: ['cl-peft', 'tasks'] });
        },
        onError: (error) => {
          showError(`Failed to start evaluation: ${error.message}`);
        }
      }
    );
  };

  // Generate text
  const generateText = (params: GenerationParams) => {
    return useApiPost<{ text: string }, GenerationParams>(
      '/api/llm/cl-peft/generate',
      {
        onError: (error) => {
          showError(`Text generation failed: ${error.message}`);
        }
      }
    );
  };

  return {
    // Adapters
    adapters,
    isLoadingAdapters,
    isErrorAdapters,
    errorAdapters,
    refetchAdapters,
    getAdapterDetails,
    createAdapter,
    deleteAdapter,
    
    // Tasks
    getTasks,
    getTaskDetails,
    
    // Datasets
    datasets,
    isLoadingDatasets,
    isErrorDatasets,
    errorDatasets,
    refetchDatasets,
    
    // CL strategies
    clStrategies,
    isLoadingCLStrategies,
    isErrorCLStrategies,
    errorCLStrategies,
    
    // PEFT methods
    peftMethods,
    isLoadingPEFTMethods,
    isErrorPEFTMethods,
    errorPEFTMethods,
    
    // Base models
    baseModels,
    isLoadingBaseModels,
    isErrorBaseModels,
    errorBaseModels,
    
    // Operations
    trainAdapter,
    evaluateAdapter,
    generateText
  };
}
