import { useApiQuery, useApiPost } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface MLModel {
  id: string;
  name: string;
  type: string;
  version: string;
  description: string;
  status: string;
  created_at: string;
  updated_at: string;
  parameters?: Record<string, any>;
  metrics?: Record<string, number>;
}

export interface Adapter {
  id: string;
  name: string;
  base_model: string;
  adapter_type: string;
  task: string;
  rank: number;
  alpha: number;
  dropout: number;
  status: string;
  created_at: string;
  updated_at: string;
  metrics?: Record<string, number>;
}

export interface TrainingParams {
  model_id: string;
  dataset_id: string;
  task: string;
  adapter_type?: string;
  rank?: number;
  alpha?: number;
  dropout?: number;
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
}

export interface EvaluationParams {
  model_id: string;
  adapter_id?: string;
  dataset_id: string;
  metrics: string[];
  batch_size?: number;
  max_samples?: number;
}

export interface EvaluationResult {
  model_id: string;
  adapter_id?: string;
  dataset_id: string;
  metrics: Record<string, number>;
  examples: Array<{
    input: string;
    expected: string;
    actual: string;
    score: number;
  }>;
  confusion_matrix?: Record<string, Record<string, number>>;
}

export interface DSPyModule {
  id: string;
  name: string;
  type: string;
  description: string;
  status: string;
  created_at: string;
  updated_at: string;
  parameters?: Record<string, any>;
  metrics?: Record<string, number>;
}

/**
 * Hook for ML models operations
 */
export function useMLModels() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Get all models
  const {
    data: models = [],
    isLoading: isLoadingModels,
    isError: isErrorModels,
    error: errorModels,
    refetch: refetchModels
  } = useApiQuery<MLModel[]>(
    '/api/ml/models',
    ['ml', 'models'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch ML models: ${error.message}`);
      }
    }
  );

  // Get model details
  const getModelDetails = (modelId: string) => {
    return useApiQuery<MLModel>(
      `/api/ml/models/${modelId}`,
      ['ml', 'models', modelId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!modelId,
        onError: (error) => {
          showError(`Failed to fetch model details: ${error.message}`);
        }
      }
    );
  };

  // Get all adapters
  const {
    data: adapters = [],
    isLoading: isLoadingAdapters,
    isError: isErrorAdapters,
    error: errorAdapters,
    refetch: refetchAdapters
  } = useApiQuery<Adapter[]>(
    '/api/ml/adapters',
    ['ml', 'adapters'],
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
      `/api/ml/adapters/${adapterId}`,
      ['ml', 'adapters', adapterId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!adapterId,
        onError: (error) => {
          showError(`Failed to fetch adapter details: ${error.message}`);
        }
      }
    );
  };

  // Train model
  const trainModel = (params: TrainingParams) => {
    return useApiPost<{ task_id: string }, TrainingParams>(
      '/api/ml/train',
      {
        onSuccess: (data) => {
          showSuccess('Training started successfully');
        },
        onError: (error) => {
          showError(`Failed to start training: ${error.message}`);
        }
      }
    );
  };

  // Evaluate model
  const evaluateModel = (params: EvaluationParams) => {
    return useApiPost<EvaluationResult, EvaluationParams>(
      '/api/ml/evaluate',
      {
        onSuccess: (data) => {
          showSuccess('Evaluation completed successfully');
        },
        onError: (error) => {
          showError(`Failed to evaluate model: ${error.message}`);
        }
      }
    );
  };

  // Get all DSPy modules
  const {
    data: dspyModules = [],
    isLoading: isLoadingDSPyModules,
    isError: isErrorDSPyModules,
    error: errorDSPyModules,
    refetch: refetchDSPyModules
  } = useApiQuery<DSPyModule[]>(
    '/api/ml/dspy/modules',
    ['ml', 'dspy', 'modules'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch DSPy modules: ${error.message}`);
      }
    }
  );

  // Get DSPy module details
  const getDSPyModuleDetails = (moduleId: string) => {
    return useApiQuery<DSPyModule>(
      `/api/ml/dspy/modules/${moduleId}`,
      ['ml', 'dspy', 'modules', moduleId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!moduleId,
        onError: (error) => {
          showError(`Failed to fetch DSPy module details: ${error.message}`);
        }
      }
    );
  };

  // Run DSPy module
  const runDSPyModule = (moduleId: string, input: Record<string, any>) => {
    return useApiPost<any, Record<string, any>>(
      `/api/ml/dspy/modules/${moduleId}/run`,
      {
        onSuccess: (data) => {
          showSuccess('DSPy module executed successfully');
        },
        onError: (error) => {
          showError(`Failed to run DSPy module: ${error.message}`);
        }
      }
    );
  };

  return {
    // Models
    models,
    isLoadingModels,
    isErrorModels,
    errorModels,
    refetchModels,
    getModelDetails,

    // Adapters
    adapters,
    isLoadingAdapters,
    isErrorAdapters,
    errorAdapters,
    refetchAdapters,
    getAdapterDetails,

    // Training and evaluation
    trainModel,
    evaluateModel,

    // DSPy modules
    dspyModules,
    isLoadingDSPyModules,
    isErrorDSPyModules,
    errorDSPyModules,
    refetchDSPyModules,
    getDSPyModuleDetails,
    runDSPyModule
  };
}
