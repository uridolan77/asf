import { useApiQuery, useApiPost, useApiDelete } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useState, useEffect } from 'react';
import useWebSocket from './useWebSocket';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
interface Document {
  id: string;
  title: string;
  file_name: string;
  file_type: string;
  upload_date: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  page_count?: number;
  error_message?: string;
}

interface DocumentDetails extends Document {
  content?: string;
  entities?: Array<{
    id: string;
    text: string;
    label: string;
    start: number;
    end: number;
    confidence?: number;
  }>;
  relations?: Array<{
    id: string;
    relation: string;
    head: string;
    head_id: string;
    tail: string;
    tail_id: string;
    confidence?: number;
  }>;
  sections?: Array<{
    id: string;
    title: string;
    content: string;
    start: number;
    end: number;
  }>;
  references?: Array<{
    id: string;
    text: string;
    doi?: string;
    url?: string;
  }>;
  summary?: {
    abstract?: string;
    key_findings?: string;
    conclusion?: string;
  };
}

interface ProcessingOptions {
  extract_entities?: boolean;
  extract_relations?: boolean;
  detect_sections?: boolean;
  parse_references?: boolean;
  use_biomedlm?: boolean;
  use_enhanced_synthesizer?: boolean;
  use_streaming?: boolean;
  use_parallel?: boolean;
}

interface TaskStatus {
  task_id: string;
  status: string;
  file_name: string;
  created_at: string;
  current_stage?: string;
  progress: number;
  entity_count?: number;
  relation_count?: number;
  error_message?: string;
  error_details?: Record<string, any>;
  processing_time?: number;
}

interface ProcessingResult {
  task_id: string;
  status: string;
  processing_time: number;
  entity_count: number;
  relation_count: number;
  results: DocumentDetails;
}

/**
 * Hook for document processing operations
 */
export function useDocumentProcessing() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Get all documents
  const {
    data: documents = [],
    isLoading: isLoadingDocuments,
    isError: isErrorDocuments,
    error: errorDocuments,
    refetch: refetchDocuments
  } = useApiQuery<Document[]>(
    '/api/document-processing/documents',
    ['document-processing', 'documents'],
    {
      staleTime: 30000, // 30 seconds
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch documents: ${error.message}`);
      }
    }
  );

  // Get document details
  const getDocumentDetails = (documentId: string) => {
    return useApiQuery<DocumentDetails>(
      `/api/document-processing/documents/${documentId}`,
      ['document-processing', 'documents', documentId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!documentId,
        onError: (error) => {
          showError(`Failed to fetch document details: ${error.message}`);
        }
      }
    );
  };

  // Upload document
  const uploadDocument = () => {
    return useApiPost<Document, FormData>(
      '/api/document-processing/upload',
      {
        onSuccess: (data) => {
          showSuccess('Document uploaded successfully');
          queryClient.invalidateQueries({ queryKey: ['document-processing', 'documents'] });
        },
        onError: (error) => {
          showError(`Failed to upload document: ${error.message}`);
        }
      }
    );
  };

  // Process document
  const processDocument = (documentId: string, options: ProcessingOptions) => {
    return useApiPost<any, ProcessingOptions>(
      `/api/document-processing/documents/${documentId}/process`,
      {
        onSuccess: (data) => {
          showSuccess('Document processing started');
          queryClient.invalidateQueries({ queryKey: ['document-processing', 'documents'] });
          queryClient.invalidateQueries({ queryKey: ['document-processing', 'documents', documentId] });
        },
        onError: (error) => {
          showError(`Failed to process document: ${error.message}`);
        }
      }
    );
  };

  // Delete document
  const deleteDocument = (documentId: string) => {
    return useApiDelete<void, void>(
      `/api/document-processing/documents/${documentId}`,
      {
        onSuccess: () => {
          showSuccess('Document deleted successfully');
          queryClient.invalidateQueries({ queryKey: ['document-processing', 'documents'] });
        },
        onError: (error) => {
          showError(`Failed to delete document: ${error.message}`);
        }
      }
    );
  };

  // Get processing status
  const getProcessingStatus = (documentId: string) => {
    return useApiQuery<{ status: string; progress: number; message?: string }>(
      `/api/document-processing/documents/${documentId}/status`,
      ['document-processing', 'documents', documentId, 'status'],
      {
        staleTime: 5000, // 5 seconds
        refetchInterval: (data) => {
          // Refetch every 2 seconds if status is 'processing', otherwise stop polling
          return data?.status === 'processing' ? 2000 : false;
        },
        enabled: !!documentId,
        onError: (error) => {
          console.error(`Failed to fetch processing status: ${error.message}`);
        }
      }
    );
  };

  // Export document
  const exportDocument = (documentId: string, format: string) => {
    return useApiPost<{ url: string }, { format: string }>(
      `/api/document-processing/documents/${documentId}/export`,
      {
        onSuccess: (data) => {
          showSuccess(`Document exported successfully as ${format}`);
        },
        onError: (error) => {
          showError(`Failed to export document: ${error.message}`);
        }
      }
    );
  };

  // Get processing results
  const getProcessingResults = (taskId: string) => {
    return useApiQuery<ProcessingResult>(
      `/api/document-processing/results/${taskId}`,
      ['document-processing', 'results', taskId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!taskId,
        onError: (error) => {
          showError(`Failed to fetch processing results: ${error.message}`);
        }
      }
    );
  };

  // Get processing settings
  const getProcessingSettings = () => {
    return useApiQuery<ProcessingOptions>(
      '/api/document-processing/settings',
      ['document-processing', 'settings'],
      {
        staleTime: 60 * 60 * 1000, // 1 hour
        onError: (error) => {
          console.error(`Failed to fetch processing settings: ${error.message}`);
        }
      }
    );
  };

  // Update processing settings
  const updateProcessingSettings = () => {
    return useApiPost<ProcessingOptions, ProcessingOptions>(
      '/api/document-processing/settings',
      {
        onSuccess: (data) => {
          showSuccess('Processing settings updated successfully');
          queryClient.invalidateQueries({ queryKey: ['document-processing', 'settings'] });
        },
        onError: (error) => {
          showError(`Failed to update processing settings: ${error.message}`);
        }
      }
    );
  };

  return {
    // Documents list
    documents,
    isLoadingDocuments,
    isErrorDocuments,
    errorDocuments,
    refetchDocuments,

    // Document details
    getDocumentDetails,

    // Operations
    uploadDocument,
    processDocument,
    deleteDocument,
    getProcessingStatus,
    getProcessingResults,
    getProcessingSettings,
    updateProcessingSettings,
    exportDocument
  };
}

/**
 * Hook for tracking document processing progress.
 * @param taskId - The task ID to track.
 * @returns Document processing hook.
 */
export function useDocumentProcessingTask(taskId: string | null) {
  const [progress, setProgress] = useState<number>(0);
  const [stage, setStage] = useState<string>('');
  const [status, setStatus] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [task, setTask] = useState<any>(null);
  const [intermediateResults, setIntermediateResults] = useState<Record<string, any>>({});

  const queryClient = useQueryClient();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // WebSocket connection
  const {
    connected,
    subscribeToTask,
    unsubscribeFromTask,
    onProgress,
    onCompleted,
    onFailed,
    onIntermediateResult
  } = useWebSocket();

  // Get task status
  const {
    data: taskData,
    isLoading: isLoadingTask,
    isError: isErrorTask,
    error: taskError,
    refetch: refetchTask
  } = useApiQuery<TaskStatus>(
    `/api/document-processing/tasks/${taskId}`,
    ['document-processing', 'tasks', taskId],
    {
      staleTime: 5000, // 5 seconds
      enabled: !!taskId && !connected, // Only fetch if WebSocket is not connected
      refetchInterval: (data) => {
        // Refetch every 2 seconds if status is 'processing', otherwise stop polling
        return data?.status === 'processing' ? 2000 : false;
      },
      onSuccess: (data) => {
        setTask(data);
        setStatus(data.status);
        setProgress(data.progress * 100);
        setStage(data.current_stage || '');

        if (data.status === 'completed') {
          // Fetch results
          queryClient.invalidateQueries({ queryKey: ['document-processing', 'results', taskId] });
        } else if (data.status === 'failed') {
          setError(data.error_message || 'Processing failed');
        }
      },
      onError: (error) => {
        console.error(`Failed to fetch task status: ${error.message}`);
        setError(`Failed to fetch task status: ${error.message}`);
      }
    }
  );

  // Get processing results
  const {
    data: resultData,
    isLoading: isLoadingResult,
    isError: isErrorResult,
    error: resultError
  } = useApiQuery<ProcessingResult>(
    `/api/document-processing/results/${taskId}`,
    ['document-processing', 'results', taskId],
    {
      staleTime: 60000, // 1 minute
      enabled: !!taskId && status === 'completed',
      onSuccess: (data) => {
        setResult(data);
      },
      onError: (error) => {
        console.error(`Failed to fetch processing results: ${error.message}`);
        setError(`Failed to fetch processing results: ${error.message}`);
      }
    }
  );

  // Subscribe to WebSocket updates
  useEffect(() => {
    if (!taskId || !connected) return;

    // Subscribe to task updates
    subscribeToTask(taskId).catch(error => {
      console.error(`Error subscribing to task ${taskId}:`, error);
    });

    // Register handlers
    const progressHandler = (message: any) => {
      setProgress(message.progress * 100);
      setStage(message.stage);
    };

    const completedHandler = (message: any) => {
      setStatus('completed');
      setProgress(100);
      setStage('completed');
      // Invalidate the results query to trigger a refetch
      queryClient.invalidateQueries({ queryKey: ['document-processing', 'results', taskId] });
    };

    const failedHandler = (message: any) => {
      setStatus('failed');

      // Store detailed error information
      const errorMessage = message.error || 'Processing failed';
      const errorDetails = message.details || {};

      // Create a more detailed error message if details are available
      let detailedError = errorMessage;
      if (errorDetails && Object.keys(errorDetails).length > 0) {
        detailedError = `${errorMessage} (${errorDetails.error_type || 'Unknown error type'})`;
      }

      setError(detailedError);

      // Store error details in the task object for display in logs
      setTask(prev => ({
        ...prev,
        error_message: detailedError,
        error_details: errorDetails
      }));
    };

    const intermediateResultHandler = (message: any) => {
      setIntermediateResults(prev => ({
        ...prev,
        [message.stage]: message.result
      }));
    };

    // Add event listeners
    const unsubscribeProgress = onProgress(taskId, progressHandler);
    const unsubscribeCompleted = onCompleted(taskId, completedHandler);
    const unsubscribeFailed = onFailed(taskId, failedHandler);
    const unsubscribeIntermediateResult = onIntermediateResult(taskId, intermediateResultHandler);

    // Cleanup
    return () => {
      unsubscribeProgress();
      unsubscribeCompleted();
      unsubscribeFailed();
      unsubscribeIntermediateResult();
      unsubscribeFromTask(taskId).catch(error => {
        console.error(`Error unsubscribing from task ${taskId}:`, error);
      });
    };
  }, [taskId, connected, subscribeToTask, unsubscribeFromTask, onProgress, onCompleted, onFailed, onIntermediateResult, queryClient]);

  // Update loading state
  useEffect(() => {
    setLoading(isLoadingTask || (status === 'completed' && isLoadingResult));
  }, [isLoadingTask, isLoadingResult, status]);

  return {
    progress,
    stage,
    status,
    result,
    error,
    loading,
    task,
    intermediateResults,
    refetchTask
  };
}
