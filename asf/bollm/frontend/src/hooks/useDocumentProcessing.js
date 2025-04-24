import { useState, useEffect, useCallback } from 'react';
import useWebSocket from './useWebSocket';
import apiService from '../services/api';

/**
 * Hook for tracking document processing progress.
 * @param {string} taskId - The task ID to track.
 * @returns {Object} Document processing hook.
 */
const useDocumentProcessing = (taskId) => {
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [status, setStatus] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [task, setTask] = useState(null);
  const [intermediateResults, setIntermediateResults] = useState({});

  const {
    connected,
    subscribeToTask,
    unsubscribeFromTask,
    onProgress,
    onCompleted,
    onFailed,
    onIntermediateResult
  } = useWebSocket();

  // Fetch initial task status
  useEffect(() => {
    if (!taskId) return;

    const fetchTask = async () => {
      try {
        const response = await apiService.documentProcessing.getTask(taskId);
        if (response.success) {
          const taskData = response.data;
          setTask(taskData);
          setStatus(taskData.status);
          setProgress(taskData.progress || 0);
          setStage(taskData.current_stage || '');

          if (taskData.status === 'completed') {
            fetchResults();
          } else if (taskData.status === 'failed') {
            setError(taskData.error_message || 'Processing failed');
          }
        } else {
          setError(response.error || 'Failed to fetch task');
        }
        setLoading(false);
      } catch (error) {
        console.error('Error fetching task:', error);
        setError('Failed to fetch task');
        setLoading(false);
      }
    };

    fetchTask();
  }, [taskId]);

  // Fetch results when task is completed
  const fetchResults = useCallback(async () => {
    if (!taskId) return;

    try {
      const response = await apiService.documentProcessing.getResults(taskId);
      if (response.success) {
        setResult(response.data);
      } else {
        setError(response.error || 'Failed to fetch results');
      }
    } catch (error) {
      console.error('Error fetching results:', error);
      setError('Failed to fetch results');
    }
  }, [taskId]);

  // Subscribe to WebSocket updates
  useEffect(() => {
    if (!taskId || !connected) return;

    // Subscribe to task updates
    subscribeToTask(taskId).catch(error => {
      console.error(`Error subscribing to task ${taskId}:`, error);
    });

    // Register handlers
    const progressHandler = (message) => {
      setProgress(message.progress * 100);
      setStage(message.stage);
    };

    const completedHandler = (message) => {
      setStatus('completed');
      setProgress(100);
      setStage('completed');
      fetchResults();
    };

    const failedHandler = (message) => {
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

    const intermediateResultHandler = (message) => {
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
  }, [taskId, connected, subscribeToTask, unsubscribeFromTask, onProgress, onCompleted, onFailed, onIntermediateResult, fetchResults]);

  return {
    progress,
    stage,
    status,
    result,
    error,
    loading,
    task,
    intermediateResults
  };
};

export default useDocumentProcessing;
