import { useState, useEffect, useCallback, useRef } from 'react';
import { useFeatureFlags } from '../context/FeatureFlagContext';

/**
 * Hook for WebSocket communication with the document processing server
 */
export default function useWebSocket() {
  const [connected, setConnected] = useState<boolean>(false);
  const socketRef = useRef<WebSocket | null>(null);
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Event listeners
  const listenersRef = useRef<Record<string, Record<string, (message: any) => void>>>({
    progress: {},
    completed: {},
    failed: {},
    intermediate_result: {}
  });

  // Connect to WebSocket
  useEffect(() => {
    // Skip if using mock data
    if (useMockData) {
      return;
    }

    // Get WebSocket URL from environment or use default
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = import.meta.env.VITE_WS_URL || `${wsProtocol}//${window.location.host}/api/ws`;

    // Create WebSocket connection
    const socket = new WebSocket(wsUrl);

    // Set up event handlers
    socket.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    socket.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const { type, task_id, data } = message;

        // Call appropriate listeners
        if (type && task_id && listenersRef.current[type]) {
          const taskListeners = listenersRef.current[type][task_id];
          if (taskListeners) {
            taskListeners(data);
          }
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    // Store socket reference
    socketRef.current = socket;

    // Clean up on unmount
    return () => {
      socket.close();
    };
  }, [useMockData]);

  // Subscribe to task updates
  const subscribeToTask = useCallback(async (taskId: string) => {
    if (!socketRef.current || !connected) {
      throw new Error('WebSocket not connected');
    }

    socketRef.current.send(JSON.stringify({
      action: 'subscribe',
      task_id: taskId
    }));
  }, [connected]);

  // Unsubscribe from task updates
  const unsubscribeFromTask = useCallback(async (taskId: string) => {
    if (!socketRef.current || !connected) {
      throw new Error('WebSocket not connected');
    }

    socketRef.current.send(JSON.stringify({
      action: 'unsubscribe',
      task_id: taskId
    }));

    // Clean up listeners
    Object.keys(listenersRef.current).forEach(type => {
      delete listenersRef.current[type][taskId];
    });
  }, [connected]);

  // Register event listener
  const registerListener = useCallback((type: string, taskId: string, callback: (message: any) => void) => {
    if (!listenersRef.current[type]) {
      listenersRef.current[type] = {};
    }

    listenersRef.current[type][taskId] = callback;

    // Return unsubscribe function
    return () => {
      if (listenersRef.current[type]) {
        delete listenersRef.current[type][taskId];
      }
    };
  }, []);

  // Event listeners
  const onProgress = useCallback((taskId: string, callback: (message: any) => void) => {
    return registerListener('progress', taskId, callback);
  }, [registerListener]);

  const onCompleted = useCallback((taskId: string, callback: (message: any) => void) => {
    return registerListener('completed', taskId, callback);
  }, [registerListener]);

  const onFailed = useCallback((taskId: string, callback: (message: any) => void) => {
    return registerListener('failed', taskId, callback);
  }, [registerListener]);

  const onIntermediateResult = useCallback((taskId: string, callback: (message: any) => void) => {
    return registerListener('intermediate_result', taskId, callback);
  }, [registerListener]);

  // Mock implementation for development
  useEffect(() => {
    if (!useMockData) {
      return;
    }

    // Simulate WebSocket connection
    setConnected(true);

    // Clean up
    return () => {
      setConnected(false);
    };
  }, [useMockData]);

  // Mock subscribe function
  const mockSubscribeToTask = useCallback(async (taskId: string) => {
    if (!useMockData) {
      throw new Error('Not using mock data');
    }

    console.log(`Mock subscribed to task ${taskId}`);

    // Simulate progress updates
    let progress = 0;
    const stages = ['initializing', 'parsing', 'extracting_entities', 'extracting_relations', 'generating_summary', 'completed'];
    let currentStageIndex = 0;

    const interval = setInterval(() => {
      progress += 5;
      
      // Call progress listeners
      if (listenersRef.current.progress[taskId]) {
        listenersRef.current.progress[taskId]({
          progress: progress / 100,
          stage: stages[currentStageIndex]
        });
      }

      // Change stage
      if (progress % 20 === 0 && currentStageIndex < stages.length - 1) {
        currentStageIndex++;

        // Send intermediate result
        if (listenersRef.current.intermediate_result[taskId]) {
          listenersRef.current.intermediate_result[taskId]({
            stage: stages[currentStageIndex],
            result: {
              entity_count: Math.floor(Math.random() * 50) + 10,
              relation_count: Math.floor(Math.random() * 30) + 5
            }
          });
        }
      }

      // Complete
      if (progress >= 100) {
        clearInterval(interval);
        
        // Call completed listeners
        if (listenersRef.current.completed[taskId]) {
          listenersRef.current.completed[taskId]({
            task_id: taskId,
            status: 'completed',
            processing_time: 12.34,
            entity_count: 42,
            relation_count: 18
          });
        }
      }
    }, 500);

    // Clean up
    return () => {
      clearInterval(interval);
    };
  }, [useMockData]);

  // Mock unsubscribe function
  const mockUnsubscribeFromTask = useCallback(async (taskId: string) => {
    if (!useMockData) {
      throw new Error('Not using mock data');
    }

    console.log(`Mock unsubscribed from task ${taskId}`);

    // Clean up listeners
    Object.keys(listenersRef.current).forEach(type => {
      delete listenersRef.current[type][taskId];
    });
  }, [useMockData]);

  return {
    connected,
    subscribeToTask: useMockData ? mockSubscribeToTask : subscribeToTask,
    unsubscribeFromTask: useMockData ? mockUnsubscribeFromTask : unsubscribeFromTask,
    onProgress,
    onCompleted,
    onFailed,
    onIntermediateResult
  };
}
