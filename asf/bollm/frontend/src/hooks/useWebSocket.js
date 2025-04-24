import { useState, useEffect, useCallback } from 'react';
import websocketService, { WS_STATUS } from '../services/websocket';

/**
 * Hook for using WebSocket connections.
 * @returns {Object} WebSocket hook.
 */
const useWebSocket = () => {
  const [status, setStatus] = useState(websocketService.getStatus());
  const [connected, setConnected] = useState(websocketService.isConnected());

  // Update status when WebSocket status changes
  useEffect(() => {
    const handleStatusChange = () => {
      const newStatus = websocketService.getStatus();
      setStatus(newStatus);
      setConnected(newStatus === WS_STATUS.OPEN);
    };

    // Register handlers for WebSocket events
    const handleOpen = () => handleStatusChange();
    const handleClose = () => handleStatusChange();
    const handleError = () => handleStatusChange();

    // Add event listeners
    websocketService.onMessage('open', handleOpen);
    websocketService.onMessage('close', handleClose);
    websocketService.onMessage('error', handleError);

    // Initial connection
    if (!websocketService.isConnected()) {
      websocketService.connect().catch(error => {
        console.error('Error connecting to WebSocket:', error);
      });
    }

    // Cleanup
    return () => {
      // Remove event listeners
      websocketService.onMessage('open', handleOpen);
      websocketService.onMessage('close', handleClose);
      websocketService.onMessage('error', handleError);
    };
  }, []);

  /**
   * Connect to the WebSocket server.
   * @returns {Promise<void>} A promise that resolves when the connection is established.
   */
  const connect = useCallback(async () => {
    try {
      await websocketService.connect();
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      throw error;
    }
  }, []);

  /**
   * Disconnect from the WebSocket server.
   */
  const disconnect = useCallback(() => {
    websocketService.disconnect();
  }, []);

  /**
   * Subscribe to a task.
   * @param {string} taskId - The task ID to subscribe to.
   * @returns {Promise<void>} A promise that resolves when the subscription is complete.
   */
  const subscribeToTask = useCallback(async (taskId) => {
    try {
      await websocketService.subscribeToTask(taskId);
    } catch (error) {
      console.error(`Error subscribing to task ${taskId}:`, error);
      throw error;
    }
  }, []);

  /**
   * Unsubscribe from a task.
   * @param {string} taskId - The task ID to unsubscribe from.
   * @returns {Promise<void>} A promise that resolves when the unsubscription is complete.
   */
  const unsubscribeFromTask = useCallback(async (taskId) => {
    try {
      await websocketService.unsubscribeFromTask(taskId);
    } catch (error) {
      console.error(`Error unsubscribing from task ${taskId}:`, error);
      throw error;
    }
  }, []);

  /**
   * Register a handler for progress updates.
   * @param {string} taskId - The task ID to track progress for.
   * @param {Function} handler - The handler function.
   * @returns {Function} A function to unregister the handler.
   */
  const onProgress = useCallback((taskId, handler) => {
    return websocketService.onProgress(taskId, handler);
  }, []);

  /**
   * Register a handler for task completion.
   * @param {string} taskId - The task ID to track completion for.
   * @param {Function} handler - The handler function.
   * @returns {Function} A function to unregister the handler.
   */
  const onCompleted = useCallback((taskId, handler) => {
    return websocketService.onCompleted(taskId, handler);
  }, []);

  /**
   * Register a handler for task failure.
   * @param {string} taskId - The task ID to track failure for.
   * @param {Function} handler - The handler function.
   * @returns {Function} A function to unregister the handler.
   */
  const onFailed = useCallback((taskId, handler) => {
    return websocketService.onFailed(taskId, handler);
  }, []);

  /**
   * Register a handler for intermediate results.
   * @param {string} taskId - The task ID to track intermediate results for.
   * @param {Function} handler - The handler function.
   * @returns {Function} A function to unregister the handler.
   */
  const onIntermediateResult = useCallback((taskId, handler) => {
    return websocketService.onIntermediateResult(taskId, handler);
  }, []);

  return {
    status,
    connected,
    connect,
    disconnect,
    subscribeToTask,
    unsubscribeFromTask,
    onProgress,
    onCompleted,
    onFailed,
    onIntermediateResult
  };
};

export default useWebSocket;
