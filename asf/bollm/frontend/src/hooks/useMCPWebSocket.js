import { useState, useEffect, useCallback } from 'react';
import websocketService, { WS_STATUS } from '../services/websocket';

/**
 * Custom hook for using WebSockets with MCP providers
 * 
 * This hook provides real-time updates for MCP providers
 * including status changes, metrics, and events.
 */
export const useMCPWebSocket = (providerId) => {
  const [status, setStatus] = useState(websocketService.getStatus());
  const [isConnected, setIsConnected] = useState(websocketService.isConnected());
  const [statusUpdates, setStatusUpdates] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [events, setEvents] = useState([]);

  // Connect to WebSocket
  useEffect(() => {
    const connect = async () => {
      try {
        await websocketService.connect();
      } catch (error) {
        console.error('Error connecting to WebSocket:', error);
      }
    };

    connect();

    // Cleanup on unmount
    return () => {
      // Don't disconnect as other components might be using the WebSocket
    };
  }, []);

  // Subscribe to status updates
  useEffect(() => {
    const statusHandler = () => {
      setStatus(websocketService.getStatus());
      setIsConnected(websocketService.isConnected());
    };

    // Listen for WebSocket status changes
    const statusInterval = setInterval(statusHandler, 1000);

    return () => {
      clearInterval(statusInterval);
    };
  }, []);

  // Subscribe to provider-specific messages
  useEffect(() => {
    if (!providerId) return;

    // Handler for status updates
    const handleStatusUpdate = (message) => {
      setStatusUpdates((prev) => [message, ...prev].slice(0, 10));
    };

    // Handler for metrics updates
    const handleMetricsUpdate = (message) => {
      setMetrics(message);
    };

    // Handler for events
    const handleEvent = (message) => {
      setEvents((prev) => [message, ...prev].slice(0, 20));
    };

    // Subscribe to provider-specific messages
    const unsubscribeStatus = websocketService.onMessage('provider_status', handleStatusUpdate, providerId);
    const unsubscribeMetrics = websocketService.onMessage('provider_metrics', handleMetricsUpdate, providerId);
    const unsubscribeEvents = websocketService.onMessage('provider_event', handleEvent, providerId);

    // Subscribe to the provider
    if (websocketService.isConnected()) {
      websocketService.subscribeToTask(providerId).catch(error => {
        console.error(`Error subscribing to provider ${providerId}:`, error);
      });
    }

    return () => {
      // Unsubscribe from provider-specific messages
      unsubscribeStatus();
      unsubscribeMetrics();
      unsubscribeEvents();

      // Unsubscribe from the provider
      if (websocketService.isConnected()) {
        websocketService.unsubscribeFromTask(providerId).catch(error => {
          console.error(`Error unsubscribing from provider ${providerId}:`, error);
        });
      }
    };
  }, [providerId]);

  // Send a message to the WebSocket server
  const sendMessage = useCallback(async (message) => {
    try {
      await websocketService.send(message);
      return true;
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      return false;
    }
  }, []);

  // Request a status update for the provider
  const requestStatusUpdate = useCallback(async () => {
    if (!providerId) return false;

    return sendMessage({
      type: 'request_status',
      provider_id: providerId
    });
  }, [providerId, sendMessage]);

  // Request metrics for the provider
  const requestMetrics = useCallback(async () => {
    if (!providerId) return false;

    return sendMessage({
      type: 'request_metrics',
      provider_id: providerId
    });
  }, [providerId, sendMessage]);

  return {
    status,
    isConnected,
    statusUpdates,
    metrics,
    events,
    sendMessage,
    requestStatusUpdate,
    requestMetrics
  };
};
