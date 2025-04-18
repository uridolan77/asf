import { useState, useEffect, useCallback, useRef } from 'react';
import { useSnackbar } from 'notistack';
import { useAuth } from '../contexts/AuthContext';

// WebSocket connection states
const CONNECTION_STATES = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  RECONNECTING: 'reconnecting',
  FAILED: 'failed'
};

// Reconnection settings
const RECONNECT_INTERVAL = 2000; // Start with 2 seconds
const MAX_RECONNECT_INTERVAL = 30000; // Max 30 seconds
const RECONNECT_DECAY = 1.5; // Exponential backoff factor
const MAX_RECONNECT_ATTEMPTS = 10;

/**
 * Custom hook for MCP WebSocket connection with reconnection logic.
 *
 * @param {string} providerId - Provider ID to subscribe to
 * @returns {Object} WebSocket connection state and methods
 */
export const useMCPWebSocket = (providerId) => {
  const { enqueueSnackbar } = useSnackbar();
  const { getAccessToken } = useAuth();

  // State
  const [connectionState, setConnectionState] = useState(CONNECTION_STATES.DISCONNECTED);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [status, setStatus] = useState(null);
  const [events, setEvents] = useState([]);
  const [statusUpdates, setStatusUpdates] = useState([]);

  // Refs to persist across renders
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectIntervalRef = useRef(RECONNECT_INTERVAL);
  const messageQueueRef = useRef([]);
  const clientIdRef = useRef(null);

  // Keep track of subscriptions
  const subscriptionsRef = useRef(new Set());

  // Add provider to subscriptions if provided
  useEffect(() => {
    if (providerId) {
      subscriptionsRef.current.add(providerId);
    }
  }, [providerId]);

  /**
   * Connect to the WebSocket server
   */
  const connect = useCallback(async () => {
    // Clean up any existing connection
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    try {
      setConnectionState(CONNECTION_STATES.CONNECTING);

      // Get access token
      const token = await getAccessToken();
      if (!token) {
        throw new Error('No access token available');
      }

      // Create WebSocket connection
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const clientId = clientIdRef.current || `client-${Math.random().toString(36).substring(2, 15)}`;
      clientIdRef.current = clientId;

      const url = `${protocol}//${host}/api/ws/mcp/${clientId}?token=${token}`;
      const socket = new WebSocket(url);

      socket.onopen = () => {
        console.log('WebSocket connection established');
        setConnectionState(CONNECTION_STATES.CONNECTED);
        setIsConnected(true);
        reconnectAttemptsRef.current = 0;
        reconnectIntervalRef.current = RECONNECT_INTERVAL;

        // Process any queued messages
        while (messageQueueRef.current.length > 0) {
          const message = messageQueueRef.current.shift();
          sendMessage(message);
        }

        // Resubscribe to providers
        subscriptionsRef.current.forEach(id => {
          sendMessage({
            type: 'subscribe',
            provider_id: id
          });
        });
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setLastMessage(message);

          // Handle different message types
          switch (message.type) {
            case 'provider_status':
              if (message.provider_id === providerId) {
                setStatus(message.data);
                setStatusUpdates((prev) => [message.data, ...prev].slice(0, 10));
              }
              break;

            case 'provider_metrics':
              if (message.provider_id === providerId) {
                setMetrics(message.data);
              }
              break;

            case 'provider_event':
              if (message.provider_id === providerId) {
                setEvents(prev => [message.data, ...prev].slice(0, 100)); // Keep last 100 events
              }
              break;

            case 'heartbeat':
              // Respond to heartbeat
              sendMessage({
                type: 'heartbeat_ack',
                timestamp: new Date().toISOString()
              });
              break;

            case 'token_refresh_required':
              // Token is about to expire, refresh it
              refreshConnection();
              break;

            case 'connection_established':
              console.log(`Connection established: ${message.client_id}`);
              if (message.reconnected) {
                enqueueSnackbar('Reconnected to server', { variant: 'success' });
              }
              break;

            default:
              // Handle other message types
              break;
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      socket.onclose = (event) => {
        setIsConnected(false);

        if (event.wasClean) {
          console.log(`WebSocket connection closed cleanly, code=${event.code}, reason=${event.reason}`);
          setConnectionState(CONNECTION_STATES.DISCONNECTED);
        } else {
          console.error('WebSocket connection died');
          setConnectionState(CONNECTION_STATES.RECONNECTING);
          scheduleReconnect();
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionState(CONNECTION_STATES.RECONNECTING);
      };

      socketRef.current = socket;

    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      setConnectionState(CONNECTION_STATES.FAILED);
      scheduleReconnect();
    }
  }, [enqueueSnackbar, getAccessToken, providerId]);

  /**
   * Schedule a reconnection attempt
   */
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    reconnectAttemptsRef.current += 1;

    if (reconnectAttemptsRef.current > MAX_RECONNECT_ATTEMPTS) {
      setConnectionState(CONNECTION_STATES.FAILED);
      enqueueSnackbar('Failed to connect to server after multiple attempts', {
        variant: 'error',
        action: (key) => (
          <button onClick={() => {
            reconnectAttemptsRef.current = 0;
            reconnectIntervalRef.current = RECONNECT_INTERVAL;
            connect();
          }}>
            Retry
          </button>
        )
      });
      return;
    }

    // Exponential backoff
    const timeout = Math.min(
      reconnectIntervalRef.current * Math.pow(RECONNECT_DECAY, reconnectAttemptsRef.current - 1),
      MAX_RECONNECT_INTERVAL
    );

    console.log(`Scheduling reconnect in ${timeout}ms (attempt ${reconnectAttemptsRef.current})`);

    reconnectTimeoutRef.current = setTimeout(() => {
      setConnectionState(CONNECTION_STATES.RECONNECTING);
      connect();
    }, timeout);
  }, [connect, enqueueSnackbar]);

  /**
   * Refresh the WebSocket connection (e.g., when token is about to expire)
   */
  const refreshConnection = useCallback(async () => {
    console.log('Refreshing WebSocket connection');
    connect();
  }, [connect]);

  /**
   * Send a message to the WebSocket server
   */
  const sendMessage = useCallback((message) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      // Queue message for later
      messageQueueRef.current.push(message);
      return false;
    }

    try {
      socketRef.current.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      messageQueueRef.current.push(message);
      return false;
    }
  }, []);

  /**
   * Subscribe to a provider
   */
  const subscribeToProvider = useCallback((id) => {
    if (!id) return;

    subscriptionsRef.current.add(id);
    sendMessage({
      type: 'subscribe',
      provider_id: id
    });
  }, [sendMessage]);

  /**
   * Unsubscribe from a provider
   */
  const unsubscribeFromProvider = useCallback((id) => {
    if (!id) return;

    subscriptionsRef.current.delete(id);
    sendMessage({
      type: 'unsubscribe',
      provider_id: id
    });
  }, [sendMessage]);

  /**
   * Request provider status
   */
  const requestStatus = useCallback(() => {
    if (!providerId) return;

    sendMessage({
      type: 'request_status',
      provider_id: providerId
    });
  }, [providerId, sendMessage]);

  /**
   * Request provider metrics
   */
  const requestMetrics = useCallback(() => {
    if (!providerId) return;

    sendMessage({
      type: 'request_metrics',
      provider_id: providerId
    });
  }, [providerId, sendMessage]);

  /**
   * Request connection statistics
   */
  const requestConnectionStats = useCallback(() => {
    sendMessage({
      type: 'request_connection_stats'
    });
  }, [sendMessage]);

  // Connect on mount and reconnect on token change
  useEffect(() => {
    connect();

    return () => {
      // Clean up on unmount
      if (socketRef.current) {
        socketRef.current.close();
      }

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  // Subscribe to provider when it changes
  useEffect(() => {
    if (isConnected && providerId) {
      subscribeToProvider(providerId);
    }

    return () => {
      if (providerId) {
        unsubscribeFromProvider(providerId);
      }
    };
  }, [isConnected, providerId, subscribeToProvider, unsubscribeFromProvider]);

  return {
    connectionState,
    isConnected,
    lastMessage,
    status,
    statusUpdates,
    metrics,
    events,
    sendMessage,
    subscribeToProvider,
    unsubscribeFromProvider,
    requestStatus,
    requestStatusUpdate: requestStatus,  // For backward compatibility
    requestMetrics,
    requestConnectionStats,
    refreshConnection
  };
};
