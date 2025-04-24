/**
 * WebSocket service for real-time updates.
 */

// WebSocket connection status
const WS_STATUS = {
  CONNECTING: 'connecting',
  OPEN: 'open',
  CLOSING: 'closing',
  CLOSED: 'closed',
  ERROR: 'error'
};

class WebSocketService {
  constructor() {
    this.socket = null;
    this.status = WS_STATUS.CLOSED;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectTimeout = null;
    this.messageHandlers = new Map();
    this.taskSubscriptions = new Set();
    this.connectionPromise = null;
    this.connectionResolve = null;
    this.connectionReject = null;
  }

  /**
   * Connect to the WebSocket server.
   * @returns {Promise<void>} A promise that resolves when the connection is established.
   */
  connect() {
    if (this.socket && (this.status === WS_STATUS.CONNECTING || this.status === WS_STATUS.OPEN)) {
      return this.connectionPromise;
    }

    this.connectionPromise = new Promise((resolve, reject) => {
      this.connectionResolve = resolve;
      this.connectionReject = reject;

      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const wsUrl = `${protocol}//${host}/ws/document-processing`;

        console.log(`Connecting to WebSocket at ${wsUrl}`);
        this.socket = new WebSocket(wsUrl);
        this.status = WS_STATUS.CONNECTING;

        this.socket.onopen = this.handleOpen.bind(this);
        this.socket.onmessage = this.handleMessage.bind(this);
        this.socket.onclose = this.handleClose.bind(this);
        this.socket.onerror = this.handleError.bind(this);
      } catch (error) {
        this.status = WS_STATUS.ERROR;
        this.connectionReject(error);
        console.error('Error connecting to WebSocket:', error);
      }
    });

    return this.connectionPromise;
  }

  /**
   * Disconnect from the WebSocket server.
   */
  disconnect() {
    if (this.socket && this.status !== WS_STATUS.CLOSED) {
      this.status = WS_STATUS.CLOSING;
      this.socket.close();
    }

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  /**
   * Handle WebSocket open event.
   */
  handleOpen() {
    console.log('WebSocket connection established');
    this.status = WS_STATUS.OPEN;
    this.reconnectAttempts = 0;

    // Resubscribe to tasks
    this.taskSubscriptions.forEach(taskId => {
      this.subscribeToTask(taskId);
    });

    if (this.connectionResolve) {
      this.connectionResolve();
      this.connectionResolve = null;
      this.connectionReject = null;
    }
  }

  /**
   * Handle WebSocket message event.
   * @param {MessageEvent} event - The message event.
   */
  handleMessage(event) {
    try {
      const message = JSON.parse(event.data);
      const messageType = message.type;
      const taskId = message.task_id;

      // Call all registered handlers for this message type
      if (this.messageHandlers.has(messageType)) {
        this.messageHandlers.get(messageType).forEach(handler => {
          handler(message);
        });
      }

      // Call task-specific handlers
      if (taskId && this.messageHandlers.has(`${messageType}:${taskId}`)) {
        this.messageHandlers.get(`${messageType}:${taskId}`).forEach(handler => {
          handler(message);
        });
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  /**
   * Handle WebSocket close event.
   * @param {CloseEvent} event - The close event.
   */
  handleClose(event) {
    console.log(`WebSocket connection closed: ${event.code} ${event.reason}`);
    this.status = WS_STATUS.CLOSED;
    this.socket = null;

    // Attempt to reconnect if not intentionally closed
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

      this.reconnectTimeout = setTimeout(() => {
        this.connect();
      }, delay);
    } else {
      console.log('Maximum reconnect attempts reached');
      if (this.connectionReject) {
        this.connectionReject(new Error('Maximum reconnect attempts reached'));
        this.connectionResolve = null;
        this.connectionReject = null;
      }
    }
  }

  /**
   * Handle WebSocket error event.
   * @param {Event} error - The error event.
   */
  handleError(error) {
    console.error('WebSocket error:', error);
    this.status = WS_STATUS.ERROR;

    if (this.connectionReject) {
      this.connectionReject(error);
      this.connectionResolve = null;
      this.connectionReject = null;
    }
  }

  /**
   * Send a message to the WebSocket server.
   * @param {Object} message - The message to send.
   * @returns {Promise<void>} A promise that resolves when the message is sent.
   */
  async send(message) {
    try {
      if (this.status !== WS_STATUS.OPEN) {
        await this.connect();
      }

      this.socket.send(JSON.stringify(message));
    } catch (error) {
      console.error('Error sending WebSocket message:', error);
      throw error;
    }
  }

  /**
   * Subscribe to a task.
   * @param {string} taskId - The task ID to subscribe to.
   * @returns {Promise<void>} A promise that resolves when the subscription is complete.
   */
  async subscribeToTask(taskId) {
    try {
      await this.send({
        type: 'subscribe',
        task_id: taskId
      });

      this.taskSubscriptions.add(taskId);
      console.log(`Subscribed to task ${taskId}`);
    } catch (error) {
      console.error(`Error subscribing to task ${taskId}:`, error);
      throw error;
    }
  }

  /**
   * Unsubscribe from a task.
   * @param {string} taskId - The task ID to unsubscribe from.
   * @returns {Promise<void>} A promise that resolves when the unsubscription is complete.
   */
  async unsubscribeFromTask(taskId) {
    try {
      await this.send({
        type: 'unsubscribe',
        task_id: taskId
      });

      this.taskSubscriptions.delete(taskId);
      console.log(`Unsubscribed from task ${taskId}`);
    } catch (error) {
      console.error(`Error unsubscribing from task ${taskId}:`, error);
      throw error;
    }
  }

  /**
   * Register a handler for a specific message type.
   * @param {string} messageType - The message type to handle.
   * @param {Function} handler - The handler function.
   * @param {string} [taskId] - Optional task ID for task-specific handlers.
   * @returns {Function} A function to unregister the handler.
   */
  onMessage(messageType, handler, taskId = null) {
    const key = taskId ? `${messageType}:${taskId}` : messageType;

    if (!this.messageHandlers.has(key)) {
      this.messageHandlers.set(key, new Set());
    }

    this.messageHandlers.get(key).add(handler);

    // Return a function to unregister the handler
    return () => {
      if (this.messageHandlers.has(key)) {
        this.messageHandlers.get(key).delete(handler);
        if (this.messageHandlers.get(key).size === 0) {
          this.messageHandlers.delete(key);
        }
      }
    };
  }

  /**
   * Register a handler for progress updates.
   * @param {string} taskId - The task ID to track progress for.
   * @param {Function} handler - The handler function.
   * @returns {Function} A function to unregister the handler.
   */
  onProgress(taskId, handler) {
    return this.onMessage('progress', handler, taskId);
  }

  /**
   * Register a handler for task completion.
   * @param {string} taskId - The task ID to track completion for.
   * @param {Function} handler - The handler function.
   * @returns {Function} A function to unregister the handler.
   */
  onCompleted(taskId, handler) {
    return this.onMessage('completed', handler, taskId);
  }

  /**
   * Register a handler for task failure.
   * @param {string} taskId - The task ID to track failure for.
   * @param {Function} handler - The handler function.
   * @returns {Function} A function to unregister the handler.
   */
  onFailed(taskId, handler) {
    return this.onMessage('failed', handler, taskId);
  }

  /**
   * Register a handler for intermediate results.
   * @param {string} taskId - The task ID to track intermediate results for.
   * @param {Function} handler - The handler function.
   * @returns {Function} A function to unregister the handler.
   */
  onIntermediateResult(taskId, handler) {
    return this.onMessage('intermediate_result', handler, taskId);
  }

  /**
   * Get the current WebSocket status.
   * @returns {string} The current status.
   */
  getStatus() {
    return this.status;
  }

  /**
   * Check if the WebSocket is connected.
   * @returns {boolean} True if connected, false otherwise.
   */
  isConnected() {
    return this.status === WS_STATUS.OPEN;
  }
}

// Create a singleton instance
const websocketService = new WebSocketService();

export default websocketService;
export { WS_STATUS };
