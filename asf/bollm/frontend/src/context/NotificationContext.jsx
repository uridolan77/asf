import React, { createContext, useContext, useState, useCallback } from 'react';
import { Snackbar, Alert, Slide } from '@mui/material';

// Create context
const NotificationContext = createContext();

// Transition component for notifications
const SlideTransition = (props) => {
  return <Slide {...props} direction="down" />;
};

/**
 * Notification provider component
 */
export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);

  // Add a notification
  const addNotification = useCallback((message, severity = 'info', duration = 5000) => {
    // Generate a truly unique ID by combining timestamp with a random string
    const id = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    setNotifications(prev => [...prev, { id, message, severity, duration }]);
    return id;
  }, []);

  // Remove a notification
  const removeNotification = useCallback((id) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  }, []);

  // Shorthand methods for different notification types
  const showSuccess = useCallback((message, duration) => {
    return addNotification(message, 'success', duration);
  }, [addNotification]);

  const showError = useCallback((message, duration) => {
    return addNotification(message, 'error', duration);
  }, [addNotification]);

  const showInfo = useCallback((message, duration) => {
    return addNotification(message, 'info', duration);
  }, [addNotification]);

  const showWarning = useCallback((message, duration) => {
    return addNotification(message, 'warning', duration);
  }, [addNotification]);

  // Handle notification close
  const handleClose = (id) => {
    removeNotification(id);
  };

  return (
    <NotificationContext.Provider
      value={{
        addNotification,
        removeNotification,
        showSuccess,
        showError,
        showInfo,
        showWarning,
      }}
    >
      {children}
      
      {/* Render notifications */}
      {notifications.map(({ id, message, severity, duration }, index) => (
        <Snackbar
          key={id}
          open={true}
          autoHideDuration={duration}
          onClose={() => handleClose(id)}
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
          TransitionComponent={SlideTransition}
          sx={{ mt: index * 8 }}
        >
          <Alert 
            onClose={() => handleClose(id)} 
            severity={severity} 
            variant="filled"
            sx={{ width: '100%' }}
          >
            {message}
          </Alert>
        </Snackbar>
      ))}
    </NotificationContext.Provider>
  );
};

// Custom hook to use the notification context
export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

export default NotificationContext;
