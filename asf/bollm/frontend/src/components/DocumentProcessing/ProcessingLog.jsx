import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip,
  Button
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  ArrowDownward as ArrowDownwardIcon,
  ContentCopy as ContentCopyIcon,
  Save as SaveIcon
} from '@mui/icons-material';

/**
 * Component for displaying real-time processing logs
 */
const ProcessingLog = ({ logs, isLoading, error, title = 'Processing Log' }) => {
  const [autoScroll, setAutoScroll] = useState(true);
  const logEndRef = useRef(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // Copy logs to clipboard
  const copyLogs = () => {
    const logText = logs.map(log => `[${log.timestamp}] [${log.level}] ${log.message}`).join('\n');
    navigator.clipboard.writeText(logText);
  };

  // Save logs to file
  const saveLogs = () => {
    const logText = logs.map(log => `[${log.timestamp}] [${log.level}] ${log.message}`).join('\n');
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `processing-log-${new Date().toISOString().replace(/:/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Get icon for log level
  const getLogIcon = (level) => {
    switch (level) {
      case 'info':
        return <InfoIcon color="info" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'error':
        return <ErrorIcon color="error" />;
      case 'success':
        return <CheckCircleIcon color="success" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6">{title}</Typography>
        <Box>
          <Tooltip title="Copy logs to clipboard">
            <IconButton onClick={copyLogs} size="small" sx={{ mr: 1 }}>
              <ContentCopyIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Save logs to file">
            <IconButton onClick={saveLogs} size="small">
              <SaveIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Divider sx={{ mb: 2 }} />

      {isLoading && logs.length === 0 ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexGrow: 1 }}>
          <CircularProgress size={24} sx={{ mr: 2 }} />
          <Typography>Waiting for processing to start...</Typography>
        </Box>
      ) : error ? (
        <Box sx={{ p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
          <Typography color="error">{error}</Typography>
        </Box>
      ) : (
        <Box sx={{ flexGrow: 1, overflow: 'auto', maxHeight: 'calc(100% - 80px)', minHeight: '300px' }}>
          <List dense>
            {logs.map((log, index) => (
              <ListItem key={index} sx={{ py: 0.5 }}>
                <ListItemIcon sx={{ minWidth: 36 }}>
                  {getLogIcon(log.level)}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box>
                      <Typography
                        variant="body2"
                        component="span"
                        sx={{
                          fontFamily: 'monospace',
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word'
                        }}
                      >
                        {log.message}
                      </Typography>
                      {log.details && (
                        <Box>
                          <Typography
                            variant="body2"
                            component="div"
                            sx={{
                              fontFamily: 'monospace',
                              whiteSpace: 'pre-wrap',
                              wordBreak: 'break-word',
                              color: 'text.secondary',
                              fontSize: '0.85em',
                              mt: 0.5,
                              ml: 1,
                              borderLeft: '2px solid',
                              borderColor: 'divider',
                              pl: 1
                            }}
                          >
                            {log.details}
                          </Typography>

                          {/* Display installation instructions if present */}
                          {log.details.includes('Installation Instructions:') && (
                            <Box
                              sx={{
                                mt: 1,
                                ml: 1,
                                p: 1,
                                bgcolor: 'info.light',
                                borderRadius: 1,
                                border: '1px solid',
                                borderColor: 'info.main'
                              }}
                            >
                              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: 'info.dark' }}>
                                Installation Instructions
                              </Typography>
                              <Typography
                                variant="body2"
                                component="div"
                                sx={{
                                  fontFamily: 'monospace',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-word',
                                  mt: 0.5
                                }}
                              >
                                {log.details.split('Installation Instructions:')[1].split('Note:')[0]}
                              </Typography>
                              <Button
                                size="small"
                                variant="contained"
                                color="info"
                                sx={{ mt: 1 }}
                                onClick={() => {
                                  const instructions = log.details.split('Installation Instructions:')[1].split('Note:')[0];
                                  navigator.clipboard.writeText(instructions.trim());
                                }}
                              >
                                Copy Instructions
                              </Button>
                            </Box>
                          )}

                          {/* Display dependency conflict warning if present */}
                          {log.details.includes('Note: There may be dependency conflicts') && (
                            <Box
                              sx={{
                                mt: 1,
                                ml: 1,
                                p: 1,
                                bgcolor: 'warning.light',
                                borderRadius: 1,
                                border: '1px solid',
                                borderColor: 'warning.main'
                              }}
                            >
                              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: 'warning.dark' }}>
                                Dependency Conflict Warning
                              </Typography>
                              <Typography
                                variant="body2"
                                component="div"
                                sx={{
                                  fontFamily: 'monospace',
                                  whiteSpace: 'pre-wrap',
                                  wordBreak: 'break-word',
                                  mt: 0.5
                                }}
                              >
                                {log.details.split('Note:')[1]}
                              </Typography>
                            </Box>
                          )}
                        </Box>
                      )}
                    </Box>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                      <Typography
                        variant="caption"
                        sx={{ fontFamily: 'monospace' }}
                      >
                        {formatTimestamp(log.timestamp)}
                      </Typography>
                    </Box>
                  }
                />
                {log.stage && (
                  <Chip
                    label={log.stage}
                    size="small"
                    color="primary"
                    variant="outlined"
                    sx={{ ml: 1 }}
                  />
                )}
              </ListItem>
            ))}
            <div ref={logEndRef} />
          </List>
        </Box>
      )}

      {logs.length > 0 && (
        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="caption">
            {logs.length} log entries
          </Typography>
          <Button
            size="small"
            variant="outlined"
            startIcon={<ArrowDownwardIcon />}
            onClick={() => {
              setAutoScroll(!autoScroll);
              if (!autoScroll && logEndRef.current) {
                logEndRef.current.scrollIntoView({ behavior: 'smooth' });
              }
            }}
            color={autoScroll ? 'primary' : 'inherit'}
          >
            {autoScroll ? 'Auto-scroll on' : 'Auto-scroll off'}
          </Button>
        </Box>
      )}
    </Paper>
  );
};

export default ProcessingLog;
