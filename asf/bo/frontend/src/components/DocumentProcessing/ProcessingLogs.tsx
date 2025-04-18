import React, { useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Divider,
  LinearProgress,
  Chip,
  Button
} from '@mui/material';
import {
  ContentCopy as CopyIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import { useNotification } from '../../context/NotificationContext';

interface ProcessingLogsProps {
  logs: string[];
  progress: number;
  stage: string;
  status: string;
  error: string | null;
}

/**
 * Component for displaying document processing logs
 */
const ProcessingLogs: React.FC<ProcessingLogsProps> = ({
  logs,
  progress,
  stage,
  status,
  error
}) => {
  const logsEndRef = useRef<HTMLDivElement>(null);
  const { showSuccess } = useNotification();

  // Auto-scroll to bottom of logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Copy logs to clipboard
  const handleCopyLogs = () => {
    navigator.clipboard.writeText(logs.join('\n'));
    showSuccess('Logs copied to clipboard');
  };

  // Download logs as text file
  const handleDownloadLogs = () => {
    const element = document.createElement('a');
    const file = new Blob([logs.join('\n')], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `document-processing-logs-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  // Get status color
  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'primary';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6">Processing Logs</Typography>
        <Box>
          <Button
            startIcon={<CopyIcon />}
            size="small"
            onClick={handleCopyLogs}
            sx={{ mr: 1 }}
          >
            Copy
          </Button>
          <Button
            startIcon={<DownloadIcon />}
            size="small"
            onClick={handleDownloadLogs}
          >
            Download
          </Button>
        </Box>
      </Box>

      <Divider sx={{ mb: 2 }} />

      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="body2">
            Progress: {progress.toFixed(0)}% - {stage}
          </Typography>
          <Chip
            label={status || 'Idle'}
            color={getStatusColor()}
            size="small"
          />
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          color={error ? 'error' : 'primary'}
          sx={{ height: 8, borderRadius: 4 }}
        />
      </Box>

      {error && (
        <Box sx={{ mb: 2, p: 1, bgcolor: 'error.light', borderRadius: 1 }}>
          <Typography variant="body2" color="error.contrastText">
            Error: {error}
          </Typography>
        </Box>
      )}

      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          bgcolor: 'grey.900',
          color: 'grey.100',
          p: 2,
          borderRadius: 1,
          fontFamily: 'monospace',
          fontSize: '0.875rem',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word'
        }}
      >
        {logs.length === 0 ? (
          <Typography variant="body2" color="grey.500">
            No logs yet. Start processing to see logs here.
          </Typography>
        ) : (
          logs.map((log, index) => (
            <div key={index}>
              {log}
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </Box>
    </Paper>
  );
};

export default ProcessingLogs;
