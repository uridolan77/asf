import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  CircularProgress,
  Alert,
  Box,
  Typography,
  Link
} from '@mui/material';
import { ReportExportFormat, ExportStatus } from '../../types/reporting';
import { exportReport, getExportStatus } from '../../api/reporting';

interface ExportReportDialogProps {
  open: boolean;
  onClose: () => void;
  reportId?: number;
}

const ExportReportDialog: React.FC<ExportReportDialogProps> = ({
  open,
  onClose,
  reportId
}) => {
  const [format, setFormat] = useState<ReportExportFormat>(ReportExportFormat.CSV);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [exportStatus, setExportStatus] = useState<ExportStatus | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  
  // Clear state when dialog opens/closes
  useEffect(() => {
    if (!open) {
      if (pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);
      }
      setExportStatus(null);
      setError(null);
    }
  }, [open]);
  
  // Poll for export status
  useEffect(() => {
    if (exportStatus && exportStatus.status !== 'completed' && exportStatus.status !== 'failed') {
      const interval = setInterval(async () => {
        try {
          const status = await getExportStatus(exportStatus.task_id);
          setExportStatus(status);
          
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(interval);
            setPollingInterval(null);
            setLoading(false);
          }
        } catch (err) {
          console.error('Error polling export status:', err);
          clearInterval(interval);
          setPollingInterval(null);
          setLoading(false);
          setError('Failed to get export status');
        }
      }, 2000);
      
      setPollingInterval(interval);
      
      return () => {
        clearInterval(interval);
      };
    }
  }, [exportStatus]);
  
  const handleFormatChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFormat(event.target.value as ReportExportFormat);
  };
  
  const handleExport = async () => {
    if (!reportId) {
      setError('Report ID is required');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const status = await exportReport(reportId, format);
      setExportStatus(status);
      
      // If already completed, no need to poll
      if (status.status === 'completed' || status.status === 'failed') {
        setLoading(false);
      }
    } catch (err) {
      console.error('Error exporting report:', err);
      setLoading(false);
      setError('Failed to export the report');
    }
  };
  
  const getStatusText = () => {
    if (!exportStatus) return '';
    
    switch (exportStatus.status) {
      case 'pending':
        return 'Export request received. Waiting to be processed...';
      case 'processing':
        return 'Export is being processed...';
      case 'completed':
        return 'Export completed successfully!';
      case 'failed':
        return `Export failed: ${exportStatus.error || 'Unknown error'}`;
      default:
        return `Status: ${exportStatus.status}`;
    }
  };
  
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Export Report</DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        
        {exportStatus && exportStatus.status === 'failed' && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {getStatusText()}
          </Alert>
        )}
        
        {exportStatus && exportStatus.status === 'completed' ? (
          <Box sx={{ textAlign: 'center', py: 2 }}>
            <Alert severity="success" sx={{ mb: 2 }}>
              {getStatusText()}
            </Alert>
            
            {exportStatus.download_url && (
              <Button
                variant="contained"
                color="primary"
                component={Link}
                href={exportStatus.download_url}
                target="_blank"
                rel="noopener"
                sx={{ mt: 2 }}
              >
                Download {format.toUpperCase()} File
              </Button>
            )}
          </Box>
        ) : (
          <>
            {exportStatus && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="body1" gutterBottom>
                  {getStatusText()}
                </Typography>
                <LinearProgressWithLabel value={
                  exportStatus.status === 'pending' ? 10 :
                  exportStatus.status === 'processing' ? 50 :
                  0
                } />
              </Box>
            )}
            
            <FormControl component="fieldset" sx={{ mt: 1 }}>
              <FormLabel component="legend">Export Format</FormLabel>
              <RadioGroup
                value={format}
                onChange={handleFormatChange}
              >
                <FormControlLabel
                  value={ReportExportFormat.CSV}
                  control={<Radio />}
                  label="CSV"
                  disabled={loading}
                />
                <FormControlLabel
                  value={ReportExportFormat.EXCEL}
                  control={<Radio />}
                  label="Excel"
                  disabled={loading}
                />
                <FormControlLabel
                  value={ReportExportFormat.PDF}
                  control={<Radio />}
                  label="PDF"
                  disabled={loading}
                />
                <FormControlLabel
                  value={ReportExportFormat.JSON}
                  control={<Radio />}
                  label="JSON"
                  disabled={loading}
                />
              </RadioGroup>
            </FormControl>
          </>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Close
        </Button>
        
        {(!exportStatus || exportStatus.status === 'failed') && (
          <Button
            onClick={handleExport}
            variant="contained"
            color="primary"
            disabled={loading || !reportId}
          >
            {loading ? <CircularProgress size={24} /> : 'Export'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

// Linear progress with label component
function LinearProgressWithLabel(props: { value: number }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Box sx={{ width: '100%', mr: 1 }}>
        <LinearProgress variant="determinate" {...props} />
      </Box>
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="body2" color="text.secondary">{`${Math.round(
          props.value,
        )}%`}</Typography>
      </Box>
    </Box>
  );
}

// Import LinearProgress
import { LinearProgress } from '@mui/material';

export default ExportReportDialog;
