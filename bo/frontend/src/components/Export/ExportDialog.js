import React, { useState } from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  Button, FormControl, InputLabel, Select, MenuItem,
  FormControlLabel, Switch, Typography, Box,
  CircularProgress, Alert, Divider, Grid
} from '@mui/material';
import {
  Download as DownloadIcon,
  PictureAsPdf as PdfIcon,
  TableChart as ExcelIcon,
  Code as JsonIcon,
  Storage as CsvIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext.jsx';
import apiService from '../../services/api';

/**
 * Export Dialog component
 *
 * This component provides a dialog for exporting search results or analysis results
 * to various formats (JSON, CSV, Excel, PDF).
 */
const ExportDialog = ({
  open,
  onClose,
  resultId = null,
  analysisId = null,
  query = null,
  title = 'Export Results'
}) => {
  const { showSuccess, showError } = useNotification();

  // State
  const [format, setFormat] = useState('pdf');
  const [includeAbstracts, setIncludeAbstracts] = useState(true);
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [maxResults, setMaxResults] = useState(20);

  // Export status
  const [isExporting, setIsExporting] = useState(false);
  const [exportTaskId, setExportTaskId] = useState(null);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportError, setExportError] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');

  // Handle format change
  const handleFormatChange = (event) => {
    setFormat(event.target.value);
  };

  // Handle export
  const handleExport = async () => {
    if (!resultId && !analysisId && !query) {
      showError('No data to export');
      return;
    }

    setIsExporting(true);
    setExportError('');
    setExportTaskId(null);
    setDownloadUrl('');

    try {
      const params = {
        result_id: resultId,
        analysis_id: analysisId,
        query,
        max_results: maxResults,
        include_abstracts: includeAbstracts,
        include_metadata: includeMetadata
      };

      const result = await apiService.export.toFormat(format, params);

      if (result.success) {
        if (result.data.data.task_id) {
          // Background task
          setExportTaskId(result.data.data.task_id);
          showSuccess('Export started. Please wait while we prepare your file.');

          // Start polling for status
          pollExportStatus(result.data.data.task_id);
        } else if (result.data.data.download_url) {
          // Immediate download
          setDownloadUrl(result.data.data.download_url);
          showSuccess('Export completed successfully');
          setIsExporting(false);
        }
      } else {
        setExportError(`Export failed: ${result.error}`);
        showError(`Export failed: ${result.error}`);
        setIsExporting(false);
      }
    } catch (error) {
      console.error('Error exporting results:', error);
      setExportError(`Export error: ${error.message}`);
      showError(`Export error: ${error.message}`);
      setIsExporting(false);
    }
  };

  // Poll export status
  const pollExportStatus = async (taskId) => {
    try {
      const result = await apiService.export.getStatus(taskId);

      if (result.success) {
        const status = result.data.data;

        setExportProgress(status.progress);

        if (status.status === 'completed') {
          setDownloadUrl(status.download_url);
          showSuccess('Export completed successfully');
          setIsExporting(false);
        } else if (status.status === 'failed') {
          setExportError(`Export failed: ${status.error}`);
          showError(`Export failed: ${status.error}`);
          setIsExporting(false);
        } else {
          // Continue polling
          setTimeout(() => pollExportStatus(taskId), 2000);
        }
      } else {
        setExportError(`Failed to check export status: ${result.error}`);
        showError(`Failed to check export status: ${result.error}`);
        setIsExporting(false);
      }
    } catch (error) {
      console.error('Error checking export status:', error);
      setExportError(`Error checking export status: ${error.message}`);
      showError(`Error checking export status: ${error.message}`);
      setIsExporting(false);
    }
  };

  // Handle download
  const handleDownload = () => {
    if (downloadUrl) {
      window.open(downloadUrl, '_blank');
    }
  };

  // Reset state when dialog closes
  const handleClose = () => {
    if (!isExporting) {
      setFormat('pdf');
      setIncludeAbstracts(true);
      setIncludeMetadata(true);
      setMaxResults(20);
      setExportTaskId(null);
      setExportProgress(0);
      setExportError('');
      setDownloadUrl('');
      onClose();
    }
  };

  // Get format icon
  const getFormatIcon = () => {
    switch (format) {
      case 'pdf':
        return <PdfIcon />;
      case 'excel':
        return <ExcelIcon />;
      case 'json':
        return <JsonIcon />;
      case 'csv':
        return <CsvIcon />;
      default:
        return <DownloadIcon />;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        {exportError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {exportError}
          </Alert>
        )}

        {isExporting ? (
          <Box sx={{ textAlign: 'center', py: 3 }}>
            <CircularProgress variant="determinate" value={exportProgress} size={60} sx={{ mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Exporting...
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Please wait while we prepare your {format.toUpperCase()} file.
            </Typography>
          </Box>
        ) : downloadUrl ? (
          <Box sx={{ textAlign: 'center', py: 3 }}>
            <Box sx={{ fontSize: 60, color: 'success.main', mb: 2 }}>
              {getFormatIcon()}
            </Box>
            <Typography variant="h6" gutterBottom>
              Export Complete!
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Your {format.toUpperCase()} file is ready for download.
            </Typography>
            <Button
              variant="contained"
              color="primary"
              startIcon={<DownloadIcon />}
              onClick={handleDownload}
            >
              Download {format.toUpperCase()}
            </Button>
          </Box>
        ) : (
          <>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Select the format and options for your export.
            </Typography>

            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel id="format-label">Export Format</InputLabel>
                  <Select
                    labelId="format-label"
                    value={format}
                    label="Export Format"
                    onChange={handleFormatChange}
                  >
                    <MenuItem value="pdf">PDF Document</MenuItem>
                    <MenuItem value="excel">Excel Spreadsheet</MenuItem>
                    <MenuItem value="csv">CSV File</MenuItem>
                    <MenuItem value="json">JSON File</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12}>
                <Divider sx={{ my: 1 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Export Options
                </Typography>
              </Grid>

              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel id="max-results-label">Maximum Results</InputLabel>
                  <Select
                    labelId="max-results-label"
                    value={maxResults}
                    label="Maximum Results"
                    onChange={(e) => setMaxResults(e.target.value)}
                  >
                    <MenuItem value={10}>10 Results</MenuItem>
                    <MenuItem value={20}>20 Results</MenuItem>
                    <MenuItem value={50}>50 Results</MenuItem>
                    <MenuItem value={100}>100 Results</MenuItem>
                    <MenuItem value={200}>200 Results</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={includeAbstracts}
                      onChange={(e) => setIncludeAbstracts(e.target.checked)}
                    />
                  }
                  label="Include Abstracts"
                />
              </Grid>

              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={includeMetadata}
                      onChange={(e) => setIncludeMetadata(e.target.checked)}
                    />
                  }
                  label="Include Metadata (authors, journal, year, etc.)"
                />
              </Grid>
            </Grid>
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} disabled={isExporting}>
          {downloadUrl ? 'Close' : 'Cancel'}
        </Button>
        {!downloadUrl && !isExporting && (
          <Button
            variant="contained"
            color="primary"
            onClick={handleExport}
            startIcon={getFormatIcon()}
          >
            Export
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default ExportDialog;
