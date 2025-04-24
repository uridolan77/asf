import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  CircularProgress,
  Alert
} from '@mui/material';
import { ReportDefinition, SavedReport } from '../../types/reporting';
import { saveReport, updateReport } from '../../api/reporting';

interface SaveReportDialogProps {
  open: boolean;
  onClose: () => void;
  reportDefinition: ReportDefinition;
  savedReport: SavedReport | null;
  onSaveSuccess: (report: SavedReport) => void;
}

const SaveReportDialog: React.FC<SaveReportDialogProps> = ({
  open,
  onClose,
  reportDefinition,
  savedReport,
  onSaveSuccess
}) => {
  const [name, setName] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Initialize form with saved report data if available
  useEffect(() => {
    if (savedReport) {
      setName(savedReport.name || '');
      setDescription(savedReport.description || '');
    }
  }, [savedReport, open]);
  
  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setName(event.target.value);
  };
  
  const handleDescriptionChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setDescription(event.target.value);
  };
  
  const handleSave = async () => {
    if (!name.trim()) {
      setError('Report name is required');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const reportData: SavedReport = {
        name,
        description,
        report_definition: reportDefinition
      };
      
      let savedReportData: SavedReport;
      
      if (savedReport?.id) {
        // Update existing report
        savedReportData = await updateReport(savedReport.id, {
          ...savedReport,
          ...reportData
        });
      } else {
        // Create new report
        savedReportData = await saveReport(reportData);
      }
      
      onSaveSuccess(savedReportData);
      onClose();
    } catch (err) {
      console.error('Error saving report:', err);
      setError('Failed to save the report');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {savedReport?.id ? 'Update Report' : 'Save Report'}
      </DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        
        <TextField
          autoFocus
          margin="dense"
          label="Report Name"
          fullWidth
          value={name}
          onChange={handleNameChange}
          required
          disabled={loading}
        />
        
        <TextField
          margin="dense"
          label="Description"
          fullWidth
          multiline
          rows={3}
          value={description}
          onChange={handleDescriptionChange}
          disabled={loading}
        />
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleSave}
          variant="contained"
          color="primary"
          disabled={loading || !name.trim()}
        >
          {loading ? <CircularProgress size={24} /> : 'Save'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default SaveReportDialog;
