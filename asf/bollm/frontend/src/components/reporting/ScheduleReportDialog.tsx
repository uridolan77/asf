import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Box,
  Typography,
  FormHelperText,
  CircularProgress,
  Alert,
  Grid,
  Checkbox,
  FormControlLabel,
  SelectChangeEvent
} from '@mui/material';
import { ReportSchedule, ReportExportFormat } from '../../types/reporting';
import { scheduleReport, getScheduledReport, updateSchedule } from '../../api/reporting';

interface ScheduleReportDialogProps {
  open: boolean;
  onClose: () => void;
  reportId?: number;
  scheduleId?: number;
  onScheduleSuccess?: (schedule: ReportSchedule) => void;
}

const ScheduleReportDialog: React.FC<ScheduleReportDialogProps> = ({
  open,
  onClose,
  reportId,
  scheduleId,
  onScheduleSuccess
}) => {
  const [name, setName] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [frequency, setFrequency] = useState<'daily' | 'weekly' | 'monthly'>('daily');
  const [time, setTime] = useState<string>('09:00:00');
  const [daysOfWeek, setDaysOfWeek] = useState<number[]>([1, 2, 3, 4, 5]); // Mon-Fri
  const [daysOfMonth, setDaysOfMonth] = useState<number[]>([1]);
  const [exportFormat, setExportFormat] = useState<ReportExportFormat>(ReportExportFormat.CSV);
  const [recipients, setRecipients] = useState<string[]>([]);
  const [newRecipient, setNewRecipient] = useState<string>('');
  
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Load existing schedule if scheduleId is provided
  useEffect(() => {
    const fetchSchedule = async () => {
      if (scheduleId && open) {
        try {
          setLoading(true);
          const schedule = await getScheduledReport(scheduleId);
          
          setName(schedule.name);
          setDescription(schedule.description || '');
          setFrequency(schedule.frequency);
          setTime(schedule.time);
          setDaysOfWeek(schedule.days_of_week || []);
          setDaysOfMonth(schedule.days_of_month || []);
          setExportFormat(schedule.export_format);
          setRecipients(schedule.recipients);
          
          setLoading(false);
        } catch (err) {
          console.error('Error fetching schedule:', err);
          setError('Failed to load the schedule');
          setLoading(false);
        }
      }
    };
    
    fetchSchedule();
  }, [scheduleId, open]);
  
  // Reset form when dialog opens
  useEffect(() => {
    if (open && !scheduleId) {
      setName('');
      setDescription('');
      setFrequency('daily');
      setTime('09:00:00');
      setDaysOfWeek([1, 2, 3, 4, 5]);
      setDaysOfMonth([1]);
      setExportFormat(ReportExportFormat.CSV);
      setRecipients([]);
      setNewRecipient('');
      setError(null);
    }
  }, [open, scheduleId]);
  
  const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setName(event.target.value);
  };
  
  const handleDescriptionChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setDescription(event.target.value);
  };
  
  const handleFrequencyChange = (event: SelectChangeEvent<string>) => {
    setFrequency(event.target.value as 'daily' | 'weekly' | 'monthly');
  };
  
  const handleTimeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTime(event.target.value + ':00'); // Add seconds
  };
  
  const handleDayOfWeekToggle = (day: number) => {
    setDaysOfWeek(prev => {
      if (prev.includes(day)) {
        return prev.filter(d => d !== day);
      } else {
        return [...prev, day].sort();
      }
    });
  };
  
  const handleDayOfMonthToggle = (day: number) => {
    setDaysOfMonth(prev => {
      if (prev.includes(day)) {
        return prev.filter(d => d !== day);
      } else {
        return [...prev, day].sort((a, b) => a - b);
      }
    });
  };
  
  const handleExportFormatChange = (event: SelectChangeEvent<string>) => {
    setExportFormat(event.target.value as ReportExportFormat);
  };
  
  const handleNewRecipientChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNewRecipient(event.target.value);
  };
  
  const handleAddRecipient = () => {
    if (newRecipient && isValidEmail(newRecipient) && !recipients.includes(newRecipient)) {
      setRecipients([...recipients, newRecipient]);
      setNewRecipient('');
    }
  };
  
  const handleDeleteRecipient = (recipient: string) => {
    setRecipients(recipients.filter(r => r !== recipient));
  };
  
  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleAddRecipient();
    }
  };
  
  const isValidEmail = (email: string): boolean => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  };
  
  const handleSave = async () => {
    if (!reportId) {
      setError('Report ID is required');
      return;
    }
    
    if (!name.trim()) {
      setError('Schedule name is required');
      return;
    }
    
    if (recipients.length === 0) {
      setError('At least one recipient is required');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const scheduleData: ReportSchedule = {
        name,
        description,
        report_id: reportId,
        frequency,
        time,
        days_of_week: frequency === 'weekly' ? daysOfWeek : undefined,
        days_of_month: frequency === 'monthly' ? daysOfMonth : undefined,
        export_format: exportFormat,
        recipients
      };
      
      let savedSchedule: ReportSchedule;
      
      if (scheduleId) {
        // Update existing schedule
        savedSchedule = await updateSchedule(scheduleId, {
          ...scheduleData,
          id: scheduleId
        });
      } else {
        // Create new schedule
        savedSchedule = await scheduleReport(scheduleData);
      }
      
      if (onScheduleSuccess) {
        onScheduleSuccess(savedSchedule);
      }
      
      onClose();
    } catch (err) {
      console.error('Error saving schedule:', err);
      setError('Failed to save the schedule');
      setLoading(false);
    }
  };
  
  const renderDaysOfWeekSelector = () => {
    const days = [
      { value: 0, label: 'Mon' },
      { value: 1, label: 'Tue' },
      { value: 2, label: 'Wed' },
      { value: 3, label: 'Thu' },
      { value: 4, label: 'Fri' },
      { value: 5, label: 'Sat' },
      { value: 6, label: 'Sun' }
    ];
    
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Days of Week
        </Typography>
        <Grid container spacing={1}>
          {days.map(day => (
            <Grid item key={day.value}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={daysOfWeek.includes(day.value)}
                    onChange={() => handleDayOfWeekToggle(day.value)}
                    size="small"
                  />
                }
                label={day.label}
              />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };
  
  const renderDaysOfMonthSelector = () => {
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Days of Month
        </Typography>
        <Grid container spacing={1}>
          {Array.from({ length: 31 }, (_, i) => i + 1).map(day => (
            <Grid item key={day}>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={daysOfMonth.includes(day)}
                    onChange={() => handleDayOfMonthToggle(day)}
                    size="small"
                  />
                }
                label={day}
              />
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };
  
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        {scheduleId ? 'Edit Schedule' : 'Schedule Report'}
      </DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <TextField
              autoFocus
              margin="dense"
              label="Schedule Name"
              fullWidth
              value={name}
              onChange={handleNameChange}
              required
              disabled={loading}
            />
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              margin="dense"
              label="Description"
              fullWidth
              multiline
              rows={2}
              value={description}
              onChange={handleDescriptionChange}
              disabled={loading}
            />
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth margin="dense">
              <InputLabel>Frequency</InputLabel>
              <Select
                value={frequency}
                onChange={handleFrequencyChange}
                label="Frequency"
                disabled={loading}
              >
                <MenuItem value="daily">Daily</MenuItem>
                <MenuItem value="weekly">Weekly</MenuItem>
                <MenuItem value="monthly">Monthly</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <TextField
              margin="dense"
              label="Time"
              type="time"
              fullWidth
              value={time.substring(0, 5)} // Remove seconds
              onChange={handleTimeChange}
              disabled={loading}
              InputLabelProps={{
                shrink: true,
              }}
            />
          </Grid>
          
          {frequency === 'weekly' && (
            <Grid item xs={12}>
              {renderDaysOfWeekSelector()}
            </Grid>
          )}
          
          {frequency === 'monthly' && (
            <Grid item xs={12}>
              {renderDaysOfMonthSelector()}
            </Grid>
          )}
          
          <Grid item xs={12}>
            <FormControl fullWidth margin="dense">
              <InputLabel>Export Format</InputLabel>
              <Select
                value={exportFormat}
                onChange={handleExportFormatChange}
                label="Export Format"
                disabled={loading}
              >
                <MenuItem value={ReportExportFormat.CSV}>CSV</MenuItem>
                <MenuItem value={ReportExportFormat.EXCEL}>Excel</MenuItem>
                <MenuItem value={ReportExportFormat.PDF}>PDF</MenuItem>
                <MenuItem value={ReportExportFormat.JSON}>JSON</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>
              Recipients
            </Typography>
            
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <TextField
                fullWidth
                size="small"
                label="Add Recipient Email"
                value={newRecipient}
                onChange={handleNewRecipientChange}
                onKeyPress={handleKeyPress}
                error={newRecipient !== '' && !isValidEmail(newRecipient)}
                helperText={newRecipient !== '' && !isValidEmail(newRecipient) ? 'Invalid email format' : ''}
                disabled={loading}
              />
              <Button
                variant="contained"
                color="primary"
                onClick={handleAddRecipient}
                disabled={!newRecipient || !isValidEmail(newRecipient) || loading}
                sx={{ ml: 1, height: 40 }}
              >
                Add
              </Button>
            </Box>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {recipients.map(recipient => (
                <Chip
                  key={recipient}
                  label={recipient}
                  onDelete={() => handleDeleteRecipient(recipient)}
                  color="primary"
                  variant="outlined"
                  disabled={loading}
                />
              ))}
            </Box>
            
            {recipients.length === 0 && (
              <FormHelperText error>
                At least one recipient is required
              </FormHelperText>
            )}
          </Grid>
        </Grid>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleSave}
          variant="contained"
          color="primary"
          disabled={loading || !name.trim() || recipients.length === 0}
        >
          {loading ? <CircularProgress size={24} /> : (scheduleId ? 'Update' : 'Schedule')}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ScheduleReportDialog;
