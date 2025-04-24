import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Button,
  Alert
} from '@mui/material';
import {
  Visibility as ViewIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Schedule as ScheduleIcon
} from '@mui/icons-material';

import { getScheduledReports, deleteSchedule } from '../../api/reporting';
import { ReportSchedule, ReportExportFormat } from '../../types/reporting';

const ScheduledReports: React.FC = () => {
  const navigate = useNavigate();
  const [schedules, setSchedules] = useState<ReportSchedule[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<boolean>(false);
  const [scheduleToDelete, setScheduleToDelete] = useState<ReportSchedule | null>(null);
  
  const fetchSchedules = async () => {
    try {
      setLoading(true);
      const data = await getScheduledReports();
      setSchedules(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching scheduled reports:', err);
      setError('Failed to load scheduled reports');
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchSchedules();
  }, []);
  
  const handleViewReport = (reportId: number) => {
    navigate(`/reports/view/${reportId}`);
  };
  
  const handleEditSchedule = (reportId: number) => {
    navigate(`/reports/view/${reportId}?schedule=true`);
  };
  
  const handleDeleteClick = (schedule: ReportSchedule) => {
    setScheduleToDelete(schedule);
    setDeleteDialogOpen(true);
  };
  
  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setScheduleToDelete(null);
  };
  
  const handleDeleteConfirm = async () => {
    if (!scheduleToDelete?.id) return;
    
    try {
      await deleteSchedule(scheduleToDelete.id);
      setSchedules(schedules.filter(s => s.id !== scheduleToDelete.id));
      setDeleteDialogOpen(false);
      setScheduleToDelete(null);
    } catch (err) {
      console.error('Error deleting schedule:', err);
      setError('Failed to delete the schedule');
    }
  };
  
  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString();
  };
  
  const formatFrequency = (schedule: ReportSchedule) => {
    switch (schedule.frequency) {
      case 'daily':
        return 'Daily';
      case 'weekly':
        return `Weekly (${formatDaysOfWeek(schedule.days_of_week)})`;
      case 'monthly':
        return `Monthly (${formatDaysOfMonth(schedule.days_of_month)})`;
      default:
        return schedule.frequency;
    }
  };
  
  const formatDaysOfWeek = (days?: number[]) => {
    if (!days || days.length === 0) return 'All days';
    
    const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    return days.map(day => dayNames[day]).join(', ');
  };
  
  const formatDaysOfMonth = (days?: number[]) => {
    if (!days || days.length === 0) return 'All days';
    
    return days.join(', ');
  };
  
  const formatRecipients = (recipients: string[]) => {
    if (recipients.length === 0) return 'None';
    
    if (recipients.length <= 2) {
      return recipients.join(', ');
    }
    
    return `${recipients.slice(0, 2).join(', ')} +${recipients.length - 2} more`;
  };
  
  const getFormatColor = (format: ReportExportFormat) => {
    switch (format) {
      case ReportExportFormat.CSV:
        return 'primary';
      case ReportExportFormat.EXCEL:
        return 'success';
      case ReportExportFormat.PDF:
        return 'error';
      case ReportExportFormat.JSON:
        return 'warning';
      default:
        return 'default';
    }
  };
  
  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          Scheduled Reports
        </Typography>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      <Paper sx={{ width: '100%', overflow: 'hidden' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        ) : schedules.length === 0 ? (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              No scheduled reports found
            </Typography>
            <Typography variant="body2" color="text.secondary">
              You can schedule a report from the report viewer or editor.
            </Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Report</TableCell>
                  <TableCell>Frequency</TableCell>
                  <TableCell>Format</TableCell>
                  <TableCell>Recipients</TableCell>
                  <TableCell>Last Run</TableCell>
                  <TableCell>Next Run</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {schedules.map((schedule) => (
                  <TableRow key={schedule.id}>
                    <TableCell>{schedule.name}</TableCell>
                    <TableCell>
                      <Button
                        variant="text"
                        size="small"
                        onClick={() => handleViewReport(schedule.report_id)}
                      >
                        View Report
                      </Button>
                    </TableCell>
                    <TableCell>{formatFrequency(schedule)}</TableCell>
                    <TableCell>
                      <Chip
                        label={schedule.export_format.toUpperCase()}
                        color={getFormatColor(schedule.export_format) as any}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{formatRecipients(schedule.recipients)}</TableCell>
                    <TableCell>{formatDate(schedule.last_run)}</TableCell>
                    <TableCell>{formatDate(schedule.next_run)}</TableCell>
                    <TableCell align="right">
                      <Tooltip title="View Report">
                        <IconButton onClick={() => handleViewReport(schedule.report_id)}>
                          <ViewIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit Schedule">
                        <IconButton onClick={() => handleEditSchedule(schedule.report_id)}>
                          <EditIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete Schedule">
                        <IconButton onClick={() => handleDeleteClick(schedule)}>
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>
      
      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
      >
        <DialogTitle>Delete Schedule</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the schedule "{scheduleToDelete?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDeleteCancel}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error">Delete</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ScheduledReports;
