import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert,
  useTheme
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  BarChart as BarChartIcon,
  Delete as DeleteIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { format } from 'date-fns';

/**
 * TaskHistory component displays the training history for an adapter
 */
const TaskHistory = ({ adapter }) => {
  const theme = useTheme();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const [loading, setLoading] = useState(false);
  
  // Handle page change
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };
  
  // Handle rows per page change
  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };
  
  // Format date
  const formatDate = (dateString) => {
    try {
      return format(new Date(dateString), 'MMM d, yyyy HH:mm:ss');
    } catch (error) {
      return dateString;
    }
  };
  
  // Get task history from adapter
  const taskHistory = adapter?.task_history || [];
  
  // Sort task history by trained_at date (newest first)
  const sortedTaskHistory = [...taskHistory].sort((a, b) => {
    return new Date(b.trained_at) - new Date(a.trained_at);
  });
  
  // Get current page of task history
  const currentPageTasks = sortedTaskHistory.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );
  
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (taskHistory.length === 0) {
    return (
      <Alert severity="info">
        No task history available for this adapter.
      </Alert>
    );
  }
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Task History
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        This adapter has been trained on {taskHistory.length} task{taskHistory.length !== 1 ? 's' : ''}.
      </Typography>
      
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Task ID</TableCell>
              <TableCell>Trained At</TableCell>
              <TableCell>Metrics</TableCell>
              <TableCell>Evaluation</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {currentPageTasks.map((task) => (
              <TableRow key={task.task_id}>
                <TableCell>
                  <Typography variant="body2" noWrap>
                    {task.task_id}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="body2">
                    {formatDate(task.trained_at)}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                    {task.metrics && (
                      <>
                        <Typography variant="caption" component="div">
                          Loss: {task.metrics.loss?.toFixed(4)}
                        </Typography>
                        <Typography variant="caption" component="div">
                          Accuracy: {task.metrics.accuracy?.toFixed(4)}
                        </Typography>
                      </>
                    )}
                  </Box>
                </TableCell>
                <TableCell>
                  {task.eval_metrics ? (
                    <Chip 
                      label="Evaluated" 
                      size="small" 
                      color="success" 
                      variant="outlined"
                    />
                  ) : (
                    <Chip 
                      label="Not Evaluated" 
                      size="small" 
                      color="default" 
                      variant="outlined"
                    />
                  )}
                </TableCell>
                <TableCell align="right">
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <Tooltip title="View Details">
                      <IconButton size="small">
                        <VisibilityIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="View Metrics">
                      <IconButton size="small">
                        <BarChartIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <TablePagination
        rowsPerPageOptions={[5, 10, 25]}
        component="div"
        count={taskHistory.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Box>
  );
};

export default TaskHistory;
