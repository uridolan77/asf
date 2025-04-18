import React, { useState } from 'react';
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
  Button,
  IconButton,
  Tooltip,
  Card,
  CardContent,
  Divider,
  Alert,
  useTheme
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent
} from '@mui/lab';
import {
  Info as InfoIcon,
  Assessment as AssessmentIcon,
  BarChart as BarChartIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon
} from '@mui/icons-material';
import moment from 'moment';
import { PerformanceChart } from './visualizations';

const TaskHistory = ({ adapter, onRefresh }) => {
  const theme = useTheme();
  const [expandedTask, setExpandedTask] = useState(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);
  const [viewMode, setViewMode] = useState('timeline'); // 'timeline' or 'table'

  // If no task history, show a message
  if (!adapter || !adapter.task_history || adapter.task_history.length === 0) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info">
          This adapter has not been trained on any tasks yet. Use the Training tab to train this adapter on a task.
        </Alert>
      </Box>
    );
  }

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleExpandTask = (taskId) => {
    if (expandedTask === taskId) {
      setExpandedTask(null);
    } else {
      setExpandedTask(taskId);
    }
  };

  const toggleViewMode = () => {
    setViewMode(viewMode === 'timeline' ? 'table' : 'timeline');
  };

  const formatMetric = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return value;
  };

  const renderMetrics = (metrics) => {
    if (!metrics) return null;

    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Metrics
        </Typography>
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Metric</TableCell>
                <TableCell align="right">Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(metrics).map(([key, value]) => (
                <TableRow key={key}>
                  <TableCell component="th" scope="row">
                    {key}
                  </TableCell>
                  <TableCell align="right">{formatMetric(value)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  const renderTimelineView = () => {
    return (
      <Timeline position="alternate">
        {adapter.task_history.map((task, index) => (
          <TimelineItem key={task.task_id || index}>
            <TimelineOppositeContent color="text.secondary">
              {moment(task.trained_at).format('MMM D, YYYY h:mm A')}
            </TimelineOppositeContent>
            <TimelineSeparator>
              <TimelineDot color="primary">
                <AssessmentIcon />
              </TimelineDot>
              {index < adapter.task_history.length - 1 && <TimelineConnector />}
            </TimelineSeparator>
            <TimelineContent>
              <Card variant="outlined">
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h6" component="div">
                      {task.task_id}
                    </Typography>
                    <IconButton
                      size="small"
                      onClick={() => handleExpandTask(task.task_id)}
                      aria-expanded={expandedTask === task.task_id}
                      aria-label="show more"
                    >
                      {expandedTask === task.task_id ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                  </Box>

                  {expandedTask === task.task_id && (
                    <>
                      <Divider sx={{ my: 1.5 }} />
                      {renderMetrics(task.metrics)}

                      {task.eval_metrics && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Evaluation Metrics
                          </Typography>
                          <TableContainer component={Paper} variant="outlined">
                            <Table size="small">
                              <TableHead>
                                <TableRow>
                                  <TableCell>Metric</TableCell>
                                  <TableCell align="right">Value</TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {Object.entries(task.eval_metrics).map(([key, value]) => (
                                  <TableRow key={key}>
                                    <TableCell component="th" scope="row">
                                      {key}
                                    </TableCell>
                                    <TableCell align="right">{formatMetric(value)}</TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        </Box>
                      )}
                    </>
                  )}
                </CardContent>
              </Card>
            </TimelineContent>
          </TimelineItem>
        ))}
      </Timeline>
    );
  };

  const renderTableView = () => {
    return (
      <>
        <TableContainer component={Paper} variant="outlined">
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Task ID</TableCell>
                <TableCell>Trained At</TableCell>
                <TableCell>Metrics</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {adapter.task_history
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((task, index) => (
                  <TableRow key={task.task_id || index}>
                    <TableCell component="th" scope="row">
                      {task.task_id}
                    </TableCell>
                    <TableCell>
                      {moment(task.trained_at).format('MMM D, YYYY h:mm A')}
                    </TableCell>
                    <TableCell>
                      {task.metrics && Object.keys(task.metrics).length > 0 ? (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {Object.entries(task.metrics)
                            .filter(([key]) => key.includes('loss') || key.includes('accuracy'))
                            .slice(0, 2)
                            .map(([key, value]) => (
                              <Chip
                                key={key}
                                label={`${key}: ${formatMetric(value)}`}
                                size="small"
                                variant="outlined"
                              />
                            ))}
                          {Object.keys(task.metrics).length > 2 && (
                            <Chip
                              label={`+${Object.keys(task.metrics).length - 2} more`}
                              size="small"
                              variant="outlined"
                            />
                          )}
                        </Box>
                      ) : (
                        <Typography variant="body2" color="text.secondary">
                          No metrics available
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => handleExpandTask(task.task_id)}
                        >
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="View Metrics">
                        <IconButton size="small">
                          <BarChartIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={adapter.task_history.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </>
    );
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" component="h2">
          Task History
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            size="small"
            onClick={toggleViewMode}
          >
            {viewMode === 'timeline' ? 'Table View' : 'Timeline View'}
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<RefreshIcon />}
            onClick={onRefresh}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {expandedTask && (
        <Box sx={{ mb: 3 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Task Details: {expandedTask}
              </Typography>
              <Divider sx={{ my: 1.5 }} />

              {adapter.task_history
                .filter(task => task.task_id === expandedTask)
                .map((task, index) => (
                  <Box key={index}>
                    <Typography variant="subtitle2" gutterBottom>
                      Trained At
                    </Typography>
                    <Typography variant="body2" paragraph>
                      {moment(task.trained_at).format('MMMM D, YYYY h:mm:ss A')}
                    </Typography>

                    {renderMetrics(task.metrics)}

                    {task.eval_metrics && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Evaluation Metrics
                        </Typography>
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Metric</TableCell>
                                <TableCell align="right">Value</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {Object.entries(task.eval_metrics).map(([key, value]) => (
                                <TableRow key={key}>
                                  <TableCell component="th" scope="row">
                                    {key}
                                  </TableCell>
                                  <TableCell align="right">{formatMetric(value)}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    )}
                  </Box>
                ))}
            </CardContent>
          </Card>
        </Box>
      )}

      {adapter.task_history && adapter.task_history.length > 1 && (
        <Box sx={{ mb: 3 }}>
          <PerformanceChart
            adapter={adapter}
            metrics={['loss', 'accuracy', 'f1']}
            title="Performance Across Tasks"
          />
        </Box>
      )}

      {viewMode === 'timeline' ? renderTimelineView() : renderTableView()}
    </Box>
  );
};

export default TaskHistory;
