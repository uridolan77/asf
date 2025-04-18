import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Tooltip,
  CircularProgress,
  Divider,
  Chip,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useTheme
} from '@mui/material';
import {
  ElectricBolt as ElectricBoltIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { format, formatDistanceToNow } from 'date-fns';
import { ResponsiveLine } from '@nivo/line';
import { useMCPWebSocket } from '../../../hooks/useMCPWebSocket';
import apiService from '../../../services/api';

/**
 * Circuit Breaker Status component for MCP providers
 * 
 * Displays the current state of the circuit breaker and provides
 * visualization of failure patterns over time.
 */
const CircuitBreakerStatus = ({ providerId }) => {
  const theme = useTheme();
  const [historyDialogOpen, setHistoryDialogOpen] = useState(false);
  const [historyData, setHistoryData] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [timeRange, setTimeRange] = useState('day');
  
  // Get real-time status via WebSocket
  const { status, isConnected, requestStatus } = useMCPWebSocket(providerId);
  
  // Fetch circuit breaker history
  const fetchHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const data = await apiService.llm.getCircuitBreakerHistory(providerId, timeRange);
      setHistoryData(data);
    } catch (error) {
      console.error('Error fetching circuit breaker history:', error);
    } finally {
      setIsLoadingHistory(false);
    }
  };
  
  // Fetch history when dialog opens or time range changes
  useEffect(() => {
    if (historyDialogOpen) {
      fetchHistory();
    }
  }, [historyDialogOpen, timeRange, providerId]);
  
  // Prepare data for visualization
  const circuitBreakerState = status?.circuit_breaker?.state || 'closed';
  const failureCount = status?.circuit_breaker?.failure_count || 0;
  const recoveryTime = status?.circuit_breaker?.recovery_time ? new Date(status.circuit_breaker.recovery_time) : null;
  const lastFailure = status?.circuit_breaker?.last_failure ? new Date(status.circuit_breaker.last_failure) : null;
  
  // Format history data for chart
  const chartData = historyData.length > 0 ? [
    {
      id: 'failures',
      color: theme.palette.error.main,
      data: historyData.map(item => ({
        x: new Date(item.timestamp),
        y: item.failure_count
      }))
    }
  ] : [];
  
  // Get status color
  const getStatusColor = () => {
    if (circuitBreakerState === 'open') {
      return theme.palette.error.main;
    }
    if (failureCount > 0) {
      return theme.palette.warning.main;
    }
    return theme.palette.success.main;
  };
  
  // Get status icon
  const getStatusIcon = () => {
    if (circuitBreakerState === 'open') {
      return <WarningIcon color="error" fontSize="large" />;
    }
    if (failureCount > 0) {
      return <ElectricBoltIcon color="warning" fontSize="large" />;
    }
    return <CheckCircleIcon color="success" fontSize="large" />;
  };
  
  return (
    <>
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="h6" component="div">
              Circuit Breaker Status
            </Typography>
            <Box>
              <Tooltip title="View failure history">
                <Button
                  size="small"
                  startIcon={<TimelineIcon />}
                  onClick={() => setHistoryDialogOpen(true)}
                >
                  History
                </Button>
              </Tooltip>
              <Tooltip title="Refresh status">
                <Button
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={() => requestStatus()}
                  disabled={!isConnected}
                >
                  Refresh
                </Button>
              </Tooltip>
            </Box>
          </Box>
          
          <Divider sx={{ my: 1 }} />
          
          <Box display="flex" alignItems="center" mb={2}>
            {getStatusIcon()}
            <Box ml={2}>
              <Typography variant="h5" component="div" sx={{ color: getStatusColor() }}>
                {circuitBreakerState === 'open' ? 'OPEN' : 'CLOSED'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {circuitBreakerState === 'open' 
                  ? `Will reset in ${formatDistanceToNow(recoveryTime)}`
                  : 'Circuit breaker is allowing requests'}
              </Typography>
            </Box>
          </Box>
          
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Failure Count
                </Typography>
                <Typography variant="h6">
                  {failureCount}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  Last Failure
                </Typography>
                <Typography variant="h6">
                  {lastFailure ? formatDistanceToNow(lastFailure, { addSuffix: true }) : 'None'}
                </Typography>
              </Box>
            </Grid>
          </Grid>
          
          {circuitBreakerState === 'open' && (
            <Box mt={2}>
              <Chip 
                icon={<InfoIcon />} 
                label={`Recovery at ${recoveryTime ? format(recoveryTime, 'HH:mm:ss') : 'unknown'}`} 
                color="error" 
                variant="outlined" 
              />
            </Box>
          )}
        </CardContent>
      </Card>
      
      {/* History Dialog */}
      <Dialog 
        open={historyDialogOpen} 
        onClose={() => setHistoryDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Circuit Breaker History
          <Typography variant="body2" color="text.secondary">
            Failure patterns over time for {providerId}
          </Typography>
        </DialogTitle>
        <DialogContent>
          {isLoadingHistory ? (
            <Box display="flex" justifyContent="center" alignItems="center" height={300}>
              <CircularProgress />
            </Box>
          ) : historyData.length === 0 ? (
            <Box display="flex" justifyContent="center" alignItems="center" height={300}>
              <Typography variant="body1" color="text.secondary">
                No failure data available for this time period
              </Typography>
            </Box>
          ) : (
            <Box height={400}>
              <ResponsiveLine
                data={chartData}
                margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
                xScale={{
                  type: 'time',
                  format: 'native',
                  precision: 'minute',
                }}
                yScale={{
                  type: 'linear',
                  min: 0,
                  max: 'auto',
                }}
                axisBottom={{
                  format: '%H:%M',
                  tickValues: 'every 1 hour',
                  legend: 'Time',
                  legendOffset: 40,
                }}
                axisLeft={{
                  legend: 'Failure Count',
                  legendOffset: -40,
                }}
                enablePoints={true}
                pointSize={8}
                pointColor={{ theme: 'background' }}
                pointBorderWidth={2}
                pointBorderColor={{ from: 'serieColor' }}
                enableArea={true}
                areaOpacity={0.1}
                useMesh={true}
                enableSlices="x"
                sliceTooltip={({ slice }) => {
                  return (
                    <div
                      style={{
                        background: 'white',
                        padding: '9px 12px',
                        border: '1px solid #ccc',
                        borderRadius: '4px',
                      }}
                    >
                      <div style={{ marginBottom: 6 }}>
                        {format(slice.points[0].data.x, 'MMM d, yyyy HH:mm:ss')}
                      </div>
                      {slice.points.map(point => (
                        <div
                          key={point.id}
                          style={{
                            color: point.serieColor,
                            padding: '3px 0',
                          }}
                        >
                          <strong>Failures:</strong> {point.data.y}
                        </div>
                      ))}
                    </div>
                  )
                }}
                theme={{
                  axis: {
                    ticks: {
                      text: {
                        fill: theme.palette.text.secondary,
                      },
                    },
                    legend: {
                      text: {
                        fill: theme.palette.text.primary,
                      },
                    },
                  },
                  grid: {
                    line: {
                      stroke: theme.palette.divider,
                    },
                  },
                  crosshair: {
                    line: {
                      stroke: theme.palette.primary.main,
                    },
                  },
                }}
              />
            </Box>
          )}
          
          <Box display="flex" justifyContent="center" mt={2}>
            <Button
              variant={timeRange === 'hour' ? 'contained' : 'outlined'}
              size="small"
              onClick={() => setTimeRange('hour')}
              sx={{ mx: 1 }}
            >
              Last Hour
            </Button>
            <Button
              variant={timeRange === 'day' ? 'contained' : 'outlined'}
              size="small"
              onClick={() => setTimeRange('day')}
              sx={{ mx: 1 }}
            >
              Last Day
            </Button>
            <Button
              variant={timeRange === 'week' ? 'contained' : 'outlined'}
              size="small"
              onClick={() => setTimeRange('week')}
              sx={{ mx: 1 }}
            >
              Last Week
            </Button>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHistoryDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

CircuitBreakerStatus.propTypes = {
  providerId: PropTypes.string.isRequired,
};

export default CircuitBreakerStatus;
