import React from 'react';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Button
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Check as CheckIcon,
  Clear as ClearIcon,
  Error as ErrorIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import { useDSPy } from '../../hooks/useDSPy';
import { useFeatureFlags } from '../../context/FeatureFlagContext';

interface DSPyCircuitBreakersProps {
  clientId: string;
}

const DSPyCircuitBreakers: React.FC<DSPyCircuitBreakersProps> = ({ clientId }) => {
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Use the DSPy hook
  const {
    getCircuitBreakersByClient,
    resetCircuitBreaker
  } = useDSPy();

  // Get circuit breakers by client
  const {
    data: circuitBreakers = [],
    isLoading,
    isError,
    error,
    refetch: refetchCircuitBreakers
  } = getCircuitBreakersByClient(clientId);

  // Reset circuit breaker
  const {
    mutate: resetCircuitBreakerMutate,
    isPending: isResetting
  } = resetCircuitBreaker(clientId, '');

  // Handle reset circuit breaker
  const handleResetCircuitBreaker = (breakerName: string) => {
    resetCircuitBreakerMutate({ breaker_name: breakerName });
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case 'CLOSED': return 'success';
      case 'OPEN': return 'error';
      case 'HALF_OPEN': return 'warning';
      default: return 'default';
    }
  };

  const getStateIcon = (state: string) => {
    switch (state) {
      case 'CLOSED': return <CheckIcon fontSize="small" />;
      case 'OPEN': return <ClearIcon fontSize="small" />;
      case 'HALF_OPEN': return <WarningIcon fontSize="small" />;
      default: return <ErrorIcon fontSize="small" />;
    }
  };

  if (useMockData) {
    return (
      <Alert severity="info">
        Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
      </Alert>
    );
  }

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (isError && circuitBreakers.length === 0) {
    return (
      <Alert severity="error">{error?.message || 'Failed to load circuit breakers'}</Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="subtitle1">Circuit Breaker Status</Typography>
        <IconButton size="small" onClick={() => refetchCircuitBreakers()} disabled={isLoading}>
          <RefreshIcon fontSize="small" />
        </IconButton>
      </Box>

      {isError && (
        <Alert severity="warning" sx={{ mb: 2 }}>{error?.message || 'Failed to load circuit breakers'}</Alert>
      )}

      {circuitBreakers.length === 0 ? (
        <Alert severity="info">No circuit breakers found for this client.</Alert>
      ) : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>State</TableCell>
                <TableCell align="right">Failure Count</TableCell>
                <TableCell align="right">Success Count</TableCell>
                <TableCell>Last Failure</TableCell>
                <TableCell>Last State Change</TableCell>
                <TableCell>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {circuitBreakers.map((cb) => (
                <TableRow key={cb.name}>
                  <TableCell>{cb.name}</TableCell>
                  <TableCell>
                    <Chip
                      icon={getStateIcon(cb.state)}
                      label={cb.state}
                      color={getStateColor(cb.state) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell align="right">{cb.failure_count} / {cb.failure_threshold}</TableCell>
                  <TableCell align="right">N/A</TableCell>
                  <TableCell>
                    {cb.last_failure_time ? new Date(cb.last_failure_time).toLocaleString() : 'N/A'}
                  </TableCell>
                  <TableCell>{new Date().toLocaleString()}</TableCell>
                  <TableCell>
                    <Tooltip title="Reset Circuit Breaker">
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => handleResetCircuitBreaker(cb.name)}
                        disabled={cb.state === 'CLOSED' || isResetting}
                      >
                        Reset
                      </Button>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default DSPyCircuitBreakers;