import React, { useState, useEffect } from 'react';
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
import ClientService from '../../services/ClientService';
import { formatDateTime } from '../../utils/formatters';

interface DSPyCircuitBreakersProps {
  clientId: string;
}

const DSPyCircuitBreakers: React.FC<DSPyCircuitBreakersProps> = ({ clientId }) => {
  const [circuitBreakers, setCircuitBreakers] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCircuitBreakers();
  }, [clientId]);

  const loadCircuitBreakers = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await ClientService.getCircuitBreakerStatus(clientId);
      setCircuitBreakers(response);
    } catch (err) {
      console.error('Error loading circuit breakers:', err);
      setError('Failed to load circuit breaker status. Please try again.');
      
      // For demonstration purposes, generate mock data if API fails
      const mockData = generateMockCircuitBreakerData();
      setCircuitBreakers(mockData);
    } finally {
      setLoading(false);
    }
  };

  const generateMockCircuitBreakerData = () => {
    const mockData = [];
    const modules = ['MedicalEvidenceExtractor', 'ContradictionDetector', 'DiagnosticReasoning', 'ClinicalQA'];
    const states = ['CLOSED', 'OPEN', 'HALF_OPEN'];
    const now = new Date();
    
    for (const module of modules) {
      const state = states[Math.floor(Math.random() * states.length)];
      const failureCount = state === 'OPEN' ? Math.floor(Math.random() * 10) + 5 : Math.floor(Math.random() * 5);
      
      const lastStateChange = new Date(now);
      lastStateChange.setMinutes(now.getMinutes() - Math.floor(Math.random() * 60));
      
      mockData.push({
        name: module,
        state: state,
        failure_count: failureCount,
        success_count: state === 'CLOSED' ? Math.floor(Math.random() * 50) + 20 : Math.floor(Math.random() * 10),
        failure_threshold: 5,
        reset_timeout: 60,
        last_failure: state !== 'CLOSED' ? lastStateChange.toISOString() : null,
        last_state_change: lastStateChange.toISOString(),
      });
    }
    
    return mockData;
  };

  const resetCircuitBreaker = async (circuitBreakerName: string) => {
    try {
      // In a real implementation, you would call an API to reset the circuit breaker
      // await ClientService.resetCircuitBreaker(clientId, circuitBreakerName);
      
      // For now, just reload the data
      await loadCircuitBreakers();
    } catch (err) {
      console.error(`Error resetting circuit breaker ${circuitBreakerName}:`, err);
      setError(`Failed to reset circuit breaker ${circuitBreakerName}. Please try again.`);
    }
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

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (error && circuitBreakers.length === 0) {
    return (
      <Alert severity="error">{error}</Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="subtitle1">Circuit Breaker Status</Typography>
        <IconButton size="small" onClick={loadCircuitBreakers} disabled={loading}>
          <RefreshIcon fontSize="small" />
        </IconButton>
      </Box>
      
      {error && (
        <Alert severity="warning" sx={{ mb: 2 }}>{error}</Alert>
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
                  <TableCell align="right">{cb.success_count}</TableCell>
                  <TableCell>
                    {cb.last_failure ? formatDateTime(cb.last_failure) : 'N/A'}
                  </TableCell>
                  <TableCell>{formatDateTime(cb.last_state_change)}</TableCell>
                  <TableCell>
                    <Tooltip title="Reset Circuit Breaker">
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => resetCircuitBreaker(cb.name)}
                        disabled={cb.state === 'CLOSED'}
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