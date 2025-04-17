import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon,
  Clear as ClearIcon
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import ClientService from '../../services/ClientService';
import { formatDateTime } from '../../utils/formatters';

interface DSPyAuditLogViewerProps {
  clientId: string;
}

const DSPyAuditLogViewer: React.FC<DSPyAuditLogViewerProps> = ({ clientId }) => {
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<number>(0);
  const [rowsPerPage, setRowsPerPage] = useState<number>(10);
  const [filters, setFilters] = useState<any>({
    event_type: '',
    user_id: '',
    component: '',
    status: '',
    date_from: null,
    date_to: null,
    search_term: ''
  });

  // For demonstration, we'll use these as options for the filters
  const eventTypes = ['LLM_CALL', 'MODULE_CALL', 'ERROR', 'CACHE_OPERATION'];
  const components = ['DSPyClient', 'EnhancedDSPyClient', 'MedicalEvidenceExtractor', 'ContradictionDetector'];
  const statuses = ['SUCCESS', 'ERROR'];

  useEffect(() => {
    loadAuditLogs();
  }, [clientId]);

  const loadAuditLogs = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // In a real implementation, we would fetch actual logs from the API
      // const response = await ClientService.getAuditLogs(clientId, filters);
      // setLogs(response.logs);
      
      // For now, generate mock audit logs
      setTimeout(() => {
        const mockLogs = generateMockAuditLogs();
        setLogs(mockLogs);
        setLoading(false);
      }, 800);
    } catch (err) {
      console.error('Error loading audit logs:', err);
      setError('Failed to load audit logs. Please try again.');
      setLoading(false);
    }
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleFilterChange = (key: string, value: any) => {
    setFilters({
      ...filters,
      [key]: value
    });
  };

  const clearFilters = () => {
    setFilters({
      event_type: '',
      user_id: '',
      component: '',
      status: '',
      date_from: null,
      date_to: null,
      search_term: ''
    });
  };

  const applyFilters = () => {
    loadAuditLogs();
  };

  // Helper function to generate mock audit logs
  const generateMockAuditLogs = () => {
    const mockLogs = [];
    const now = new Date();
    const eventTypeOptions = ['LLM_CALL', 'MODULE_CALL', 'ERROR', 'CACHE_OPERATION'];
    const componentOptions = ['DSPyClient', 'EnhancedDSPyClient', 'MedicalEvidenceExtractor', 'ContradictionDetector'];
    
    for (let i = 0; i < 50; i++) {
      const date = new Date(now);
      date.setMinutes(now.getMinutes() - i * 15);
      
      const eventType = eventTypeOptions[Math.floor(Math.random() * eventTypeOptions.length)];
      const component = componentOptions[Math.floor(Math.random() * componentOptions.length)];
      const hasError = Math.random() > 0.85; // 15% chance of error
      
      const log = {
        audit_id: `audit-${i}`,
        event_type: eventType,
        event_timestamp: date.toISOString(),
        user_id: `user-${Math.floor(Math.random() * 5) + 1}`,
        session_id: `session-${Math.floor(Math.random() * 10) + 1}`,
        component: component,
        inputs: {
          text: 'Sample text input...',
          parameters: { model: 'gpt-4', temperature: 0.7 }
        },
        outputs: hasError ? null : {
          result: 'Sample output...'
        },
        error: hasError ? 'Sample error message' : null,
        latency: Math.random() * 3 + 0.5,
        status: hasError ? 'ERROR' : 'SUCCESS'
      };
      
      mockLogs.push(log);
    }
    
    return mockLogs;
  };

  const filteredLogs = logs.filter(log => {
    // Apply filters
    if (filters.event_type && log.event_type !== filters.event_type) return false;
    if (filters.component && log.component !== filters.component) return false;
    if (filters.user_id && log.user_id !== filters.user_id) return false;
    if (filters.status && log.status !== filters.status) return false;
    
    if (filters.date_from && new Date(log.event_timestamp) < new Date(filters.date_from)) return false;
    if (filters.date_to && new Date(log.event_timestamp) > new Date(filters.date_to)) return false;
    
    if (filters.search_term) {
      const searchTerm = filters.search_term.toLowerCase();
      const searchableFields = [
        log.event_type,
        log.component,
        log.user_id,
        log.session_id,
        JSON.stringify(log.inputs),
        JSON.stringify(log.outputs),
        log.error
      ].filter(field => field !== null && field !== undefined);
      
      return searchableFields.some(field => 
        field.toString().toLowerCase().includes(searchTerm)
      );
    }
    
    return true;
  });

  const slicedLogs = filteredLogs.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  return (
    <Box>
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5">Audit Logs</Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadAuditLogs}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
        
        <Accordion defaultExpanded>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="filters-panel-content"
            id="filters-panel-header"
          >
            <Typography>Filters</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={3}>
                <FormControl fullWidth variant="outlined" size="small">
                  <InputLabel id="event-type-label">Event Type</InputLabel>
                  <Select
                    labelId="event-type-label"
                    value={filters.event_type}
                    onChange={(e) => handleFilterChange('event_type', e.target.value)}
                    label="Event Type"
                  >
                    <MenuItem value="">All</MenuItem>
                    {eventTypes.map((type) => (
                      <MenuItem key={type} value={type}>{type}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth variant="outlined" size="small">
                  <InputLabel id="component-label">Component</InputLabel>
                  <Select
                    labelId="component-label"
                    value={filters.component}
                    onChange={(e) => handleFilterChange('component', e.target.value)}
                    label="Component"
                  >
                    <MenuItem value="">All</MenuItem>
                    {components.map((comp) => (
                      <MenuItem key={comp} value={comp}>{comp}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <TextField
                  fullWidth
                  label="User ID"
                  variant="outlined"
                  size="small"
                  value={filters.user_id}
                  onChange={(e) => handleFilterChange('user_id', e.target.value)}
                />
              </Grid>
              
              <Grid item xs={12} md={3}>
                <FormControl fullWidth variant="outlined" size="small">
                  <InputLabel id="status-label">Status</InputLabel>
                  <Select
                    labelId="status-label"
                    value={filters.status}
                    onChange={(e) => handleFilterChange('status', e.target.value)}
                    label="Status"
                  >
                    <MenuItem value="">All</MenuItem>
                    {statuses.map((status) => (
                      <MenuItem key={status} value={status}>{status}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <DatePicker
                  label="From Date"
                  value={filters.date_from}
                  onChange={(newValue) => handleFilterChange('date_from', newValue)}
                  slotProps={{ textField: { size: 'small', fullWidth: true } }}
                />
              </Grid>
              
              <Grid item xs={12} md={3}>
                <DatePicker
                  label="To Date"
                  value={filters.date_to}
                  onChange={(newValue) => handleFilterChange('date_to', newValue)}
                  slotProps={{ textField: { size: 'small', fullWidth: true } }}
                />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Search"
                  variant="outlined"
                  size="small"
                  value={filters.search_term}
                  onChange={(e) => handleFilterChange('search_term', e.target.value)}
                  InputProps={{
                    startAdornment: <SearchIcon color="action" sx={{ mr: 1 }} />,
                  }}
                />
              </Grid>
              
              <Grid item xs={6} md={1}>
                <Button
                  variant="outlined"
                  onClick={clearFilters}
                  startIcon={<ClearIcon />}
                  fullWidth
                >
                  Clear
                </Button>
              </Grid>
              
              <Grid item xs={6} md={1}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={applyFilters}
                  fullWidth
                >
                  Apply
                </Button>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
        
        {loading ? (
          <Box sx={{ mt: 3 }}>
            <LinearProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mt: 3 }}>
            {error}
          </Alert>
        ) : filteredLogs.length === 0 ? (
          <Alert severity="info" sx={{ mt: 3 }}>
            No audit logs match your filters. Try adjusting your search criteria.
          </Alert>
        ) : (
          <Box sx={{ mt: 3 }}>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Timestamp</TableCell>
                    <TableCell>Event Type</TableCell>
                    <TableCell>Component</TableCell>
                    <TableCell>User ID</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Latency</TableCell>
                    <TableCell>Details</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {slicedLogs.map((log) => (
                    <TableRow key={log.audit_id} hover>
                      <TableCell>{formatDateTime(log.event_timestamp)}</TableCell>
                      <TableCell>
                        <Chip 
                          label={log.event_type}
                          size="small"
                          color={
                            log.event_type === 'ERROR' ? 'error' :
                            log.event_type === 'LLM_CALL' ? 'primary' :
                            log.event_type === 'MODULE_CALL' ? 'secondary' :
                            'default'
                          }
                        />
                      </TableCell>
                      <TableCell>{log.component}</TableCell>
                      <TableCell>{log.user_id}</TableCell>
                      <TableCell>
                        <Chip 
                          label={log.status}
                          size="small"
                          color={log.status === 'SUCCESS' ? 'success' : 'error'}
                        />
                      </TableCell>
                      <TableCell>
                        {log.latency ? `${log.latency.toFixed(2)}s` : 'N/A'}
                      </TableCell>
                      <TableCell>
                        <Accordion sx={{ boxShadow: 'none', '&:before': { display: 'none' } }}>
                          <AccordionSummary
                            expandIcon={<ExpandMoreIcon />}
                            aria-controls={`log-${log.audit_id}-content`}
                            id={`log-${log.audit_id}-header`}
                            sx={{ 
                              padding: 0, 
                              minHeight: 'auto',
                              '& .MuiAccordionSummary-content': { margin: 0 }
                            }}
                          >
                            <Typography variant="caption">Details</Typography>
                          </AccordionSummary>
                          <AccordionDetails sx={{ padding: 1 }}>
                            <Box sx={{ fontSize: '0.75rem' }}>
                              <Typography variant="caption" component="div" sx={{ fontWeight: 'bold' }}>
                                Inputs:
                              </Typography>
                              <Box component="pre" sx={{ 
                                fontSize: '0.75rem', 
                                backgroundColor: '#f5f5f5',
                                padding: 1,
                                borderRadius: 1,
                                overflow: 'auto',
                                maxHeight: '100px'
                              }}>
                                {JSON.stringify(log.inputs, null, 2)}
                              </Box>
                              
                              {log.outputs && (
                                <>
                                  <Typography variant="caption" component="div" sx={{ fontWeight: 'bold', mt: 1 }}>
                                    Outputs:
                                  </Typography>
                                  <Box component="pre" sx={{ 
                                    fontSize: '0.75rem', 
                                    backgroundColor: '#f5f5f5',
                                    padding: 1,
                                    borderRadius: 1,
                                    overflow: 'auto',
                                    maxHeight: '100px'
                                  }}>
                                    {JSON.stringify(log.outputs, null, 2)}
                                  </Box>
                                </>
                              )}
                              
                              {log.error && (
                                <>
                                  <Typography variant="caption" component="div" sx={{ fontWeight: 'bold', mt: 1, color: 'error.main' }}>
                                    Error:
                                  </Typography>
                                  <Box component="pre" sx={{ 
                                    fontSize: '0.75rem', 
                                    backgroundColor: '#fee8e8',
                                    padding: 1,
                                    borderRadius: 1,
                                    color: 'error.main',
                                    overflow: 'auto',
                                    maxHeight: '100px'
                                  }}>
                                    {log.error}
                                  </Box>
                                </>
                              )}
                            </Box>
                          </AccordionDetails>
                        </Accordion>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            <TablePagination
              component="div"
              count={filteredLogs.length}
              page={page}
              onPageChange={handleChangePage}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              rowsPerPageOptions={[10, 25, 50, 100]}
            />
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default DSPyAuditLogViewer;