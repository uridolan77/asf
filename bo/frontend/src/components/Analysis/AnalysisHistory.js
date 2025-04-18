import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, Button, Grid, Chip,
  Divider, CircularProgress, Alert, Tooltip,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, TablePagination,
  IconButton, MenuItem, Select, FormControl,
  InputLabel, TextField
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Visibility as VisibilityIcon,
  Delete as DeleteIcon,
  FilterList as FilterListIcon,
  Search as SearchIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext';
import apiService from '../../services/api';
import { ContentLoader } from '../UI/LoadingIndicators.js';
import { FadeIn, StaggeredList } from '../UI/Animations.js';

/**
 * Analysis History component
 *
 * This component displays the history of analyses performed by the user.
 */
const AnalysisHistory = ({ onViewAnalysis, onExport }) => {
  const { showSuccess, showError } = useNotification();

  // State
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState(null);
  const [error, setError] = useState('');

  // Pagination state
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Filter state
  const [analysisType, setAnalysisType] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // Load analysis history on mount and when filters change
  useEffect(() => {
    loadAnalysisHistory();
  }, [page, rowsPerPage, analysisType]);

  // Load analysis history
  const loadAnalysisHistory = async () => {
    setIsLoading(true);
    setError('');

    try {
      const params = {
        page: page + 1, // API uses 1-based indexing
        page_size: rowsPerPage
      };

      if (analysisType) {
        params.analysis_type = analysisType;
      }

      if (searchQuery) {
        params.query = searchQuery;
      }

      const result = await apiService.analysis.getHistory(params);

      if (result.success) {
        setHistory(result.data.data);
      } else {
        setError(`Failed to load analysis history: ${result.error}`);
        showError(`Failed to load analysis history: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading analysis history:', error);
      setError(`Error loading analysis history: ${error.message}`);
      showError(`Error loading analysis history: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle page change
  const handleChangePage = (_, newPage) => {
    setPage(newPage);
  };

  // Handle rows per page change
  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Handle view analysis
  const handleViewAnalysis = (analysisId) => {
    if (onViewAnalysis) {
      onViewAnalysis(analysisId);
    }
  };

  // Handle export analysis
  const handleExportAnalysis = (analysisId, format) => {
    if (onExport) {
      onExport(format, { analysis_id: analysisId });
    }
  };

  // Get analysis type chip color
  const getAnalysisTypeColor = (type) => {
    switch (type.toLowerCase()) {
      case 'contradiction':
        return 'error';
      case 'cap':
        return 'primary';
      case 'custom':
        return 'secondary';
      default:
        return 'default';
    }
  };

  // Handle search
  const handleSearch = () => {
    setPage(0);
    loadAnalysisHistory();
  };

  if (isLoading && !history) {
    return <ContentLoader height={400} message="Loading analysis history..." />;
  }

  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Analysis History
          </Typography>

          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadAnalysisHistory}
            disabled={isLoading}
          >
            Refresh
          </Button>
        </Box>

        <Divider sx={{ mb: 3 }} />

        {/* Filters */}
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                variant="outlined"
                size="small"
                placeholder="Search by query or description"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: <SearchIcon color="action" sx={{ mr: 1 }} />
                }}
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel id="analysis-type-label">Analysis Type</InputLabel>
                <Select
                  labelId="analysis-type-label"
                  value={analysisType}
                  label="Analysis Type"
                  onChange={(e) => setAnalysisType(e.target.value)}
                >
                  <MenuItem value="">All Types</MenuItem>
                  <MenuItem value="contradiction">Contradiction</MenuItem>
                  <MenuItem value="cap">CAP</MenuItem>
                  <MenuItem value="custom">Custom</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <Button
                variant="contained"
                startIcon={<FilterListIcon />}
                onClick={handleSearch}
                fullWidth
              >
                Apply Filters
              </Button>
            </Grid>
          </Grid>
        </Box>

        {error ? (
          <Alert
            severity="error"
            action={
              <Button color="inherit" size="small" onClick={loadAnalysisHistory}>
                Retry
              </Button>
            }
          >
            {error}
          </Alert>
        ) : history ? (
          <FadeIn>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Analysis ID</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Query/Description</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell>Results</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {history.analyses.map((analysis) => (
                    <TableRow key={analysis.id} hover>
                      <TableCell>{analysis.id}</TableCell>
                      <TableCell>
                        <Chip
                          label={analysis.type}
                          color={getAnalysisTypeColor(analysis.type)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{analysis.query || analysis.description}</TableCell>
                      <TableCell>{new Date(analysis.timestamp).toLocaleString()}</TableCell>
                      <TableCell>
                        {analysis.result_count ? (
                          <Chip
                            label={`${analysis.result_count} results`}
                            color="primary"
                            variant="outlined"
                            size="small"
                          />
                        ) : (
                          <Chip
                            label="No results"
                            variant="outlined"
                            size="small"
                          />
                        )}
                      </TableCell>
                      <TableCell>
                        <IconButton
                          color="primary"
                          onClick={() => handleViewAnalysis(analysis.id)}
                          size="small"
                          sx={{ mr: 1 }}
                        >
                          <VisibilityIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          color="secondary"
                          onClick={() => handleExportAnalysis(analysis.id, 'pdf')}
                          size="small"
                        >
                          <DownloadIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            <TablePagination
              component="div"
              count={history.total_count}
              page={page}
              onPageChange={handleChangePage}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              rowsPerPageOptions={[5, 10, 25, 50]}
            />
          </FadeIn>
        ) : (
          <Alert severity="info">
            No analysis history found. Perform an analysis to see it here.
          </Alert>
        )}
      </Paper>
    </Box>
  );
};

export default AnalysisHistory;
