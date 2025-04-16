import React, { useState } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid, Chip,
  FormControlLabel, Switch, Slider, Divider, CircularProgress,
  Card, CardContent, CardHeader, CardActions, Tooltip, Alert
} from '@mui/material';
import {
  Search as SearchIcon,
  Science as ScienceIcon,
  Compare as CompareIcon,
  Download as DownloadIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext.jsx';
import apiService from '../../services/api';
import { ButtonLoader } from '../UI/LoadingIndicators.js';
import { FadeIn, StaggeredList } from '../UI/Animations.js';

/**
 * Contradiction Analysis component
 *
 * This component allows users to analyze contradictions in medical literature
 * based on a query.
 */
const ContradictionAnalysis = ({ onExport }) => {
  const { showSuccess, showError } = useNotification();

  // Form state
  const [query, setQuery] = useState('');
  const [maxResults, setMaxResults] = useState(20);
  const [threshold, setThreshold] = useState(0.7);
  const [useBioMedLM, setUseBioMedLM] = useState(true);
  const [useTSMixer, setUseTSMixer] = useState(false);
  const [useLorentz, setUseLorentz] = useState(false);

  // UI state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!query.trim()) {
      showError('Please enter a search query');
      setError('Please enter a search query');
      return;
    }

    setIsAnalyzing(true);
    setError('');

    try {
      const params = {
        query: query.trim(),
        max_results: maxResults,
        threshold,
        use_biomedlm: useBioMedLM,
        use_tsmixer: useTSMixer,
        use_lorentz: useLorentz
      };

      const result = await apiService.analysis.contradictions(params);

      if (result.success) {
        setResults(result.data.data);
        showSuccess('Contradiction analysis completed successfully');
      } else {
        setError(`Analysis failed: ${result.error}`);
        showError(`Analysis failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Error performing contradiction analysis:', error);
      setError(`Analysis error: ${error.message}`);
      showError(`Analysis error: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle export
  const handleExport = (format) => {
    if (!results) return;

    if (onExport) {
      onExport(format, {
        analysis_id: results.analysis_id
      });
    }
  };

  // Get contradiction severity color
  const getSeverityColor = (severity) => {
    if (severity >= 0.8) return 'error';
    if (severity >= 0.5) return 'warning';
    return 'info';
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <CompareIcon sx={{ mr: 1 }} />
          Contradiction Analysis
          <Tooltip title="Identifies contradictory statements in medical literature">
            <InfoIcon fontSize="small" sx={{ ml: 1, color: 'text.secondary' }} />
          </Tooltip>
        </Typography>

        <Divider sx={{ mb: 3 }} />

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {/* Query field */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Search Query
                <Chip size="small" label="Required" color="primary" sx={{ ml: 1 }} />
              </Typography>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="e.g., efficacy of antibiotics in treating community acquired pneumonia"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                required
              />
            </Grid>

            {/* Max results */}
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" gutterBottom>
                Maximum Results: {maxResults}
              </Typography>
              <Slider
                value={maxResults}
                onChange={(_, value) => setMaxResults(value)}
                min={5}
                max={50}
                step={5}
                marks={[
                  { value: 5, label: '5' },
                  { value: 20, label: '20' },
                  { value: 50, label: '50' }
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>

            {/* Threshold */}
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" gutterBottom>
                Contradiction Threshold: {threshold}
              </Typography>
              <Slider
                value={threshold}
                onChange={(_, value) => setThreshold(value)}
                min={0.5}
                max={0.9}
                step={0.05}
                marks={[
                  { value: 0.5, label: '0.5' },
                  { value: 0.7, label: '0.7' },
                  { value: 0.9, label: '0.9' }
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>

            {/* Model options */}
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" gutterBottom>
                Models
              </Typography>
              <Box>
                <FormControlLabel
                  control={
                    <Switch
                      checked={useBioMedLM}
                      onChange={(e) => setUseBioMedLM(e.target.checked)}
                    />
                  }
                  label="BioMedLM"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={useTSMixer}
                      onChange={(e) => setUseTSMixer(e.target.checked)}
                    />
                  }
                  label="TSMixer"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={useLorentz}
                      onChange={(e) => setUseLorentz(e.target.checked)}
                    />
                  }
                  label="Lorentz"
                />
              </Box>
            </Grid>

            {/* Action buttons */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={isAnalyzing ? <ButtonLoader size={20} /> : <ScienceIcon />}
                  disabled={!query.trim() || isAnalyzing}
                >
                  {isAnalyzing ? 'Analyzing...' : 'Analyze Contradictions'}
                </Button>

                <Button
                  variant="outlined"
                  onClick={() => {
                    setQuery('');
                    setResults(null);
                    setError('');
                  }}
                >
                  Clear
                </Button>
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>

      {/* Results */}
      {results && (
        <FadeIn>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Analysis Results
                <Chip
                  size="small"
                  label={`${results.contradictions.length} contradictions found`}
                  color="primary"
                  sx={{ ml: 1 }}
                />
              </Typography>

              <Box>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={() => handleExport('pdf')}
                  sx={{ mr: 1 }}
                >
                  Export as PDF
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={() => handleExport('csv')}
                >
                  Export as CSV
                </Button>
              </Box>
            </Box>

            <Divider sx={{ mb: 3 }} />

            <Typography variant="subtitle1" gutterBottom>
              Query: {results.query}
            </Typography>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Analysis ID: {results.analysis_id} | Date: {new Date(results.timestamp).toLocaleString()}
            </Typography>

            {results.contradictions.length > 0 ? (
              <StaggeredList>
                {results.contradictions.map((contradiction, index) => (
                  <Card key={index} variant="outlined" sx={{ mb: 2 }}>
                    <CardHeader
                      title={`Contradiction #${index + 1}`}
                      subheader={`Severity: ${contradiction.severity.toFixed(2)}`}
                      action={
                        <Chip
                          label={`Severity: ${contradiction.severity.toFixed(2)}`}
                          color={getSeverityColor(contradiction.severity)}
                          size="small"
                        />
                      }
                    />
                    <CardContent>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Statement 1:
                            </Typography>
                            <Typography variant="body2">
                              {contradiction.statement1}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                              Source: {contradiction.source1}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Statement 2:
                            </Typography>
                            <Typography variant="body2">
                              {contradiction.statement2}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                              Source: {contradiction.source2}
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>

                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Explanation:
                        </Typography>
                        <Typography variant="body2">
                          {contradiction.explanation}
                        </Typography>
                      </Box>
                    </CardContent>
                    <CardActions>
                      <Button size="small">View Source 1</Button>
                      <Button size="small">View Source 2</Button>
                    </CardActions>
                  </Card>
                ))}
              </StaggeredList>
            ) : (
              <Alert severity="info">
                No contradictions found for the given query and threshold.
              </Alert>
            )}
          </Paper>
        </FadeIn>
      )}
    </Box>
  );
};

export default ContradictionAnalysis;
