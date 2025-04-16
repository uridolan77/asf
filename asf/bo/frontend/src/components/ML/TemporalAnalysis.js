import React, { useState } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid, Chip,
  FormControlLabel, Switch, Divider, CircularProgress,
  Card, CardContent, CardHeader, CardActions, Tooltip, Alert,
  LinearProgress, FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import {
  AccessTime as AccessTimeIcon,
  Science as ScienceIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  CalendarToday as CalendarIcon,
  Timeline as TimelineIcon
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { format } from 'date-fns';

import { useNotification } from '../../context/NotificationContext';
// Import apiService as a fallback
import defaultApiService from '../../services/api';
import { ButtonLoader } from '../UI/LoadingIndicators.js';
import { FadeIn, HoverAnimation } from '../UI/Animations.js';

/**
 * Temporal Analysis component
 *
 * This component allows users to analyze the temporal confidence of medical claims
 * based on publication dates and domain-specific characteristics.
 */
const TemporalAnalysis = ({ onExport, apiService }) => {
  // Use provided apiService or fall back to the default
  const api = apiService || defaultApiService;
  const { showSuccess, showError } = useNotification();

  // Form state
  const [publicationDate, setPublicationDate] = useState(null);
  const [referenceDate, setReferenceDate] = useState(null);
  const [domain, setDomain] = useState('general');
  const [includeDetails, setIncludeDetails] = useState(true);

  // UI state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  // Domain options
  const domainOptions = [
    { value: 'general', label: 'General Medicine' },
    { value: 'cardiology', label: 'Cardiology' },
    { value: 'oncology', label: 'Oncology' },
    { value: 'neurology', label: 'Neurology' },
    { value: 'infectious_disease', label: 'Infectious Disease' },
    { value: 'pediatrics', label: 'Pediatrics' },
    { value: 'psychiatry', label: 'Psychiatry' },
    { value: 'surgery', label: 'Surgery' }
  ];

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!publicationDate) {
      showError('Please select a publication date');
      setError('Please select a publication date');
      return;
    }

    setIsAnalyzing(true);
    setError('');

    try {
      const params = {
        publication_date: format(publicationDate, 'yyyy-MM-dd'),
        domain,
        include_details: includeDetails
      };

      if (referenceDate) {
        params.reference_date = format(referenceDate, 'yyyy-MM-dd');
      }

      const result = await api.ml.calculateTemporalConfidence(params);

      if (result.success) {
        setResult(result.data);
        showSuccess('Temporal analysis completed successfully');
      } else {
        setError(`Analysis failed: ${result.error}`);
        showError(`Analysis failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Error calculating temporal confidence:', error);
      setError(`Analysis error: ${error.message}`);
      showError(`Analysis error: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle export
  const handleExport = (format) => {
    if (!result) return;

    if (onExport) {
      onExport(format, {
        analysis_id: result.analysis_id
      });
    }
  };

  // Get confidence color
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.5) return 'primary';
    if (confidence >= 0.3) return 'warning';
    return 'error';
  };

  // Get decay rate description
  const getDecayRateDescription = (decayRate) => {
    if (decayRate >= 0.8) return 'Very Fast';
    if (decayRate >= 0.5) return 'Fast';
    if (decayRate >= 0.3) return 'Moderate';
    if (decayRate >= 0.1) return 'Slow';
    return 'Very Slow';
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box sx={{ width: '100%' }}>
        <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <AccessTimeIcon sx={{ mr: 1 }} />
            Temporal Confidence Analysis
            <Tooltip title="Analyze the temporal confidence of medical claims based on publication dates and domain-specific characteristics">
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
              {/* Publication Date */}
              <Grid item xs={12} md={6}>
                <DatePicker
                  label="Publication Date"
                  value={publicationDate}
                  onChange={(newValue) => setPublicationDate(newValue)}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      fullWidth
                      required
                      helperText="Date when the medical claim was published"
                    />
                  )}
                />
              </Grid>

              {/* Reference Date */}
              <Grid item xs={12} md={6}>
                <DatePicker
                  label="Reference Date (Optional)"
                  value={referenceDate}
                  onChange={(newValue) => setReferenceDate(newValue)}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      fullWidth
                      helperText="Date to calculate confidence relative to (defaults to today)"
                    />
                  )}
                />
              </Grid>

              {/* Domain */}
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel id="domain-label">Medical Domain</InputLabel>
                  <Select
                    labelId="domain-label"
                    value={domain}
                    label="Medical Domain"
                    onChange={(e) => setDomain(e.target.value)}
                  >
                    {domainOptions.map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              {/* Include Details */}
              <Grid item xs={12} md={6}>
                <Box sx={{ display: 'flex', alignItems: 'center', height: '100%' }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={includeDetails}
                        onChange={(e) => setIncludeDetails(e.target.checked)}
                      />
                    }
                    label="Include Detailed Analysis"
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
                    startIcon={isAnalyzing ? <ButtonLoader size={20} /> : <TimelineIcon />}
                    disabled={!publicationDate || isAnalyzing}
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Calculate Temporal Confidence'}
                  </Button>

                  <Button
                    variant="outlined"
                    onClick={() => {
                      setPublicationDate(null);
                      setReferenceDate(null);
                      setDomain('general');
                      setResult(null);
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
        {result && (
          <FadeIn>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Temporal Confidence Results
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
                    onClick={() => handleExport('json')}
                  >
                    Export as JSON
                  </Button>
                </Box>
              </Box>

              <Divider sx={{ mb: 3 }} />

              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <HoverAnimation>
                    <Card variant="outlined">
                      <CardHeader
                        title="Temporal Confidence Assessment"
                        subheader={`Analysis ID: ${result.analysis_id}`}
                        action={
                          <Chip
                            label={`${(result.confidence * 100).toFixed(1)}% Confidence`}
                            color={getConfidenceColor(result.confidence)}
                          />
                        }
                      />
                      <CardContent>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <CalendarIcon sx={{ mr: 1, color: 'text.secondary' }} />
                              <Typography variant="subtitle2">
                                Publication Date:
                              </Typography>
                            </Box>
                            <Typography variant="body1" sx={{ mb: 2 }}>
                              {new Date(result.publication_date).toLocaleDateString()}
                            </Typography>

                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                              <AccessTimeIcon sx={{ mr: 1, color: 'text.secondary' }} />
                              <Typography variant="subtitle2">
                                Reference Date:
                              </Typography>
                            </Box>
                            <Typography variant="body1">
                              {new Date(result.reference_date).toLocaleDateString()}
                            </Typography>
                          </Grid>

                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Domain:
                            </Typography>
                            <Typography variant="body1" sx={{ mb: 2 }}>
                              {domainOptions.find(d => d.value === result.domain)?.label || result.domain}
                            </Typography>

                            <Typography variant="subtitle2" gutterBottom>
                              Time Elapsed:
                            </Typography>
                            <Typography variant="body1">
                              {result.years_elapsed.toFixed(1)} years
                            </Typography>
                          </Grid>

                          <Grid item xs={12}>
                            <Typography variant="subtitle2" gutterBottom>
                              Temporal Confidence: {(result.confidence * 100).toFixed(1)}%
                            </Typography>
                            <LinearProgress
                              variant="determinate"
                              value={result.confidence * 100}
                              color={getConfidenceColor(result.confidence)}
                              sx={{ height: 10, borderRadius: 5, mb: 2 }}
                            />

                            <Typography variant="body2" sx={{ mt: 2 }}>
                              {result.explanation}
                            </Typography>
                          </Grid>
                        </Grid>
                      </CardContent>
                    </Card>
                  </HoverAnimation>
                </Grid>

                {/* Detailed Analysis */}
                {result.details && (
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                      Detailed Analysis
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <HoverAnimation>
                          <Card variant="outlined">
                            <CardHeader
                              title="Knowledge Decay Rate"
                              subheader={getDecayRateDescription(result.details.decay_rate)}
                              action={
                                <Chip
                                  label={`${(result.details.decay_rate * 100).toFixed(0)}%`}
                                  color={result.details.decay_rate >= 0.5 ? 'error' : 'warning'}
                                  size="small"
                                />
                              }
                            />
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Decay Rate: {(result.details.decay_rate * 100).toFixed(1)}%
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={result.details.decay_rate * 100}
                                color="error"
                                sx={{ height: 10, borderRadius: 5, mb: 2 }}
                              />

                              <Typography variant="body2">
                                {result.details.decay_explanation}
                              </Typography>
                            </CardContent>
                          </Card>
                        </HoverAnimation>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <HoverAnimation>
                          <Card variant="outlined">
                            <CardHeader
                              title="Domain Stability"
                              subheader={`${result.domain} domain stability`}
                              action={
                                <Chip
                                  label={`${(result.details.domain_stability * 100).toFixed(0)}%`}
                                  color={result.details.domain_stability >= 0.7 ? 'success' : 'primary'}
                                  size="small"
                                />
                              }
                            />
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Stability: {(result.details.domain_stability * 100).toFixed(1)}%
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={result.details.domain_stability * 100}
                                color="success"
                                sx={{ height: 10, borderRadius: 5, mb: 2 }}
                              />

                              <Typography variant="body2">
                                {result.details.stability_explanation}
                              </Typography>
                            </CardContent>
                          </Card>
                        </HoverAnimation>
                      </Grid>

                      {result.details.recent_developments && (
                        <Grid item xs={12}>
                          <HoverAnimation>
                            <Card variant="outlined">
                              <CardHeader
                                title="Recent Developments"
                                subheader="Impact on temporal confidence"
                              />
                              <CardContent>
                                <Typography variant="body2" sx={{ mb: 2 }}>
                                  {result.details.developments_explanation}
                                </Typography>

                                {result.details.recent_developments.map((development, index) => (
                                  <Box key={index} sx={{ mb: 2 }}>
                                    <Typography variant="subtitle2" gutterBottom>
                                      {development.title}
                                    </Typography>
                                    <Typography variant="body2" sx={{ mb: 1 }}>
                                      {development.description}
                                    </Typography>
                                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                      <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
                                        Impact:
                                      </Typography>
                                      <Chip
                                        label={development.impact}
                                        color={development.impact === 'High' ? 'error' :
                                               development.impact === 'Medium' ? 'warning' : 'info'}
                                        size="small"
                                      />
                                    </Box>
                                  </Box>
                                ))}
                              </CardContent>
                            </Card>
                          </HoverAnimation>
                        </Grid>
                      )}
                    </Grid>
                  </Grid>
                )}
              </Grid>
            </Paper>
          </FadeIn>
        )}
      </Box>
    </LocalizationProvider>
  );
};

export default TemporalAnalysis;
