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
import { useMedicalResearch, TemporalAnalysisParams, TemporalAnalysisResult } from '../../hooks/useMedicalResearch';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import { ButtonLoader } from '../UI/LoadingIndicators';
import { FadeIn, HoverAnimation } from '../UI/Animations';

interface TemporalAnalysisProps {
  onExport?: (format: string, data: any) => void;
}

/**
 * Temporal Analysis component
 *
 * This component allows users to analyze the temporal confidence of medical claims
 * based on publication dates and domain-specific characteristics.
 */
const TemporalAnalysis: React.FC<TemporalAnalysisProps> = ({ onExport }) => {
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Form state
  const [publicationDate, setPublicationDate] = useState<Date | null>(null);
  const [referenceDate, setReferenceDate] = useState<Date | null>(null);
  const [domain, setDomain] = useState<string>('general');
  const [includeDetails, setIncludeDetails] = useState<boolean>(true);

  // UI state
  const [error, setError] = useState<string>('');
  const [result, setResult] = useState<TemporalAnalysisResult | null>(null);

  // Medical research hooks
  const {
    calculateTemporalConfidence
  } = useMedicalResearch();

  // Get the mutation function
  const {
    mutate: executeTemporalAnalysis,
    isPending: isAnalyzing,
    isError: isAnalysisError,
    error: analysisError
  } = calculateTemporalConfidence();

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
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!publicationDate) {
      showError('Please select a publication date');
      setError('Please select a publication date');
      return;
    }

    setError('');

    const params: TemporalAnalysisParams = {
      publication_date: format(publicationDate, 'yyyy-MM-dd'),
      domain,
      include_details: includeDetails
    };

    if (referenceDate) {
      params.reference_date = format(referenceDate, 'yyyy-MM-dd');
    }

    executeTemporalAnalysis(params, {
      onSuccess: (data) => {
        setResult(data);
      },
      onError: (error) => {
        setError(`Analysis failed: ${error.message}`);
      }
    });
  };

  // Handle export
  const handleExport = (format: string) => {
    if (!result) return;

    if (onExport) {
      onExport(format, {
        analysis_id: result.analysis_id
      });
    }
  };

  // Get confidence color
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.5) return 'primary';
    if (confidence >= 0.3) return 'warning';
    return 'error';
  };

  // Get decay rate description
  const getDecayRateDescription = (decayRate: number) => {
    if (decayRate >= 0.8) return 'Very Fast';
    if (decayRate >= 0.5) return 'Fast';
    if (decayRate >= 0.3) return 'Moderate';
    if (decayRate >= 0.1) return 'Slow';
    return 'Very Slow';
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box sx={{ width: '100%' }}>
        {useMockData && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
          </Alert>
        )}

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

          {isAnalysisError && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {analysisError?.message || 'An error occurred during analysis'}
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
                  slotProps={{
                    textField: {
                      fullWidth: true,
                      required: true,
                      helperText: "Date when the medical claim was published"
                    }
                  }}
                />
              </Grid>

              {/* Reference Date */}
              <Grid item xs={12} md={6}>
                <DatePicker
                  label="Reference Date (Optional)"
                  value={referenceDate}
                  onChange={(newValue) => setReferenceDate(newValue)}
                  slotProps={{
                    textField: {
                      fullWidth: true,
                      helperText: "Date to calculate confidence relative to (defaults to today)"
                    }
                  }}
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
                            label={`${(result.confidence_score * 100).toFixed(1)}% Confidence`}
                            color={getConfidenceColor(result.confidence_score)}
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
                              Temporal Relevance:
                            </Typography>
                            <Typography variant="body1">
                              {(result.temporal_relevance * 100).toFixed(1)}%
                            </Typography>
                          </Grid>

                          <Grid item xs={12}>
                            <Typography variant="subtitle2" gutterBottom>
                              Confidence Score: {(result.confidence_score * 100).toFixed(1)}%
                            </Typography>
                            <LinearProgress
                              variant="determinate"
                              value={result.confidence_score * 100}
                              color={getConfidenceColor(result.confidence_score)}
                              sx={{ height: 10, borderRadius: 5, mb: 2 }}
                            />

                            <Typography variant="body2" sx={{ mt: 2 }}>
                              {result.temporal_context}
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
                              title="Knowledge Decay"
                              subheader={getDecayRateDescription(result.details.time_decay)}
                              action={
                                <Chip
                                  label={`${(result.details.time_decay * 100).toFixed(0)}%`}
                                  color={result.details.time_decay >= 0.5 ? 'error' : 'warning'}
                                  size="small"
                                />
                              }
                            />
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Time Decay: {(result.details.time_decay * 100).toFixed(1)}%
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={result.details.time_decay * 100}
                                color="error"
                                sx={{ height: 10, borderRadius: 5, mb: 2 }}
                              />
                            </CardContent>
                          </Card>
                        </HoverAnimation>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <HoverAnimation>
                          <Card variant="outlined">
                            <CardHeader
                              title="Domain Evolution"
                              subheader={`${domainOptions.find(d => d.value === result.domain)?.label || result.domain} evolution`}
                              action={
                                <Chip
                                  label={`${(result.details.domain_evolution * 100).toFixed(0)}%`}
                                  color={result.details.domain_evolution >= 0.7 ? 'success' : 'primary'}
                                  size="small"
                                />
                              }
                            />
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Evolution Rate: {(result.details.domain_evolution * 100).toFixed(1)}%
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={result.details.domain_evolution * 100}
                                color="success"
                                sx={{ height: 10, borderRadius: 5, mb: 2 }}
                              />
                            </CardContent>
                          </Card>
                        </HoverAnimation>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <HoverAnimation>
                          <Card variant="outlined">
                            <CardHeader
                              title="Citation Impact"
                              action={
                                <Chip
                                  label={`${(result.details.citation_impact * 100).toFixed(0)}%`}
                                  color="primary"
                                  size="small"
                                />
                              }
                            />
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Impact: {(result.details.citation_impact * 100).toFixed(1)}%
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={result.details.citation_impact * 100}
                                color="primary"
                                sx={{ height: 10, borderRadius: 5, mb: 2 }}
                              />
                            </CardContent>
                          </Card>
                        </HoverAnimation>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <HoverAnimation>
                          <Card variant="outlined">
                            <CardHeader
                              title="Guideline Changes"
                              action={
                                <Chip
                                  label={`${(result.details.guideline_changes * 100).toFixed(0)}%`}
                                  color="secondary"
                                  size="small"
                                />
                              }
                            />
                            <CardContent>
                              <Typography variant="subtitle2" gutterBottom>
                                Change Rate: {(result.details.guideline_changes * 100).toFixed(1)}%
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={result.details.guideline_changes * 100}
                                color="secondary"
                                sx={{ height: 10, borderRadius: 5, mb: 2 }}
                              />
                            </CardContent>
                          </Card>
                        </HoverAnimation>
                      </Grid>
                    </Grid>
                  </Grid>
                )}

                {/* Recommendations */}
                {result.recommendations && result.recommendations.length > 0 && (
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                      Recommendations
                    </Typography>
                    <Card variant="outlined">
                      <CardContent>
                        <Box component="ul" sx={{ pl: 2 }}>
                          {result.recommendations.map((recommendation, index) => (
                            <Box component="li" key={index} sx={{ mb: 1 }}>
                              <Typography variant="body2">{recommendation}</Typography>
                            </Box>
                          ))}
                        </Box>
                      </CardContent>
                    </Card>
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
