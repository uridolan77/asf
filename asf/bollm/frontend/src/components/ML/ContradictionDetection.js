import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid, Chip,
  FormControlLabel, Switch, Slider, Divider, CircularProgress,
  Card, CardContent, CardHeader, CardActions, Tooltip, Alert,
  LinearProgress
} from '@mui/material';
import {
  Compare as CompareIcon,
  Science as ScienceIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  AccessTime as AccessTimeIcon,
  Layers as LayersIcon,
  Psychology as PsychologyIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext.jsx';
// Import apiService as a fallback
import defaultApiService from '../../services/api';
import { ButtonLoader } from '../UI/LoadingIndicators.js';
import { FadeIn, StaggeredList, HoverAnimation } from '../UI/Animations.js';

/**
 * Contradiction Detection component
 *
 * This component allows users to detect contradictions between two medical claims
 * using various ML models.
 */
const ContradictionDetection = ({ onExport, api, onProcessingStateChange }) => {
  // Use provided apiService or fall back to the default
  const apiService = api || defaultApiService;
  const { showSuccess, showError } = useNotification();

  // Form state
  const [claim1, setClaim1] = useState('');
  const [claim2, setClaim2] = useState('');
  const [threshold, setThreshold] = useState(0.7);
  const [useBioMedLM, setUseBioMedLM] = useState(true);
  const [useTSMixer, setUseTSMixer] = useState(false);
  const [useLorentz, setUseLorentz] = useState(false);
  const [useTemporal, setUseTemporal] = useState(false);
  const [useShap, setUseShap] = useState(false);
  const [domain, setDomain] = useState('');

  // UI state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [abortController, setAbortController] = useState(null);

  // Mark component as initialized after mount
  useEffect(() => {
    setIsInitialized(true);
    return () => {
      // Abort any in-progress requests when component unmounts
      if (abortController) {
        abortController.abort();
      }
    };
  }, []);

  // Update parent component when processing state changes
  useEffect(() => {
    if (onProcessingStateChange) {
      onProcessingStateChange(isAnalyzing);
    }
  }, [isAnalyzing, onProcessingStateChange]);

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!claim1.trim() || !claim2.trim()) {
      showError('Please enter both claims');
      setError('Please enter both claims');
      return;
    }

    setIsAnalyzing(true);
    setError('');

    // Create a new AbortController for this request
    const controller = new AbortController();
    setAbortController(controller);

    try {
      const params = {
        claim1: claim1.trim(),
        claim2: claim2.trim(),
        threshold,
        use_biomedlm: useBioMedLM,
        use_tsmixer: useTSMixer,
        use_lorentz: useLorentz,
        use_temporal: useTemporal,
        use_shap: useShap,
        domain: domain.trim() || null
      };

      // Pass the signal to the API call
      const result = await apiService.ml.detectContradiction(params, controller.signal);

      if (result.success) {
        setResult(result.data);
        showSuccess('Contradiction detection completed successfully');
      } else {
        setError(`Analysis failed: ${result.error}`);
        showError(`Analysis failed: ${result.error}`);
      }
    } catch (error) {
      // Don't show errors for aborted requests
      if (error.name !== 'AbortError') {
        console.error('Error detecting contradiction:', error);
        setError(`Analysis error: ${error.message}`);
        showError(`Analysis error: ${error.message}`);
      }
    } finally {
      setIsAnalyzing(false);
      setAbortController(null);
    }
  };

  // Cancel ongoing analysis
  const handleCancel = () => {
    if (abortController) {
      abortController.abort();
      setIsAnalyzing(false);
      setAbortController(null);
      showSuccess('Analysis canceled');
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

  // Get contradiction severity color
  const getContradictionColor = (score) => {
    if (score >= 0.8) return 'error';
    if (score >= 0.5) return 'warning';
    return 'success';
  };

  // Get model contribution color
  const getModelContributionColor = (contribution) => {
    if (contribution >= 0.7) return 'primary';
    if (contribution >= 0.4) return 'secondary';
    return 'default';
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <CompareIcon sx={{ mr: 1 }} />
          Contradiction Detection
          <Tooltip title="Detect contradictions between two medical claims using various ML models">
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
            {/* Claim 1 field */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Claim 1
                <Chip size="small" label="Required" color="primary" sx={{ ml: 1 }} />
              </Typography>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="e.g., Amoxicillin is effective for treating community-acquired pneumonia"
                value={claim1}
                onChange={(e) => setClaim1(e.target.value)}
                required
                multiline
                rows={3}
              />
            </Grid>

            {/* Claim 2 field */}
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Claim 2
                <Chip size="small" label="Required" color="primary" sx={{ ml: 1 }} />
              </Typography>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="e.g., Antibiotics are not recommended for community-acquired pneumonia"
                value={claim2}
                onChange={(e) => setClaim2(e.target.value)}
                required
                multiline
                rows={3}
              />
            </Grid>

            {/* Domain field */}
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" gutterBottom>
                Domain (Optional)
              </Typography>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="e.g., cardiology, oncology"
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
              />
            </Grid>

            {/* Threshold */}
            <Grid item xs={12} md={8}>
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
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Models
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={useBioMedLM}
                      onChange={(e) => setUseBioMedLM(e.target.checked)}
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <PsychologyIcon fontSize="small" sx={{ mr: 0.5 }} />
                      BioMedLM
                    </Box>
                  }
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={useTSMixer}
                      onChange={(e) => setUseTSMixer(e.target.checked)}
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <AccessTimeIcon fontSize="small" sx={{ mr: 0.5 }} />
                      TSMixer
                    </Box>
                  }
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={useLorentz}
                      onChange={(e) => setUseLorentz(e.target.checked)}
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <LayersIcon fontSize="small" sx={{ mr: 0.5 }} />
                      Lorentz
                    </Box>
                  }
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={useTemporal}
                      onChange={(e) => setUseTemporal(e.target.checked)}
                    />
                  }
                  label="Temporal Analysis"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={useShap}
                      onChange={(e) => setUseShap(e.target.checked)}
                    />
                  }
                  label="SHAP Explanations"
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
                  startIcon={isAnalyzing ? <ButtonLoader size={20} /> : <CompareIcon />}
                  disabled={!claim1.trim() || !claim2.trim() || isAnalyzing}
                >
                  {isAnalyzing ? 'Analyzing...' : 'Detect Contradiction'}
                </Button>

                <Button
                  variant="outlined"
                  onClick={() => {
                    setClaim1('');
                    setClaim2('');
                    setDomain('');
                    setResult(null);
                    setError('');
                  }}
                >
                  Clear
                </Button>

                {isAnalyzing && (
                  <Button
                    variant="outlined"
                    color="secondary"
                    onClick={handleCancel}
                  >
                    Cancel
                  </Button>
                )}
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
                Contradiction Analysis Results
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
                      title="Contradiction Assessment"
                      subheader={`Analysis ID: ${result.analysis_id}`}
                      action={
                        <Chip
                          label={result.is_contradiction ? 'Contradiction Detected' : 'No Contradiction'}
                          color={result.is_contradiction ? 'error' : 'success'}
                        />
                      }
                    />
                    <CardContent>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Claim 1:
                            </Typography>
                            <Typography variant="body2">
                              {result.claim1}
                            </Typography>
                          </Paper>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Claim 2:
                            </Typography>
                            <Typography variant="body2">
                              {result.claim2}
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>

                      <Box sx={{ mt: 3 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Contradiction Score: {(result.contradiction_score * 100).toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={result.contradiction_score * 100}
                          color={getContradictionColor(result.contradiction_score)}
                          sx={{ height: 10, borderRadius: 5, mb: 2 }}
                        />

                        {result.explanation && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Explanation:
                            </Typography>
                            <Typography variant="body2">
                              {result.explanation}
                            </Typography>
                          </Box>
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </HoverAnimation>
              </Grid>

              {/* Model Contributions */}
              {result.model_contributions && (
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Model Contributions
                  </Typography>
                  <Grid container spacing={2}>
                    {Object.entries(result.model_contributions).map(([model, contribution], index) => (
                      <Grid item xs={12} md={4} key={index}>
                        <HoverAnimation>
                          <Card variant="outlined">
                            <CardHeader
                              title={model}
                              subheader={`Contribution: ${(contribution * 100).toFixed(1)}%`}
                              action={
                                <Chip
                                  label={`${(contribution * 100).toFixed(0)}%`}
                                  color={getModelContributionColor(contribution)}
                                  size="small"
                                />
                              }
                            />
                            <CardContent>
                              <LinearProgress
                                variant="determinate"
                                value={contribution * 100}
                                color={getModelContributionColor(contribution)}
                                sx={{ height: 10, borderRadius: 5 }}
                              />
                            </CardContent>
                          </Card>
                        </HoverAnimation>
                      </Grid>
                    ))}
                  </Grid>
                </Grid>
              )}

              {/* Temporal Analysis */}
              {result.temporal_analysis && (
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Temporal Analysis
                  </Typography>
                  <HoverAnimation>
                    <Card variant="outlined">
                      <CardHeader
                        title="Temporal Contradiction Assessment"
                        subheader={result.temporal_analysis.is_temporal_contradiction ?
                          'Temporal contradiction detected' : 'No temporal contradiction detected'}
                        action={
                          <Chip
                            label={result.temporal_analysis.is_temporal_contradiction ?
                              'Temporal Contradiction' : 'No Temporal Contradiction'}
                            color={result.temporal_analysis.is_temporal_contradiction ? 'error' : 'success'}
                            size="small"
                          />
                        }
                      />
                      <CardContent>
                        <Grid container spacing={2}>
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Claim 1 Temporal Context:
                            </Typography>
                            <Typography variant="body2">
                              {result.temporal_analysis.claim1_temporal_context || 'No temporal context detected'}
                            </Typography>
                          </Grid>
                          <Grid item xs={12} md={6}>
                            <Typography variant="subtitle2" gutterBottom>
                              Claim 2 Temporal Context:
                            </Typography>
                            <Typography variant="body2">
                              {result.temporal_analysis.claim2_temporal_context || 'No temporal context detected'}
                            </Typography>
                          </Grid>
                        </Grid>

                        {result.temporal_analysis.explanation && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Temporal Explanation:
                            </Typography>
                            <Typography variant="body2">
                              {result.temporal_analysis.explanation}
                            </Typography>
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  </HoverAnimation>
                </Grid>
              )}

              {/* SHAP Explanations */}
              {result.shap_explanations && (
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    SHAP Explanations
                  </Typography>
                  <HoverAnimation>
                    <Card variant="outlined">
                      <CardHeader
                        title="Feature Importance"
                        subheader="SHAP values for key features"
                      />
                      <CardContent>
                        <Typography variant="body2" sx={{ mb: 2 }}>
                          The following features contributed most to the contradiction assessment:
                        </Typography>

                        {result.shap_explanations.features.map((feature, index) => (
                          <Box key={index} sx={{ mb: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>
                              {feature.name}: {feature.importance.toFixed(3)}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <Box sx={{ width: '100%', mr: 1 }}>
                                <LinearProgress
                                  variant="determinate"
                                  value={Math.abs(feature.importance) * 100}
                                  color={feature.importance > 0 ? 'error' : 'success'}
                                  sx={{ height: 10, borderRadius: 5 }}
                                />
                              </Box>
                              <Box sx={{ minWidth: 35 }}>
                                <Typography variant="body2" color="text.secondary">
                                  {feature.importance > 0 ? 'Supports' : 'Opposes'} contradiction
                                </Typography>
                              </Box>
                            </Box>
                          </Box>
                        ))}
                      </CardContent>
                    </Card>
                  </HoverAnimation>
                </Grid>
              )}
            </Grid>
          </Paper>
        </FadeIn>
      )}
    </Box>
  );
};

export default ContradictionDetection;
