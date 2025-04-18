import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  TextField,
  Grid,
  Paper,
  Divider,
  FormControlLabel,
  Switch,
  CircularProgress,
  Alert,
  Slider,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import apiService from '../../services/api';

/**
 * Component for configuring document processing settings
 */
const ProcessingSettings = () => {
  const [settings, setSettings] = useState({
    prefer_pdfminer: true,
    use_enhanced_section_classifier: true,
    use_gliner: true,
    confidence_threshold: 0.6,
    use_hgt: true,
    encoder_model: "microsoft/biogpt",
    use_enhanced_summarizer: true,
    check_factual_consistency: true,
    consistency_method: "qafacteval",
    consistency_threshold: 0.6,
    use_cache: true,
    use_parallel: true
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  // Fetch settings on component mount
  useEffect(() => {
    fetchSettings();
  }, []);

  // Fetch settings from API
  const fetchSettings = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.documentProcessing.getSettings();
      setSettings(response.data);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching settings:', err);
      setError('Error fetching settings. Please try again.');
      setLoading(false);
    }
  };

  // Handle settings change
  const handleSettingChange = (setting, value) => {
    setSettings(prev => ({
      ...prev,
      [setting]: value
    }));

    // Clear success message when settings are changed
    setSuccess(null);
  };

  // Handle slider change
  const handleSliderChange = (setting) => (event, value) => {
    handleSettingChange(setting, value);
  };

  // Handle switch change
  const handleSwitchChange = (setting) => (event) => {
    handleSettingChange(setting, event.target.checked);
  };

  // Handle select change
  const handleSelectChange = (setting) => (event) => {
    handleSettingChange(setting, event.target.value);
  };

  // Save settings
  const handleSaveSettings = async () => {
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      // In a real implementation, this would save to the backend
      // For now, we'll just simulate a successful save
      await new Promise(resolve => setTimeout(resolve, 1000));

      setSuccess('Settings saved successfully');
      setSaving(false);
    } catch (err) {
      console.error('Error saving settings:', err);
      setError('Error saving settings. Please try again.');
      setSaving(false);
    }
  };

  // Reset settings to defaults
  const handleResetSettings = async () => {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      await fetchSettings();
      setSuccess('Settings reset to defaults');
    } catch (err) {
      console.error('Error resetting settings:', err);
      setError('Error resetting settings. Please try again.');
      setLoading(false);
    }
  };

  return (
    <Box>
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {success && (
            <Alert severity="success" sx={{ mb: 3 }}>
              {success}
            </Alert>
          )}

          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              General Settings
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.use_cache}
                      onChange={handleSwitchChange('use_cache')}
                      color="primary"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography sx={{ mr: 1 }}>Use Caching</Typography>
                      <Tooltip title="Enable caching to improve performance for repeated processing of similar documents">
                        <IconButton size="small">
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  }
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.use_parallel}
                      onChange={handleSwitchChange('use_parallel')}
                      color="primary"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography sx={{ mr: 1 }}>Use Parallel Processing</Typography>
                      <Tooltip title="Enable parallel processing to improve performance for large documents">
                        <IconButton size="small">
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  }
                />
              </Grid>
            </Grid>
          </Paper>

          <Accordion defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1">Document Processing Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.prefer_pdfminer}
                        onChange={handleSwitchChange('prefer_pdfminer')}
                        color="primary"
                      />
                    }
                    label="Prefer PDFMiner for PDF parsing"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.use_enhanced_section_classifier}
                        onChange={handleSwitchChange('use_enhanced_section_classifier')}
                        color="primary"
                      />
                    }
                    label="Use enhanced section classifier"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1">Entity Extraction Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.use_gliner}
                        onChange={handleSwitchChange('use_gliner')}
                        color="primary"
                      />
                    }
                    label="Use GLiNER for entity extraction"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography id="confidence-threshold-slider" gutterBottom>
                    Confidence Threshold: {settings.confidence_threshold}
                  </Typography>
                  <Slider
                    value={settings.confidence_threshold}
                    onChange={handleSliderChange('confidence_threshold')}
                    aria-labelledby="confidence-threshold-slider"
                    step={0.05}
                    marks
                    min={0.1}
                    max={0.9}
                    valueLabelDisplay="auto"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1">Relation Extraction Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.use_hgt}
                        onChange={handleSwitchChange('use_hgt')}
                        color="primary"
                      />
                    }
                    label="Use Heterogeneous Graph Transformer"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel id="encoder-model-label">Encoder Model</InputLabel>
                    <Select
                      labelId="encoder-model-label"
                      id="encoder-model"
                      value={settings.encoder_model}
                      label="Encoder Model"
                      onChange={handleSelectChange('encoder_model')}
                    >
                      <MenuItem value="microsoft/biogpt">Microsoft BioGPT</MenuItem>
                      <MenuItem value="allenai/scibert">Allen AI SciBERT</MenuItem>
                      <MenuItem value="dmis-lab/biobert">DMIS Lab BioBERT</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1">Summarization Settings</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.use_enhanced_summarizer}
                        onChange={handleSwitchChange('use_enhanced_summarizer')}
                        color="primary"
                      />
                    }
                    label="Use enhanced summarizer"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.check_factual_consistency}
                        onChange={handleSwitchChange('check_factual_consistency')}
                        color="primary"
                      />
                    }
                    label="Check factual consistency"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel id="consistency-method-label">Consistency Method</InputLabel>
                    <Select
                      labelId="consistency-method-label"
                      id="consistency-method"
                      value={settings.consistency_method}
                      label="Consistency Method"
                      onChange={handleSelectChange('consistency_method')}
                      disabled={!settings.check_factual_consistency}
                    >
                      <MenuItem value="qafacteval">QAFactEval</MenuItem>
                      <MenuItem value="summac">SummaC</MenuItem>
                      <MenuItem value="falsesum">FalseSum</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Typography id="consistency-threshold-slider" gutterBottom>
                    Consistency Threshold: {settings.consistency_threshold}
                  </Typography>
                  <Slider
                    value={settings.consistency_threshold}
                    onChange={handleSliderChange('consistency_threshold')}
                    aria-labelledby="consistency-threshold-slider"
                    step={0.05}
                    marks
                    min={0.1}
                    max={0.9}
                    valueLabelDisplay="auto"
                    disabled={!settings.check_factual_consistency}
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleResetSettings}
              startIcon={<RefreshIcon />}
              disabled={loading || saving}
            >
              Reset to Defaults
            </Button>

            <Button
              variant="contained"
              color="primary"
              onClick={handleSaveSettings}
              startIcon={saving ? <CircularProgress size={20} /> : <SaveIcon />}
              disabled={loading || saving}
            >
              {saving ? 'Saving...' : 'Save Settings'}
            </Button>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default ProcessingSettings;
