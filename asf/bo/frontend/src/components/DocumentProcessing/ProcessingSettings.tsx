import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
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
  IconButton,
  SelectChangeEvent
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import { useMedicalResearchSynthesis } from '../../hooks/useMedicalResearchSynthesis';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import { ButtonLoader } from '../UI/LoadingIndicators';

interface ProcessingSettingsState {
  prefer_pdfminer: boolean;
  use_enhanced_section_classifier: boolean;
  use_gliner: boolean;
  confidence_threshold: number;
  use_hgt: boolean;
  encoder_model: string;
  use_enhanced_summarizer: boolean;
  check_factual_consistency: boolean;
  consistency_method: string;
  consistency_threshold: number;
  use_cache: boolean;
  use_parallel: boolean;
  use_biomedlm: boolean;
}

/**
 * Component for configuring document processing settings
 */
const ProcessingSettings: React.FC = () => {
  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Use the medical research synthesis hook
  const {
    getSynthesisSettings,
    updateSynthesisSettings
  } = useMedicalResearchSynthesis();

  // Get synthesis settings
  const {
    data: settingsData,
    isLoading: isLoadingSettings,
    isError: isErrorSettings,
    error: settingsError,
    refetch: refetchSettings
  } = getSynthesisSettings();

  // Update synthesis settings
  const {
    mutate: updateSettingsMutate,
    isPending: isUpdatingSettings
  } = updateSynthesisSettings();

  // Local state for settings
  const [localSettings, setLocalSettings] = useState<ProcessingSettingsState>({
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
    use_parallel: true,
    use_biomedlm: false
  });

  // Success message state
  const [success, setSuccess] = useState<string | null>(null);

  // Update local settings when API data is loaded
  React.useEffect(() => {
    if (settingsData) {
      setLocalSettings({
        ...localSettings,
        ...settingsData
      });
    }
  }, [settingsData]);

  // Handle settings change
  const handleSettingChange = (setting: keyof ProcessingSettingsState, value: any) => {
    setLocalSettings(prev => ({
      ...prev,
      [setting]: value
    }));

    // Clear success message when settings are changed
    setSuccess(null);
  };

  // Handle slider change
  const handleSliderChange = (setting: keyof ProcessingSettingsState) => (_event: Event, value: number | number[]) => {
    handleSettingChange(setting, value);
  };

  // Handle switch change
  const handleSwitchChange = (setting: keyof ProcessingSettingsState) => (_event: React.ChangeEvent<HTMLInputElement>, checked: boolean) => {
    handleSettingChange(setting, checked);
  };

  // Handle select change
  const handleSelectChange = (setting: keyof ProcessingSettingsState) => (event: SelectChangeEvent<string>) => {
    handleSettingChange(setting, event.target.value);
  };

  // Save settings
  const handleSaveSettings = () => {
    updateSettingsMutate(localSettings, {
      onSuccess: () => {
        setSuccess('Settings saved successfully');
      }
    });
  };

  // Reset settings to defaults
  const handleResetSettings = () => {
    refetchSettings().then(() => {
      if (settingsData) {
        setLocalSettings({
          ...localSettings,
          ...settingsData
        });
        setSuccess('Settings reset to defaults');
      }
    });
  };

  if (useMockData) {
    return (
      <Alert severity="info">
        Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
      </Alert>
    );
  }

  return (
    <Box>
      {isLoadingSettings ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          {isErrorSettings && (
            <Alert 
              severity="error" 
              sx={{ mb: 3 }}
              action={
                <Button color="inherit" size="small" onClick={() => refetchSettings()}>
                  Retry
                </Button>
              }
            >
              Error loading settings: {settingsError?.message || 'Unknown error'}
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
                      checked={localSettings.use_cache}
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
                      checked={localSettings.use_parallel}
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
                        checked={localSettings.prefer_pdfminer}
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
                        checked={localSettings.use_enhanced_section_classifier}
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
                        checked={localSettings.use_gliner}
                        onChange={handleSwitchChange('use_gliner')}
                        color="primary"
                      />
                    }
                    label="Use GLiNER for entity extraction"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Typography id="confidence-threshold-slider" gutterBottom>
                    Confidence Threshold: {localSettings.confidence_threshold}
                  </Typography>
                  <Slider
                    value={localSettings.confidence_threshold}
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
                        checked={localSettings.use_hgt}
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
                      value={localSettings.encoder_model}
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
                        checked={localSettings.use_enhanced_summarizer}
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
                        checked={localSettings.use_biomedlm}
                        onChange={handleSwitchChange('use_biomedlm')}
                        color="primary"
                      />
                    }
                    label={
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography sx={{ mr: 1 }}>Use BiomedLM</Typography>
                        <Tooltip title="Use BiomedLM for improved biomedical summarization">
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
                        checked={localSettings.check_factual_consistency}
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
                      value={localSettings.consistency_method}
                      label="Consistency Method"
                      onChange={handleSelectChange('consistency_method')}
                      disabled={!localSettings.check_factual_consistency}
                    >
                      <MenuItem value="qafacteval">QAFactEval</MenuItem>
                      <MenuItem value="summac">SummaC</MenuItem>
                      <MenuItem value="falsesum">FalseSum</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Typography id="consistency-threshold-slider" gutterBottom>
                    Consistency Threshold: {localSettings.consistency_threshold}
                  </Typography>
                  <Slider
                    value={localSettings.consistency_threshold}
                    onChange={handleSliderChange('consistency_threshold')}
                    aria-labelledby="consistency-threshold-slider"
                    step={0.05}
                    marks
                    min={0.1}
                    max={0.9}
                    valueLabelDisplay="auto"
                    disabled={!localSettings.check_factual_consistency}
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
              disabled={isLoadingSettings || isUpdatingSettings}
            >
              Reset to Defaults
            </Button>

            <Button
              variant="contained"
              color="primary"
              onClick={handleSaveSettings}
              startIcon={isUpdatingSettings ? <ButtonLoader size={20} /> : <SaveIcon />}
              disabled={isLoadingSettings || isUpdatingSettings}
            >
              {isUpdatingSettings ? 'Saving...' : 'Save Settings'}
            </Button>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default ProcessingSettings;
