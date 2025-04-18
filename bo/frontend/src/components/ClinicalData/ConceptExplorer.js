import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography,
  Alert,
  Chip,
  Link,
  Paper,
  Slider,
  Tabs,
  Tab,
  Tooltip,
  IconButton,
  InputAdornment,
  LinearProgress
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ScienceIcon from '@mui/icons-material/Science';
import BarChartIcon from '@mui/icons-material/BarChart';
import InfoIcon from '@mui/icons-material/Info';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import NumbersIcon from '@mui/icons-material/Numbers';

// Import the clinical data service
import clinicalDataService from '../../services/clinicalDataService';

/**
 * Concept Explorer component
 * Allows exploring clinical trials related to a specific medical concept
 * and analyzing trial phases for that concept
 */
const ConceptExplorer = () => {
  // State
  const [conceptId, setConceptId] = useState('');
  const [terminology, setTerminology] = useState('SNOMEDCT');
  const [maxTrials, setMaxTrials] = useState(10);
  const [includeDescendants, setIncludeDescendants] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [tabValue, setTabValue] = useState(0);

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Handle search for trials by concept
  const handleSearchTrials = async () => {
    if (!conceptId.trim()) {
      setError('Please enter a concept ID');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await clinicalDataService.getTrialsByConceptId(conceptId, terminology, maxTrials);
      setResults(response.data);
    } catch (err) {
      setError(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle analysis of trial phases
  const handleAnalyzePhases = async () => {
    if (!conceptId.trim()) {
      setError('Please enter a concept ID');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await clinicalDataService.analyzeTrialPhasesByConceptId(
        conceptId,
        terminology,
        includeDescendants,
        500
      );
      setAnalysisResults(response.data);
    } catch (err) {
      setError(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ mr: 1 }}>
          Explore Clinical Trials by Medical Concept
        </Typography>
        <Tooltip title="Enter a SNOMED CT concept ID to explore related clinical trials and analyze their distribution by phase">
          <IconButton size="small">
            <InfoIcon fontSize="small" color="primary" />
          </IconButton>
        </Tooltip>
      </Box>
      <Typography variant="body2" paragraph>
        Enter a SNOMED CT concept ID to find related clinical trials and analyze trial phases.
        This tool helps understand the clinical research landscape for specific medical concepts.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper elevation={0} sx={{ p: 3, mb: 4, bgcolor: 'background.paper', borderRadius: 2 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <TextField
              label="Concept ID"
              variant="outlined"
              fullWidth
              value={conceptId}
              onChange={(e) => setConceptId(e.target.value)}
              placeholder="e.g., 73211009"
              helperText="Enter a SNOMED CT concept ID"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <NumbersIcon color="action" />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Terminology</InputLabel>
              <Select
                value={terminology}
                label="Terminology"
                onChange={(e) => setTerminology(e.target.value)}
              >
                <MenuItem value="SNOMEDCT">SNOMED CT</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Typography gutterBottom>
              Maximum Trials: {maxTrials}
            </Typography>
            <Slider
              value={maxTrials}
              onChange={(e, newValue) => setMaxTrials(newValue)}
              min={1}
              max={50}
              valueLabelDisplay="auto"
              aria-labelledby="max-trials-slider"
              color="secondary"
              marks={[
                { value: 1, label: '1' },
                { value: 25, label: '25' },
                { value: 50, label: '50' }
              ]}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Include Descendants</InputLabel>
              <Select
                value={includeDescendants}
                label="Include Descendants"
                onChange={(e) => setIncludeDescendants(e.target.value)}
              >
                <MenuItem value={true}>Yes</MenuItem>
                <MenuItem value={false}>No</MenuItem>
              </Select>
              <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                Include child concepts in hierarchy
              </Typography>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleSearchTrials}
              disabled={loading}
              startIcon={loading && tabValue === 0 ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
              sx={{ height: '56px', flex: 1 }}
            >
              Find Trials
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleAnalyzePhases}
              disabled={loading}
              startIcon={loading && tabValue === 1 ? <CircularProgress size={20} color="inherit" /> : <BarChartIcon />}
              sx={{ height: '56px', flex: 1 }}
            >
              Analyze
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Results tabs */}
      {(results || analysisResults) && (
        <Card sx={{ width: '100%' }}>
          <CardHeader
            title={results ?
              `Trials for Concept: ${results.concept?.preferredTerm || conceptId}` :
              `Phase Analysis for: ${analysisResults?.concept?.preferredTerm || conceptId}`
            }
            subheader={results?.concept ?
              `SNOMED CT ID: ${results.concept.conceptId}` :
              analysisResults?.concept ? `SNOMED CT ID: ${analysisResults.concept.conceptId}` :
              `Concept ID: ${conceptId}`
            }
          />
          <Divider />
          <Box sx={{ borderBottom: 1, borderColor: 'divider', px: 2 }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              aria-label="concept explorer tabs"
              variant="fullWidth"
            >
              <Tab label="Related Trials" icon={<ScienceIcon />} iconPosition="start" />
              <Tab label="Phase Analysis" icon={<BarChartIcon />} iconPosition="start" />
            </Tabs>
          </Box>

          {/* Trials Tab */}
          <Box role="tabpanel" hidden={tabValue !== 0} sx={{ p: 3 }}>
            {results ? (
              <Box>
                {/* Concept Details */}
                {results.concept && (
                  <Card variant="outlined" sx={{ mb: 3 }}>
                    <CardHeader title="Concept Details" />
                    <Divider />
                    <CardContent>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Chip
                              icon={<LocalHospitalIcon />}
                              label={results.concept.conceptId}
                              color="secondary"
                              variant="outlined"
                              sx={{ mr: 1 }}
                            />
                          </Box>
                          <Typography variant="body1" gutterBottom>
                            <strong>Preferred Term:</strong> {results.concept.preferredTerm}
                          </Typography>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          {results.concept.definition && (
                            <Typography variant="body2">
                              <strong>Definition:</strong> {results.concept.definition}
                            </Typography>
                          )}
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                )}

                {/* Clinical Trials */}
                <Card variant="outlined">
                  <CardHeader
                    title="Related Clinical Trials"
                    avatar={<ScienceIcon color="primary" />}
                    subheader={`Found ${results.trials?.length || 0} trials related to this concept`}
                  />
                  <Divider />
                  <CardContent>
                    {results.trials && results.trials.length > 0 ? (
                      <Grid container spacing={2}>
                        {results.trials.map((trial, index) => (
                          <Grid item xs={12} key={trial.NCTId || index}>
                            <Card variant="outlined" sx={{ bgcolor: '#f8f9fa' }}>
                              <CardContent>
                                <Typography variant="subtitle2" color="primary" gutterBottom>
                                  {trial.BriefTitle}
                                </Typography>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                                  <Chip
                                    label={trial.OverallStatus || 'Unknown Status'}
                                    size="small"
                                    color={
                                      trial.OverallStatus === 'Recruiting' ? 'success' :
                                      trial.OverallStatus === 'Completed' ? 'primary' :
                                      trial.OverallStatus === 'Active, not recruiting' ? 'info' :
                                      'default'
                                    }
                                    variant="outlined"
                                  />
                                  {trial.Phase && (
                                    <Chip label={trial.Phase} size="small" color="secondary" variant="outlined" />
                                  )}
                                  {trial.EnrollmentCount && (
                                    <Chip label={`${trial.EnrollmentCount} participants`} size="small" variant="outlined" />
                                  )}
                                </Box>
                                <Box sx={{ mt: 2 }}>
                                  <Button
                                    variant="outlined"
                                    size="small"
                                    startIcon={<ScienceIcon />}
                                    component={Link}
                                    href={`https://clinicaltrials.gov/study/${trial.NCTId}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                  >
                                    View on ClinicalTrials.gov
                                  </Button>
                                </Box>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    ) : (
                      <Alert severity="info">
                        No clinical trials found for this concept.
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Box>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <Typography>Please search for a concept to view related trials</Typography>
              </Box>
            )}
          </Box>

          {/* Phase Analysis Tab */}
          <Box role="tabpanel" hidden={tabValue !== 1} sx={{ p: 3 }}>
            {analysisResults ? (
              <Box>
                {/* Phase Distribution */}
                <Card variant="outlined" sx={{ mb: 3 }}>
                  <CardHeader
                    title="Trial Phase Distribution"
                    subheader={`Analysis based on ${analysisResults.total_trials || 0} clinical trials`}
                  />
                  <Divider />
                  <CardContent>
                    {analysisResults.phase_distribution ? (
                      <Box>
                        <Grid container spacing={3} sx={{ mt: 1 }}>
                          {Object.entries(analysisResults.phase_distribution).map(([phase, count]) => (
                            <Grid item xs={12} sm={6} md={4} lg={3} key={phase}>
                              <Card
                                variant="outlined"
                                sx={{
                                  height: '100%',
                                  bgcolor: phase === 'Phase 3' ? '#e3f2fd' :
                                          phase === 'Phase 4' ? '#e8f5e9' :
                                          phase === 'Phase 1' ? '#fff3e0' :
                                          phase === 'Phase 2' ? '#e0f7fa' :
                                          'background.paper'
                                }}
                              >
                                <CardContent>
                                  <Typography variant="h4" align="center" color="primary" gutterBottom>
                                    {count}
                                  </Typography>
                                  <Typography variant="body1" color="text.secondary" align="center" gutterBottom>
                                    {phase}
                                  </Typography>
                                  <LinearProgress
                                    variant="determinate"
                                    value={(count / analysisResults.total_trials) * 100}
                                    color={
                                      phase === 'Phase 3' ? 'primary' :
                                      phase === 'Phase 4' ? 'success' :
                                      'secondary'
                                    }
                                    sx={{ height: 8, borderRadius: 4, mt: 1 }}
                                  />
                                  <Typography variant="caption" align="center" display="block" sx={{ mt: 1 }}>
                                    {Math.round((count / analysisResults.total_trials) * 100)}% of trials
                                  </Typography>
                                </CardContent>
                              </Card>
                            </Grid>
                          ))}
                        </Grid>
                      </Box>
                    ) : (
                      <Alert severity="info">
                        No phase distribution data available.
                      </Alert>
                    )}
                  </CardContent>
                </Card>

                {/* Additional Analysis */}
                {analysisResults.status_distribution && (
                  <Card variant="outlined">
                    <CardHeader title="Trial Status Distribution" />
                    <Divider />
                    <CardContent>
                      <Grid container spacing={2}>
                        {Object.entries(analysisResults.status_distribution).map(([status, count]) => (
                          <Grid item xs={6} sm={4} md={3} key={status}>
                            <Card variant="outlined" sx={{ textAlign: 'center', p: 2 }}>
                              <Typography variant="body2" color="text.secondary" gutterBottom>
                                {status}
                              </Typography>
                              <Typography variant="h6" color={
                                status === 'Recruiting' ? 'success.main' :
                                status === 'Completed' ? 'primary.main' :
                                status === 'Active, not recruiting' ? 'info.main' :
                                'text.primary'
                              }>
                                {count}
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={(count / analysisResults.total_trials) * 100}
                                color={
                                  status === 'Recruiting' ? 'success' :
                                  status === 'Completed' ? 'primary' :
                                  status === 'Active, not recruiting' ? 'info' :
                                  'secondary'
                                }
                                sx={{ mt: 1 }}
                              />
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </CardContent>
                  </Card>
                )}
              </Box>
            ) : (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                <Typography>Please analyze a concept to view phase distribution</Typography>
              </Box>
            )}
          </Box>
        </Card>
      )}
    </Box>
  );
};

export default ConceptExplorer;
