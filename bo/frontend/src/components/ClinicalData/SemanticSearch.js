import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Divider,
  FormControlLabel,
  Grid,
  Switch,
  TextField,
  Typography,
  Alert,
  Chip,
  Link,
  Paper,
  Slider,
  Tooltip,
  IconButton,
  InputAdornment
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import ScienceIcon from '@mui/icons-material/Science';
import PsychologyIcon from '@mui/icons-material/Psychology';
import InfoIcon from '@mui/icons-material/Info';
import BiotechIcon from '@mui/icons-material/Biotech';

// Import the clinical data service
import clinicalDataService from '../../services/clinicalDataService';

/**
 * Semantic Search component
 * Performs semantic search with term expansion to find clinical trials
 */
const SemanticSearch = () => {
  // State
  const [term, setTerm] = useState('');
  const [includeSimilar, setIncludeSimilar] = useState(true);
  const [maxTrials, setMaxTrials] = useState(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);

  // Handle semantic search
  const handleSearch = async () => {
    if (!term.trim()) {
      setError('Please enter a medical term');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await clinicalDataService.findTrialsWithSemanticExpansion(term, includeSimilar, maxTrials);
      setResults(response.data);
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
          Semantic Search with Term Expansion
        </Typography>
        <Tooltip title="This tool expands your search term with semantically similar medical concepts to find more relevant clinical trials">
          <IconButton size="small">
            <InfoIcon fontSize="small" color="primary" />
          </IconButton>
        </Tooltip>
      </Box>
      <Typography variant="body2" paragraph>
        Enter a medical term to find clinical trials with semantic expansion.
        This tool normalizes your search term using SNOMED CT and can expand the search
        to include semantically similar concepts.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper elevation={0} sx={{ p: 3, mb: 4, bgcolor: 'background.paper', borderRadius: 2 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={5}>
            <TextField
              label="Medical Term"
              variant="outlined"
              fullWidth
              value={term}
              onChange={(e) => setTerm(e.target.value)}
              placeholder="e.g., heart attack, high blood pressure"
              helperText="Enter a medical condition, disease, or symptom"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <PsychologyIcon color="action" />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <Typography gutterBottom>
              Maximum Trials: {maxTrials}
            </Typography>
            <Slider
              value={maxTrials}
              onChange={(e, newValue) => setMaxTrials(newValue)}
              min={1}
              max={100}
              valueLabelDisplay="auto"
              aria-labelledby="max-trials-slider"
              color="secondary"
              marks={[
                { value: 1, label: '1' },
                { value: 50, label: '50' },
                { value: 100, label: '100' }
              ]}
            />
          </Grid>
          <Grid item xs={12} md={2} sx={{ display: 'flex', alignItems: 'center' }}>
            <FormControlLabel
              control={
                <Switch
                  checked={includeSimilar}
                  onChange={(e) => setIncludeSimilar(e.target.checked)}
                  color="secondary"
                />
              }
              label={
                <Box>
                  <Typography variant="body2">Include Similar Concepts</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Expand search with related terms
                  </Typography>
                </Box>
              }
            />
          </Grid>
          <Grid item xs={12} md={2} sx={{ display: 'flex', alignItems: 'center' }}>
            <Button
              variant="contained"
              color="primary"
              fullWidth
              onClick={handleSearch}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
              sx={{ height: '56px' }}
            >
              Semantic Search
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Results section */}
      {results && (
        <Box>
          {/* Semantic Expansion Information */}
          <Card sx={{ mb: 3 }}>
            <CardHeader
              title="Semantic Expansion"
              avatar={<PsychologyIcon color="primary" />}
              subheader={`Original search term: "${term}"`}
            />
            <Divider />
            <CardContent>
              <Paper elevation={0} sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 2, mb: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" gutterBottom>
                      <strong>Original Term:</strong> {term}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Normalized Term:</strong> {results.normalized_term || term}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Typography variant="body2" gutterBottom>
                      <strong>Search Terms Used:</strong>
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {results.search_terms_used && results.search_terms_used.map((searchTerm, index) => (
                        <Chip
                          key={index}
                          label={searchTerm}
                          color={searchTerm === results.normalized_term ? 'primary' : 'info'}
                          size="small"
                          variant={searchTerm === results.normalized_term ? 'filled' : 'outlined'}
                        />
                      ))}
                    </Box>
                  </Grid>
                </Grid>
              </Paper>

              <Alert severity="info" icon={<InfoIcon />}>
                Semantic expansion found {results.search_terms_used?.length || 0} related terms to enhance your search results.
              </Alert>
            </CardContent>
          </Card>

          {/* Clinical Trials */}
          <Card>
            <CardHeader
              title={`Clinical Trials (${results.trials ? results.trials.length : 0})`}
              avatar={<ScienceIcon color="primary" />}
              subheader={`Found ${results.trials?.length || 0} trials using semantic expansion`}
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
                                'warning'
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

                          {trial.Condition && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                                Conditions:
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                                {Array.isArray(trial.Condition) ?
                                  trial.Condition.map((condition, i) => (
                                    <Chip
                                      key={i}
                                      label={condition}
                                      size="small"
                                      variant="outlined"
                                      icon={<BiotechIcon fontSize="small" />}
                                    />
                                  )) :
                                  <Chip
                                    label={trial.Condition}
                                    size="small"
                                    variant="outlined"
                                    icon={<BiotechIcon fontSize="small" />}
                                  />
                                }
                              </Box>
                            </Box>
                          )}

                          {trial.LeadSponsorName && (
                            <Typography variant="body2" sx={{ mt: 2 }}>
                              <strong>Sponsor:</strong> {trial.LeadSponsorName}
                            </Typography>
                          )}

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
                  No clinical trials found for this term.
                </Alert>
              )}
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default SemanticSearch;
