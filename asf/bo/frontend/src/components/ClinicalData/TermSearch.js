import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Divider,
  Grid,
  TextField,
  Typography,
  Alert,
  Chip,
  Link,
  Paper,
  Slider,
  IconButton,
  Tooltip,
  InputAdornment
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import ScienceIcon from '@mui/icons-material/Science';
import InfoIcon from '@mui/icons-material/Info';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';

// Import the clinical data service
import clinicalDataService from '../../services/clinicalDataService';

/**
 * Term Search component
 * Allows searching for a medical term and finding related SNOMED CT concepts and clinical trials
 */
const TermSearch = () => {
  // State
  const [term, setTerm] = useState('');
  const [maxTrials, setMaxTrials] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);

  // Handle search
  const handleSearch = async () => {
    if (!term.trim()) {
      setError('Please enter a medical term');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await clinicalDataService.searchConceptAndTrials(term, maxTrials);
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
          Search for Medical Terms and Related Clinical Trials
        </Typography>
        <Tooltip title="This tool finds SNOMED CT concepts for medical terms and connects them with relevant clinical trials">
          <IconButton size="small">
            <InfoIcon fontSize="small" color="primary" />
          </IconButton>
        </Tooltip>
      </Box>
      <Typography variant="body2" paragraph>
        Enter a medical term to find related SNOMED CT concepts and clinical trials.
        This tool combines terminology standardization with clinical trials data.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper elevation={0} sx={{ p: 3, mb: 4, bgcolor: 'background.paper', borderRadius: 2 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              label="Medical Term"
              variant="outlined"
              fullWidth
              value={term}
              onChange={(e) => setTerm(e.target.value)}
              placeholder="e.g., diabetes, heart attack, asthma"
              helperText="Enter a medical condition, disease, or symptom"
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <MedicalServicesIcon color="action" />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={4}>
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
              Search
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Results section */}
      {results && (
        <Box>
          <Paper elevation={0} sx={{ p: 2, mb: 3, bgcolor: '#f8f9fa', borderRadius: 2, borderLeft: '4px solid #3498db' }}>
            <Typography variant="h6" gutterBottom>
              Results for: <strong>{results.term}</strong>
            </Typography>
            <Typography variant="body2">
              Found {results.concepts?.length || 0} SNOMED CT concepts and {results.trials?.length || 0} clinical trials
            </Typography>
          </Paper>

          {/* SNOMED CT Concepts */}
          <Card sx={{ mb: 3 }}>
            <CardContent sx={{ pb: 1 }}>
              <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <MedicalServicesIcon sx={{ mr: 1 }} color="primary" />
                SNOMED CT Concepts
              </Typography>
            </CardContent>
            <Divider />
            <CardContent>
              {results.concepts && results.concepts.length > 0 ? (
                <Grid container spacing={2}>
                  {results.concepts.map((concept, index) => (
                    <Grid item xs={12} md={6} key={concept.conceptId || index}>
                      <Card variant="outlined" sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle2" color="primary" gutterBottom>
                            {concept.preferredTerm}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                            <Chip
                              label={concept.conceptId}
                              size="small"
                              color="secondary"
                              variant="outlined"
                              icon={<LocalHospitalIcon />}
                            />
                          </Box>
                          {concept.definition && (
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              {concept.definition}
                            </Typography>
                          )}
                          {concept.synonyms && concept.synonyms.length > 0 && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="caption" color="text.secondary">
                                Synonyms:
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                                {concept.synonyms.slice(0, 5).map((synonym, i) => (
                                  <Chip key={i} label={synonym} size="small" />
                                ))}
                                {concept.synonyms.length > 5 && (
                                  <Chip label={`+${concept.synonyms.length - 5} more`} size="small" variant="outlined" />
                                )}
                              </Box>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Alert severity="info">
                  No SNOMED CT concepts found for this term.
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Clinical Trials */}
          <Card>
            <CardContent sx={{ pb: 1 }}>
              <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <ScienceIcon sx={{ mr: 1 }} color="primary" />
                Related Clinical Trials
              </Typography>
            </CardContent>
            <Divider />
            <CardContent>
              {results.trials && results.trials.length > 0 ? (
                <Grid container spacing={2}>
                  {results.trials.map((trial, index) => (
                    <Grid item xs={12} key={trial.NCTId || index}>
                      <Card variant="outlined">
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

export default TermSearch;
