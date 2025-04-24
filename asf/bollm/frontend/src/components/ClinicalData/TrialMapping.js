import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Divider,
  Grid,
  TextField,
  Typography,
  Alert,
  Chip,
  Link,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tooltip,
  IconButton,
  InputAdornment
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import LinkIcon from '@mui/icons-material/Link';
import MedicalServicesIcon from '@mui/icons-material/MedicalServices';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import InfoIcon from '@mui/icons-material/Info';
import CodeIcon from '@mui/icons-material/Code';

// Import the clinical data service
import clinicalDataService from '../../services/clinicalDataService';

/**
 * Trial Mapping component
 * Maps conditions in a clinical trial to SNOMED CT concepts
 * and provides semantic context for the trial
 */
const TrialMapping = () => {
  // State
  const [nctId, setNctId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [mappingResults, setMappingResults] = useState(null);
  const [semanticContext, setSemanticContext] = useState(null);

  // Handle mapping of trial conditions
  const handleMapConditions = async () => {
    if (!nctId.trim()) {
      setError('Please enter an NCT ID');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await clinicalDataService.mapTrialConditions(nctId);
      setMappingResults(response.data);

      // Also get semantic context
      const contextResponse = await clinicalDataService.getTrialSemanticContext(nctId);
      setSemanticContext(contextResponse.data);
    } catch (err) {
      setError(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Format NCT ID for display and links
  const formatNctId = (id) => {
    if (!id) return '';
    return id.startsWith('NCT') ? id : `NCT${id}`;
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ mr: 1 }}>
          Map Clinical Trial Conditions to SNOMED CT
        </Typography>
        <Tooltip title="This tool maps conditions in clinical trials to standardized SNOMED CT concepts and provides semantic context">
          <IconButton size="small">
            <InfoIcon fontSize="small" color="primary" />
          </IconButton>
        </Tooltip>
      </Box>
      <Typography variant="body2" paragraph>
        Enter a ClinicalTrials.gov identifier (NCT number) to map its conditions to standardized
        SNOMED CT concepts and get semantic context for the trial.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper elevation={0} sx={{ p: 3, mb: 4, bgcolor: 'background.paper', borderRadius: 2 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <TextField
              label="ClinicalTrials.gov Identifier (NCT Number)"
              variant="outlined"
              fullWidth
              value={nctId}
              onChange={(e) => setNctId(e.target.value)}
              placeholder="e.g., NCT01234567 or 01234567"
              helperText="Enter an NCT ID with or without the 'NCT' prefix"
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <CodeIcon color="action" />
                  </InputAdornment>
                ),
              }}
            />
          </Grid>
          <Grid item xs={12} md={4} sx={{ display: 'flex', alignItems: 'center' }}>
            <Button
              variant="contained"
              color="primary"
              fullWidth
              onClick={handleMapConditions}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
              sx={{ height: '56px' }}
            >
              Map Trial Conditions
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Results section */}
      {(mappingResults || semanticContext) && (
        <Box>
          {/* Trial Information */}
          {semanticContext && (
            <Card sx={{ mb: 3 }}>
              <CardHeader
                title={semanticContext.study_title || `Trial ${formatNctId(semanticContext.nct_id)}`}
                subheader={formatNctId(semanticContext.nct_id)}
                action={
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<LinkIcon />}
                    component={Link}
                    href={`https://clinicaltrials.gov/study/${formatNctId(semanticContext.nct_id)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    View on ClinicalTrials.gov
                  </Button>
                }
              />
              <Divider />
              <CardContent>
                {semanticContext.brief_summary && (
                  <Typography variant="body2" paragraph>
                    {semanticContext.brief_summary}
                  </Typography>
                )}

                <Paper elevation={0} sx={{ p: 2, bgcolor: '#f8f9fa', borderRadius: 2 }}>
                  <Grid container spacing={2}>
                    {semanticContext.study_type && (
                      <Grid item xs={6} md={3}>
                        <Typography variant="body2">
                          <strong>Study Type:</strong> {semanticContext.study_type}
                        </Typography>
                      </Grid>
                    )}
                    {semanticContext.phase && (
                      <Grid item xs={6} md={3}>
                        <Typography variant="body2">
                          <strong>Phase:</strong> {semanticContext.phase}
                        </Typography>
                      </Grid>
                    )}
                    {semanticContext.status && (
                      <Grid item xs={6} md={3}>
                        <Typography variant="body2">
                          <strong>Status:</strong> {semanticContext.status}
                        </Typography>
                      </Grid>
                    )}
                    {semanticContext.enrollment && (
                      <Grid item xs={6} md={3}>
                        <Typography variant="body2">
                          <strong>Enrollment:</strong> {semanticContext.enrollment}
                        </Typography>
                      </Grid>
                    )}
                  </Grid>
                </Paper>
              </CardContent>
            </Card>
          )}

          {/* Condition Mappings */}
          {mappingResults && mappingResults.conditions && (
            <Card sx={{ mb: 3 }}>
              <CardHeader
                title="Condition Mappings"
                avatar={<LocalHospitalIcon color="primary" />}
                subheader="Trial conditions mapped to SNOMED CT concepts"
              />
              <Divider />
              <CardContent>
                <Grid container spacing={3}>
                  {Object.entries(mappingResults.conditions).map(([condition, concepts]) => (
                    <Grid item xs={12} md={6} key={condition}>
                      <Card variant="outlined" sx={{ height: '100%' }}>
                        <CardContent>
                          <Typography variant="subtitle2" color="primary" gutterBottom>
                            {condition}
                          </Typography>

                          {concepts && concepts.length > 0 ? (
                            <List dense>
                              {concepts.map((concept, index) => (
                                <ListItem key={concept.conceptId || index}>
                                  <ListItemIcon sx={{ minWidth: 36 }}>
                                    <MedicalServicesIcon fontSize="small" color="secondary" />
                                  </ListItemIcon>
                                  <ListItemText
                                    primary={concept.preferredTerm}
                                    secondary={
                                      <Chip
                                        label={concept.conceptId}
                                        size="small"
                                        variant="outlined"
                                        color="secondary"
                                      />
                                    }
                                  />
                                </ListItem>
                              ))}
                            </List>
                          ) : (
                            <Alert severity="info" sx={{ mt: 1 }}>
                              No SNOMED CT mappings found.
                            </Alert>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          )}

          {/* Intervention Mappings */}
          {semanticContext && semanticContext.intervention_mappings && (
            <Card>
              <CardHeader
                title="Intervention Mappings"
                avatar={<MedicalServicesIcon color="primary" />}
                subheader="Trial interventions mapped to SNOMED CT concepts"
              />
              <Divider />
              <CardContent>
                <Grid container spacing={3}>
                  {Object.entries(semanticContext.intervention_mappings).map(([intervention, concepts]) => (
                    <Grid item xs={12} md={6} key={intervention}>
                      <Card variant="outlined" sx={{ height: '100%', bgcolor: '#f8f9fa' }}>
                        <CardContent>
                          <Typography variant="subtitle2" color="primary" gutterBottom>
                            {intervention}
                          </Typography>

                          {concepts && concepts.length > 0 ? (
                            <List dense>
                              {concepts.map((concept, index) => (
                                <ListItem key={concept.conceptId || index}>
                                  <ListItemIcon sx={{ minWidth: 36 }}>
                                    <MedicalServicesIcon fontSize="small" color="secondary" />
                                  </ListItemIcon>
                                  <ListItemText
                                    primary={concept.preferredTerm}
                                    secondary={
                                      <Chip
                                        label={concept.conceptId}
                                        size="small"
                                        variant="outlined"
                                        color="secondary"
                                      />
                                    }
                                  />
                                </ListItem>
                              ))}
                            </List>
                          ) : (
                            <Alert severity="info" sx={{ mt: 1 }}>
                              No SNOMED CT mappings found.
                            </Alert>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          )}
        </Box>
      )}
    </Box>
  );
};

export default TrialMapping;
