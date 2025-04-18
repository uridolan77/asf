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
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Tooltip,
  IconButton,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import {
  Search as SearchIcon,
  Science as ScienceIcon,
  Info as InfoIcon,
  LocalHospital as LocalHospitalIcon
} from '@mui/icons-material';

import { useClinicalData, ConceptSearchParams, ClinicalTerm, ClinicalTrial } from '../../hooks/useClinicalData';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import { ButtonLoader } from '../UI/LoadingIndicators';

/**
 * ConceptExplorer component for finding trials by medical concept ID
 */
const ConceptExplorer: React.FC = () => {
  // State
  const [conceptId, setConceptId] = useState<string>('');
  const [terminology, setTerminology] = useState<string>('SNOMEDCT');
  const [maxTrials, setMaxTrials] = useState<number>(15);
  const [error, setError] = useState<string>('');
  const [results, setResults] = useState<any>(null);

  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Clinical data hooks
  const {
    getTrialsByConceptId
  } = useClinicalData();

  // Get the mutation function
  const {
    mutate: executeSearch,
    isPending: isLoading,
    isError: isSearchError,
    error: searchError
  } = getTrialsByConceptId();

  // Handle search
  const handleSearch = () => {
    if (!conceptId.trim()) {
      setError('Please enter a concept ID');
      return;
    }

    setError('');

    const params: ConceptSearchParams = {
      concept_id: conceptId.trim(),
      terminology,
      max_trials: maxTrials
    };

    executeSearch(params, {
      onSuccess: (data) => {
        setResults(data);
      }
    });
  };

  return (
    <Box>
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}

      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <ScienceIcon sx={{ mr: 1 }} />
          Concept Explorer
          <Tooltip title="Search for clinical trials by medical concept ID">
            <IconButton size="small" sx={{ ml: 1 }}>
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>

        <Divider sx={{ mb: 3 }} />

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {isSearchError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {searchError?.message || 'An error occurred during search'}
          </Alert>
        )}

        <Grid container spacing={2}>
          <Grid item xs={12} md={5}>
            <TextField
              fullWidth
              label="Concept ID"
              placeholder="e.g., 38341003"
              value={conceptId}
              onChange={(e) => setConceptId(e.target.value)}
              variant="outlined"
              required
              helperText="Enter a SNOMED CT or other terminology concept ID"
            />
          </Grid>

          <Grid item xs={12} md={3}>
            <FormControl fullWidth variant="outlined">
              <InputLabel>Terminology</InputLabel>
              <Select
                value={terminology}
                onChange={(e) => setTerminology(e.target.value)}
                label="Terminology"
              >
                <MenuItem value="SNOMEDCT">SNOMED CT</MenuItem>
                <MenuItem value="ICD10">ICD-10</MenuItem>
                <MenuItem value="LOINC">LOINC</MenuItem>
                <MenuItem value="RXNORM">RxNorm</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Max Trials"
              type="number"
              value={maxTrials}
              onChange={(e) => setMaxTrials(parseInt(e.target.value) || 15)}
              variant="outlined"
              InputProps={{
                inputProps: { min: 1, max: 50 }
              }}
            />
          </Grid>

          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              startIcon={isLoading ? <ButtonLoader size={20} /> : <SearchIcon />}
              onClick={handleSearch}
              disabled={isLoading || !conceptId.trim()}
            >
              {isLoading ? 'Searching...' : 'Find Trials by Concept'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {results && (
        <Grid container spacing={3}>
          {/* Concept Details */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardHeader
                title="Concept Details"
                subheader={results.concept.term}
                avatar={<ScienceIcon />}
              />
              <Divider />
              <CardContent>
                <Typography variant="body2" paragraph>
                  <strong>ID:</strong> {results.concept.id}
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>Code:</strong> {results.concept.code}
                </Typography>
                <Typography variant="body2" paragraph>
                  <strong>System:</strong> {results.concept.system}
                </Typography>
                {results.concept.definition && (
                  <Typography variant="body2" paragraph>
                    <strong>Definition:</strong> {results.concept.definition}
                  </Typography>
                )}
                {results.concept.synonyms && results.concept.synonyms.length > 0 && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      Synonyms:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                      {results.concept.synonyms.map((synonym: string, index: number) => (
                        <Chip key={index} label={synonym} size="small" />
                      ))}
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Clinical Trials */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardHeader
                title="Clinical Trials"
                subheader={`Found ${results.trials.length} trials for concept "${results.concept.term}"`}
                avatar={<LocalHospitalIcon />}
              />
              <Divider />
              <CardContent>
                {results.trials.length > 0 ? (
                  <List>
                    {results.trials.map((trial: ClinicalTrial) => (
                      <React.Fragment key={trial.id}>
                        <ListItem sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                          <ListItemText
                            primary={
                              <Link href={trial.url} target="_blank" rel="noopener noreferrer">
                                {trial.title}
                              </Link>
                            }
                            secondary={`Status: ${trial.status} | Phase: ${trial.phase || 'N/A'}`}
                          />
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                              Conditions:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                              {trial.conditions.map((condition, index) => (
                                <Chip key={index} label={condition} size="small" />
                              ))}
                            </Box>
                          </Box>
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                              Interventions:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                              {trial.interventions.map((intervention, index) => (
                                <Chip key={index} label={intervention} size="small" color="primary" variant="outlined" />
                              ))}
                            </Box>
                          </Box>
                          <Box sx={{ mt: 1, width: '100%', display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="body2">
                              Start: {new Date(trial.start_date).toLocaleDateString()}
                            </Typography>
                            {trial.completion_date && (
                              <Typography variant="body2">
                                Completion: {new Date(trial.completion_date).toLocaleDateString()}
                              </Typography>
                            )}
                            <Typography variant="body2">
                              Enrollment: {trial.enrollment}
                            </Typography>
                          </Box>
                        </ListItem>
                        <Divider component="li" />
                      </React.Fragment>
                    ))}
                  </List>
                ) : (
                  <Typography>No clinical trials found for this concept.</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default ConceptExplorer;
