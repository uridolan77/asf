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
  Autocomplete,
  Tooltip,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  LocalHospital as LocalHospitalIcon,
  MedicalInformation as MedicalIcon
} from '@mui/icons-material';

import { useClinicalData, TermSearchParams, ClinicalTerm, ClinicalTrial } from '../../hooks/useClinicalData';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import { ButtonLoader } from '../UI/LoadingIndicators';

/**
 * TermSearch component for searching medical terms and finding related concepts and trials
 */
const TermSearch: React.FC = () => {
  // State
  const [term, setTerm] = useState<string>('');
  const [maxTrials, setMaxTrials] = useState<number>(10);
  const [error, setError] = useState<string>('');
  const [results, setResults] = useState<any>(null);
  const [expandedConcept, setExpandedConcept] = useState<string | null>(null);

  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Clinical data hooks
  const {
    searchConceptAndTrials,
    getTermSuggestions
  } = useClinicalData();

  // Get the mutation function
  const {
    mutate: executeSearch,
    isPending: isLoading,
    isError: isSearchError,
    error: searchError
  } = searchConceptAndTrials();

  // Get term suggestions
  const {
    data: suggestionsData
  } = getTermSuggestions(term);

  // Extract suggestions
  const suggestions = suggestionsData?.suggestions || [];

  // Handle search
  const handleSearch = () => {
    if (!term.trim()) {
      setError('Please enter a medical term');
      return;
    }

    setError('');

    const params: TermSearchParams = {
      term: term.trim(),
      max_trials: maxTrials,
      include_hierarchy: true,
      include_mappings: true
    };

    executeSearch(params, {
      onSuccess: (data) => {
        setResults(data);
      }
    });
  };

  // Toggle expanded concept
  const toggleExpandedConcept = (conceptId: string) => {
    setExpandedConcept(expandedConcept === conceptId ? null : conceptId);
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
          <MedicalIcon sx={{ mr: 1 }} />
          Medical Term Search
          <Tooltip title="Search for medical terms to find related SNOMED CT concepts and clinical trials">
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
          <Grid item xs={12} md={8}>
            <Autocomplete
              freeSolo
              options={suggestions}
              value={term}
              onChange={(_, newValue) => setTerm(newValue || '')}
              onInputChange={(_, newValue) => setTerm(newValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  fullWidth
                  label="Medical Term"
                  placeholder="e.g., Hypertension, Diabetes, Asthma"
                  variant="outlined"
                  required
                />
              )}
            />
          </Grid>

          <Grid item xs={12} md={4}>
            <TextField
              fullWidth
              label="Max Trials"
              type="number"
              value={maxTrials}
              onChange={(e) => setMaxTrials(parseInt(e.target.value) || 10)}
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
              disabled={isLoading || !term.trim()}
            >
              {isLoading ? 'Searching...' : 'Search'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {results && (
        <Grid container spacing={3}>
          {/* SNOMED CT Concepts */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader
                title="SNOMED CT Concepts"
                subheader={`Found ${results.concepts.length} concepts for "${results.term}"`}
                avatar={<MedicalIcon />}
              />
              <Divider />
              <CardContent>
                {results.concepts.length > 0 ? (
                  <List>
                    {results.concepts.map((concept: ClinicalTerm) => (
                      <React.Fragment key={concept.id}>
                        <ListItem
                          button
                          onClick={() => toggleExpandedConcept(concept.id)}
                          sx={{ flexDirection: 'column', alignItems: 'flex-start' }}
                        >
                          <Box sx={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <ListItemText
                              primary={concept.term}
                              secondary={`Code: ${concept.code}`}
                            />
                            <Chip
                              label={concept.system}
                              size="small"
                              color="primary"
                            />
                          </Box>

                          <Accordion
                            expanded={expandedConcept === concept.id}
                            onChange={() => toggleExpandedConcept(concept.id)}
                            sx={{ width: '100%', mt: 1, boxShadow: 'none', '&:before': { display: 'none' } }}
                          >
                            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography variant="body2">
                                {expandedConcept === concept.id ? 'Hide Details' : 'Show Details'}
                              </Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              {concept.definition && (
                                <Typography variant="body2" paragraph>
                                  <strong>Definition:</strong> {concept.definition}
                                </Typography>
                              )}

                              {concept.synonyms && concept.synonyms.length > 0 && (
                                <Box sx={{ mb: 2 }}>
                                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                    Synonyms:
                                  </Typography>
                                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                                    {concept.synonyms.map((synonym, index) => (
                                      <Chip key={index} label={synonym} size="small" />
                                    ))}
                                  </Box>
                                </Box>
                              )}

                              {concept.parent_concepts && concept.parent_concepts.length > 0 && (
                                <Box sx={{ mb: 2 }}>
                                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                    Parent Concepts:
                                  </Typography>
                                  <Box component="ul" sx={{ pl: 2, mt: 0.5 }}>
                                    {concept.parent_concepts.map((parent, index) => (
                                      <Box component="li" key={index}>
                                        <Typography variant="body2">{parent}</Typography>
                                      </Box>
                                    ))}
                                  </Box>
                                </Box>
                              )}

                              {concept.child_concepts && concept.child_concepts.length > 0 && (
                                <Box>
                                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                    Child Concepts:
                                  </Typography>
                                  <Box component="ul" sx={{ pl: 2, mt: 0.5 }}>
                                    {concept.child_concepts.map((child, index) => (
                                      <Box component="li" key={index}>
                                        <Typography variant="body2">{child}</Typography>
                                      </Box>
                                    ))}
                                  </Box>
                                </Box>
                              )}
                            </AccordionDetails>
                          </Accordion>
                        </ListItem>
                        <Divider component="li" />
                      </React.Fragment>
                    ))}
                  </List>
                ) : (
                  <Typography>No SNOMED CT concepts found for this term.</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Clinical Trials */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardHeader
                title="Clinical Trials"
                subheader={`Found ${results.trials.length} trials for "${results.term}"`}
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
                  <Typography>No clinical trials found for this term.</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default TermSearch;
