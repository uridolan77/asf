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
  Autocomplete,
  Tooltip,
  IconButton,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import {
  Search as SearchIcon,
  Science as ScienceIcon,
  Psychology as PsychologyIcon,
  Info as InfoIcon,
  Biotech as BiotechIcon
} from '@mui/icons-material';

import { useClinicalData, SemanticSearchParams, ClinicalTrial } from '../../hooks/useClinicalData';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import { ButtonLoader } from '../UI/LoadingIndicators';

/**
 * SemanticSearch component for searching medical terms with semantic expansion
 */
const SemanticSearch: React.FC = () => {
  // State
  const [term, setTerm] = useState<string>('');
  const [includeSimilar, setIncludeSimilar] = useState<boolean>(true);
  const [maxTrials, setMaxTrials] = useState<number>(20);
  const [error, setError] = useState<string>('');
  const [results, setResults] = useState<any>(null);

  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Clinical data hooks
  const {
    findTrialsWithSemanticExpansion,
    getTermSuggestions
  } = useClinicalData();

  // Get the mutation function
  const {
    mutate: executeSearch,
    isPending: isLoading,
    isError: isSearchError,
    error: searchError
  } = findTrialsWithSemanticExpansion();

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

    const params: SemanticSearchParams = {
      term: term.trim(),
      include_similar: includeSimilar,
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
          <PsychologyIcon sx={{ mr: 1 }} />
          Semantic Search
          <Tooltip title="Search for clinical trials with semantic expansion of medical terms">
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
          <Grid item xs={12} md={6}>
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
                  placeholder="e.g., Heart Attack, Cancer, Pneumonia"
                  variant="outlined"
                  required
                />
              )}
            />
          </Grid>

          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <TextField
                fullWidth
                label="Max Trials"
                type="number"
                value={maxTrials}
                onChange={(e) => setMaxTrials(parseInt(e.target.value) || 20)}
                variant="outlined"
                InputProps={{
                  inputProps: { min: 1, max: 100 }
                }}
                sx={{ mr: 2 }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={includeSimilar}
                    onChange={(e) => setIncludeSimilar(e.target.checked)}
                    color="primary"
                  />
                }
                label="Include Similar Terms"
              />
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              startIcon={isLoading ? <ButtonLoader size={20} /> : <SearchIcon />}
              onClick={handleSearch}
              disabled={isLoading || !term.trim()}
            >
              {isLoading ? 'Searching...' : 'Semantic Search'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {results && (
        <Grid container spacing={3}>
          {/* Expanded Terms */}
          <Grid item xs={12}>
            <Card>
              <CardHeader
                title="Semantic Expansion"
                subheader={`Expanded "${results.term}" to ${results.expanded_terms.length} related terms`}
                avatar={<BiotechIcon />}
              />
              <Divider />
              <CardContent>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {results.expanded_terms.map((expandedTerm: string, index: number) => (
                    <Chip
                      key={index}
                      label={expandedTerm}
                      color={expandedTerm === results.term ? 'primary' : 'default'}
                      variant={expandedTerm === results.term ? 'filled' : 'outlined'}
                    />
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Clinical Trials */}
          <Grid item xs={12}>
            <Card>
              <CardHeader
                title="Clinical Trials"
                subheader={`Found ${results.trials.length} trials using semantic expansion`}
                avatar={<ScienceIcon />}
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

export default SemanticSearch;
