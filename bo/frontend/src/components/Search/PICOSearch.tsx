import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid, Chip,
  IconButton, Autocomplete, Divider, CircularProgress,
  Card, CardContent, CardHeader, CardActions, Accordion,
  AccordionSummary, AccordionDetails, Tooltip, Avatar,
  Alert
} from '@mui/material';
import {
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  Clear as ClearIcon,
  Download as DownloadIcon,
  Save as SaveIcon,
  Science as ScienceIcon,
  BarChart as BarChartIcon,
  BiotechOutlined as BiotechIcon,
  PersonOutline as PersonIcon,
  MedicalInformation as MedicalIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext';
import { usePICOSearch, PICOSearchParams, SearchResult, SearchHistoryItem } from '../../hooks/usePICOSearch';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import { ButtonLoader } from '../UI/LoadingIndicators';
import { HoverAnimation, StaggeredList } from '../UI/Animations';

interface PICOSearchProps {
  onSearchResults?: (results: any) => void;
}

/**
 * Enhanced PICO search component with improved UX
 * (Population, Intervention, Comparison, Outcome)
 */
const PICOSearch: React.FC<PICOSearchProps> = ({ onSearchResults }) => {
  const { showSuccess, showError, showInfo } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Form state
  const [condition, setCondition] = useState<string>('');
  const [population, setPopulation] = useState<string>('');
  const [interventions, setInterventions] = useState<string[]>(['']);
  const [outcomes, setOutcomes] = useState<string[]>(['']);
  const [studyDesign, setStudyDesign] = useState<string>('');
  const [years, setYears] = useState<number>(5);
  const [maxResults, setMaxResults] = useState<number>(50);

  // UI state
  const [searchResults, setSearchResults] = useState<any>(null);
  const [expandedResult, setExpandedResult] = useState<number | null>(null);
  const [error, setError] = useState<string>('');

  // PICO search hooks
  const {
    picoSearch,
    getConditionSuggestions,
    getInterventionSuggestions,
    getOutcomeSuggestions,
    searchHistory
  } = usePICOSearch();

  // Get the mutation function
  const {
    mutate: executePicoSearch,
    isPending: isSearching,
    isError: isSearchError,
    error: searchError
  } = picoSearch();

  // Get suggestions
  const {
    data: conditionSuggestionsData
  } = getConditionSuggestions(condition);

  const {
    data: interventionSuggestionsData
  } = getInterventionSuggestions(interventions[0]);

  const {
    data: outcomeSuggestionsData
  } = getOutcomeSuggestions(outcomes[0]);

  // Extract suggestions
  const conditionSuggestions = conditionSuggestionsData?.suggestions || [];
  const interventionSuggestions = interventionSuggestionsData?.suggestions || [];
  const outcomeSuggestions = outcomeSuggestionsData?.suggestions || [];

  // Handle intervention field changes
  const handleInterventionChange = (index: number, value: string) => {
    const newInterventions = [...interventions];
    newInterventions[index] = value;
    setInterventions(newInterventions);
  };

  // Add intervention field
  const addInterventionField = () => {
    setInterventions([...interventions, '']);
  };

  // Remove intervention field
  const removeInterventionField = (index: number) => {
    if (interventions.length > 1) {
      const newInterventions = [...interventions];
      newInterventions.splice(index, 1);
      setInterventions(newInterventions);
    }
  };

  // Handle outcome field changes
  const handleOutcomeChange = (index: number, value: string) => {
    const newOutcomes = [...outcomes];
    newOutcomes[index] = value;
    setOutcomes(newOutcomes);
  };

  // Add outcome field
  const addOutcomeField = () => {
    setOutcomes([...outcomes, '']);
  };

  // Remove outcome field
  const removeOutcomeField = (index: number) => {
    if (outcomes.length > 1) {
      const newOutcomes = [...outcomes];
      newOutcomes.splice(index, 1);
      setOutcomes(newOutcomes);
    }
  };

  // Handle search submission
  const handleSearch = () => {
    if (!condition.trim()) {
      showError('Please enter a medical condition');
      setError('Please enter a medical condition');
      return;
    }

    setError('');

    // Filter out empty values
    const filteredInterventions = interventions.filter(item => item.trim());
    const filteredOutcomes = outcomes.filter(item => item.trim());

    // Create search parameters
    const searchParams: PICOSearchParams = {
      condition,
      interventions: filteredInterventions,
      outcomes: filteredOutcomes,
      max_results: maxResults,
      page: 1,
      page_size: 20
    };

    if (population) {
      searchParams.population = population;
    }

    if (studyDesign) {
      searchParams.study_design = studyDesign;
    }

    if (years > 0) {
      searchParams.years = years;
    }

    // Execute the search
    executePicoSearch(searchParams, {
      onSuccess: (data) => {
        setSearchResults(data);

        // If callback provided, send results up
        if (onSearchResults) {
          onSearchResults(data);
        }

        showSuccess(`Found ${data.total_count} results for your PICO search`);
      }
    });
  };

  // Reset the form
  const handleReset = () => {
    setCondition('');
    setPopulation('');
    setInterventions(['']);
    setOutcomes(['']);
    setStudyDesign('');
    setYears(5);
    setMaxResults(50);
    setSearchResults(null);
    setError('');
  };

  // Load a search from history
  const loadSearchFromHistory = (search: SearchHistoryItem) => {
    setCondition(search.condition);
    setPopulation(search.population || '');
    setInterventions(search.interventions.length > 0 ? search.interventions : ['']);
    setOutcomes(search.outcomes.length > 0 ? search.outcomes : ['']);
    setStudyDesign(search.studyDesign || '');
    setYears(search.years || 5);

    // Execute the search automatically
    setTimeout(() => {
      handleSearch();
    }, 100);
  };

  // Toggle expanded result
  const toggleExpandedResult = (index: number) => {
    setExpandedResult(expandedResult === index ? null : index);
  };

  // Get relevance color
  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'primary';
    if (score >= 0.4) return 'warning';
    return 'default';
  };

  return (
    <Box sx={{ width: '100%' }}>
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}

      <Paper
        elevation={3}
        sx={{
          p: 3,
          mb: 3,
          borderRadius: 2,
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}
      >
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <MedicalIcon sx={{ mr: 1 }} />
          PICO Search
          <Tooltip title="Population, Intervention, Comparison, Outcome - A framework for formulating clinical questions">
            <IconButton size="small" sx={{ ml: 1 }}>
              <ScienceIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>

        <Divider sx={{ mb: 3 }} />

        {error && (
          <Box sx={{ mb: 2 }}>
            <Typography color="error">{error}</Typography>
          </Box>
        )}

        {isSearchError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {searchError?.message || 'An error occurred during search'}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Condition */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>
              Medical Condition
              <Chip size="small" label="Required" color="primary" sx={{ ml: 1 }} />
            </Typography>
            <Autocomplete
              freeSolo
              options={conditionSuggestions}
              value={condition}
              onChange={(_, newValue) => setCondition(newValue || '')}
              onInputChange={(_, newValue) => setCondition(newValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  fullWidth
                  variant="outlined"
                  placeholder="e.g., Hypertension, Diabetes, Asthma"
                  required
                />
              )}
            />
          </Grid>

          {/* Population */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" gutterBottom>
              Population (Optional)
            </Typography>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="e.g., Adults over 65, Pregnant women, Children"
              value={population}
              onChange={(e) => setPopulation(e.target.value)}
              InputProps={{
                startAdornment: (
                  <PersonIcon color="action" sx={{ mr: 1 }} />
                ),
              }}
            />
          </Grid>

          {/* Interventions */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle2">
                Interventions
                <Chip size="small" label="At least one" color="primary" sx={{ ml: 1 }} />
              </Typography>
              <Button
                startIcon={<AddIcon />}
                size="small"
                onClick={addInterventionField}
              >
                Add Intervention
              </Button>
            </Box>

            {interventions.map((intervention, index) => (
              <Box key={index} sx={{ display: 'flex', mb: 2 }}>
                <Autocomplete
                  freeSolo
                  options={interventionSuggestions}
                  value={intervention}
                  onChange={(_, newValue) => handleInterventionChange(index, newValue || '')}
                  onInputChange={(_, newValue) => handleInterventionChange(index, newValue)}
                  sx={{ flex: 1 }}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      fullWidth
                      variant="outlined"
                      placeholder={`e.g., ${index === 0 ? 'Medication, Surgery, Therapy' : 'Additional intervention'}`}
                    />
                  )}
                />
                {interventions.length > 1 && (
                  <IconButton
                    color="error"
                    onClick={() => removeInterventionField(index)}
                    sx={{ ml: 1 }}
                  >
                    <RemoveIcon />
                  </IconButton>
                )}
              </Box>
            ))}
          </Grid>

          {/* Outcomes */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="subtitle2">
                Outcomes
                <Chip size="small" label="At least one" color="primary" sx={{ ml: 1 }} />
              </Typography>
              <Button
                startIcon={<AddIcon />}
                size="small"
                onClick={addOutcomeField}
              >
                Add Outcome
              </Button>
            </Box>

            {outcomes.map((outcome, index) => (
              <Box key={index} sx={{ display: 'flex', mb: 2 }}>
                <Autocomplete
                  freeSolo
                  options={outcomeSuggestions}
                  value={outcome}
                  onChange={(_, newValue) => handleOutcomeChange(index, newValue || '')}
                  onInputChange={(_, newValue) => handleOutcomeChange(index, newValue)}
                  sx={{ flex: 1 }}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      fullWidth
                      variant="outlined"
                      placeholder={`e.g., ${index === 0 ? 'Mortality, Recovery rate, Side effects' : 'Additional outcome'}`}
                    />
                  )}
                />
                {outcomes.length > 1 && (
                  <IconButton
                    color="error"
                    onClick={() => removeOutcomeField(index)}
                    sx={{ ml: 1 }}
                  >
                    <RemoveIcon />
                  </IconButton>
                )}
              </Box>
            ))}
          </Grid>

          {/* Study Design */}
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" gutterBottom>
              Study Design (Optional)
            </Typography>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="e.g., RCT, Cohort study, Meta-analysis"
              value={studyDesign}
              onChange={(e) => setStudyDesign(e.target.value)}
            />
          </Grid>

          {/* Years */}
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" gutterBottom>
              Publication Years
            </Typography>
            <TextField
              fullWidth
              variant="outlined"
              type="number"
              placeholder="Number of years to search"
              value={years}
              onChange={(e) => setYears(parseInt(e.target.value) || 5)}
              InputProps={{
                inputProps: { min: 1, max: 50 }
              }}
            />
          </Grid>

          {/* Max Results */}
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" gutterBottom>
              Maximum Results
            </Typography>
            <TextField
              fullWidth
              variant="outlined"
              type="number"
              placeholder="Maximum number of results"
              value={maxResults}
              onChange={(e) => setMaxResults(parseInt(e.target.value) || 50)}
              InputProps={{
                inputProps: { min: 10, max: 100 }
              }}
            />
          </Grid>

          {/* Action Buttons */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                color="primary"
                startIcon={isSearching ? <ButtonLoader size={20} /> : <SearchIcon />}
                onClick={handleSearch}
                disabled={isSearching || !condition.trim()}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </Button>

              <Button
                variant="outlined"
                startIcon={<ClearIcon />}
                onClick={handleReset}
                disabled={isSearching}
              >
                Reset
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Search History */}
      {searchHistory.length > 0 && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Recent Searches
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {searchHistory.slice(0, 5).map((search) => (
              <Chip
                key={search.id}
                label={`${search.condition} (${search.resultCount} results)`}
                onClick={() => loadSearchFromHistory(search)}
                onDelete={() => {/* Implement delete from history */}}
                color="primary"
                variant="outlined"
              />
            ))}
          </Box>
        </Paper>
      )}

      {/* Search Results */}
      {searchResults && (
        <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <BarChartIcon sx={{ mr: 1 }} />
            Search Results
            <Chip
              label={`${searchResults.total_count} results`}
              color="primary"
              size="small"
              sx={{ ml: 2 }}
            />
          </Typography>

          <Divider sx={{ mb: 3 }} />

          <Grid container spacing={3}>
            {searchResults.articles?.length > 0 ? (
              <StaggeredList staggerDelay={100}>
                {searchResults.articles.map((article: SearchResult, index: number) => (
                  <Grid item xs={12} key={article.id || index}>
                    <HoverAnimation>
                      <Card
                        variant="outlined"
                        sx={{
                          transition: 'box-shadow 0.3s ease-in-out'
                        }}
                      >
                      <CardHeader
                        title={article.title}
                        subheader={`${article.journal} (${article.year})`}
                        action={
                          <Chip
                            label={`Relevance: ${article.relevance_score ? Math.round(article.relevance_score * 100) : 'N/A'}%`}
                            color={getRelevanceColor(article.relevance_score)}
                            size="small"
                          />
                        }
                      />
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          {article.authors.join(', ')}
                        </Typography>

                        <Accordion
                          expanded={expandedResult === index}
                          onChange={() => toggleExpandedResult(index)}
                          sx={{ mt: 2, boxShadow: 'none', '&:before': { display: 'none' } }}
                        >
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Typography>
                              {expandedResult === index ? 'Hide Abstract' : 'Show Abstract'}
                            </Typography>
                          </AccordionSummary>
                          <AccordionDetails>
                            <Typography variant="body2">
                              {article.abstract}
                            </Typography>
                          </AccordionDetails>
                        </Accordion>
                      </CardContent>

                      <CardActions>
                        <Button
                          size="small"
                          onClick={() => toggleExpandedResult(index)}
                        >
                          {expandedResult === index ? 'Show Less' : 'Read More'}
                        </Button>

                        {article.pmid && (
                          <Button
                            size="small"
                            component="a"
                            href={`https://pubmed.ncbi.nlm.nih.gov/${article.pmid}`}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            View on PubMed
                          </Button>
                        )}

                        <Button size="small" color="secondary">
                          Save to Knowledge Base
                        </Button>
                      </CardActions>
                      </Card>
                    </HoverAnimation>
                  </Grid>
                ))}
              </StaggeredList>
            ) : (
              <Typography>No results found matching your criteria.</Typography>
            )}
          </Grid>
        </Paper>
      )}
    </Box>
  );
};

export default PICOSearch;
