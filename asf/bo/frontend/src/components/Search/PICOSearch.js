import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid, Chip,
  IconButton, Autocomplete, Divider, CircularProgress,
  Card, CardContent, CardHeader, CardActions, Accordion,
  AccordionSummary, AccordionDetails, Tooltip, Avatar
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
import apiService from '../../services/api';
import { ButtonLoader } from '../UI/LoadingIndicators.js';
import { HoverAnimation, StaggeredList } from '../UI/Animations.js';

/**
 * Enhanced PICO search component with improved UX
 * (Population, Intervention, Comparison, Outcome)
 */
const PICOSearch = ({ onSearchResults }) => {
  const { showSuccess, showError, showInfo } = useNotification();

  // Form state
  const [condition, setCondition] = useState('');
  const [population, setPopulation] = useState('');
  const [interventions, setInterventions] = useState(['']);
  const [outcomes, setOutcomes] = useState(['']);
  const [studyDesign, setStudyDesign] = useState('');
  const [years, setYears] = useState(5);
  const [maxResults, setMaxResults] = useState(50);

  // Suggestions state
  const [conditionSuggestions, setConditionSuggestions] = useState([]);
  const [interventionSuggestions, setInterventionSuggestions] = useState([]);
  const [outcomeSuggestions, setOutcomeSuggestions] = useState([]);

  // UI state
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);
  const [expandedResult, setExpandedResult] = useState(null);
  const [error, setError] = useState('');

  // Load search history from localStorage
  useEffect(() => {
    const savedHistory = localStorage.getItem('picoSearchHistory');
    if (savedHistory) {
      setSearchHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Study design options
  const studyDesignOptions = [
    { value: 'Randomized Controlled Trial', label: 'Randomized Controlled Trial (RCT)' },
    { value: 'Meta-Analysis', label: 'Meta-Analysis' },
    { value: 'Systematic Review', label: 'Systematic Review' },
    { value: 'Cohort Study', label: 'Cohort Study' },
    { value: 'Case-Control Study', label: 'Case-Control Study' },
    { value: 'Case Series', label: 'Case Series' },
    { value: 'Case Report', label: 'Case Report' },
  ];

  // Year options
  const yearOptions = [
    { value: 1, label: 'Last 1 Year' },
    { value: 2, label: 'Last 2 Years' },
    { value: 3, label: 'Last 3 Years' },
    { value: 5, label: 'Last 5 Years' },
    { value: 10, label: 'Last 10 Years' },
    { value: 20, label: 'Last 20 Years' },
    { value: 0, label: 'Any Year' },
  ];

  // Fetch terminology suggestions from API
  const fetchSuggestions = async (term, type) => {
    if (!term || term.length < 2) return [];

    try {
      const result = await apiService.medical.terminologySearch(term);
      if (result.success) {
        // Filter suggestions based on type
        let filtered = result.data.data.results;
        if (type === 'condition') {
          // Filter for disease/condition concepts
          filtered = filtered.filter(item =>
            item.semanticTypes?.includes('Disease or Syndrome') ||
            item.semanticTypes?.includes('Clinical Finding')
          );
        } else if (type === 'intervention') {
          // Filter for treatment/intervention concepts
          filtered = filtered.filter(item =>
            item.semanticTypes?.includes('Therapeutic or Preventive Procedure') ||
            item.semanticTypes?.includes('Pharmacologic Substance') ||
            item.semanticTypes?.includes('Medical Device')
          );
        } else if (type === 'outcome') {
          // Keep a broader range of concepts for outcomes
          filtered = filtered.filter(item =>
            item.semanticTypes?.includes('Clinical Attribute') ||
            item.semanticTypes?.includes('Finding') ||
            item.semanticTypes?.includes('Laboratory Procedure')
          );
        }

        return filtered.map(result => result.preferredTerm);
      }
      return [];
    } catch (error) {
      console.error(`Error fetching ${type} suggestions:`, error);
      return [];
    }
  };

  // Handle condition input change with suggestions
  const handleConditionChange = async (event, value) => {
    setCondition(value);
    if (value?.length >= 2) {
      const suggestions = await fetchSuggestions(value, 'condition');
      setConditionSuggestions(suggestions);
    }
  };

  // Handle intervention input change with suggestions
  const handleInterventionChange = async (index, value) => {
    const newInterventions = [...interventions];
    newInterventions[index] = value;
    setInterventions(newInterventions);

    if (value?.length >= 2) {
      const suggestions = await fetchSuggestions(value, 'intervention');
      setInterventionSuggestions(suggestions);
    }
  };

  // Handle outcome input change with suggestions
  const handleOutcomeChange = async (index, value) => {
    const newOutcomes = [...outcomes];
    newOutcomes[index] = value;
    setOutcomes(newOutcomes);

    if (value?.length >= 2) {
      const suggestions = await fetchSuggestions(value, 'outcome');
      setOutcomeSuggestions(suggestions);
    }
  };

  // Add a new intervention field
  const addIntervention = () => {
    setInterventions([...interventions, '']);
  };

  // Remove an intervention field
  const removeIntervention = (index) => {
    if (interventions.length <= 1) return;
    const newInterventions = [...interventions];
    newInterventions.splice(index, 1);
    setInterventions(newInterventions);
  };

  // Add a new outcome field
  const addOutcome = () => {
    setOutcomes([...outcomes, '']);
  };

  // Remove an outcome field
  const removeOutcome = (index) => {
    if (outcomes.length <= 1) return;
    const newOutcomes = [...outcomes];
    newOutcomes.splice(index, 1);
    setOutcomes(newOutcomes);
  };

  // Handle search submission
  const handleSearch = async () => {
    if (!condition.trim()) {
      showError('Please enter a medical condition');
      setError('Please enter a medical condition');
      return;
    }

    setIsSearching(true);
    setError('');

    try {
      // Filter out empty values
      const filteredInterventions = interventions.filter(item => item.trim());
      const filteredOutcomes = outcomes.filter(item => item.trim());

      // Create search parameters
      const searchParams = {
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
      const result = await apiService.medical.picoSearch(searchParams);

      if (result.success) {
        const results = result.data.data;
        setSearchResults(results);

        // Update search history
        const newSearch = {
          id: Date.now(),
          timestamp: new Date().toISOString(),
          condition,
          interventions: filteredInterventions,
          outcomes: filteredOutcomes,
          population,
          studyDesign,
          years,
          resultCount: results.total_count || 0
        };

        const updatedHistory = [newSearch, ...searchHistory.slice(0, 9)];
        setSearchHistory(updatedHistory);
        localStorage.setItem('picoSearchHistory', JSON.stringify(updatedHistory));

        // If callback provided, send results up
        if (onSearchResults) {
          onSearchResults(results);
        }

        showSuccess(`Found ${results.total_count} results for your PICO search`);
      } else {
        setError(`Search failed: ${result.error}`);
        showError(`Search failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Error executing PICO search:', error);
      setError(`Search error: ${error.message}`);
      showError(`Search error: ${error.message}`);
    } finally {
      setIsSearching(false);
    }
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
  const loadSearchFromHistory = (search) => {
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
  const toggleExpandedResult = (index) => {
    setExpandedResult(expandedResult === index ? null : index);
  };

  // Format the PICO query for display
  const formatPICOQuery = () => {
    const parts = [];

    if (population) {
      parts.push(`Population: ${population}`);
    }

    parts.push(`Condition: ${condition}`);

    if (interventions.some(i => i.trim())) {
      parts.push(`Interventions: ${interventions.filter(i => i.trim()).join(', ')}`);
    }

    if (outcomes.some(o => o.trim())) {
      parts.push(`Outcomes: ${outcomes.filter(o => o.trim()).join(', ')}`);
    }

    if (studyDesign) {
      parts.push(`Study Design: ${studyDesign}`);
    }

    if (years > 0) {
      parts.push(`Timeframe: Last ${years} years`);
    }

    return parts.join(' | ');
  };

  // Helper function to get color based on relevance score
  const getRelevanceColor = (score) => {
    if (score >= 0.9) return 'success'; // High relevance - green
    if (score >= 0.7) return 'primary'; // Medium relevance - blue
    return 'default'; // Low relevance - default
  };

  return (
    <Box sx={{ width: '100%' }}>
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

        <Grid container spacing={3}>
          {/* Condition field (required) */}
          <Grid item xs={12} md={8}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              Condition / Problem
              <Chip size="small" label="Required" color="primary" sx={{ ml: 1 }} />
            </Typography>
            <Autocomplete
              freeSolo
              options={conditionSuggestions}
              inputValue={condition}
              onInputChange={handleConditionChange}
              renderInput={(params) => (
                <TextField
                  {...params}
                  variant="outlined"
                  fullWidth
                  placeholder="e.g., Community Acquired Pneumonia"
                  required
                />
              )}
            />
          </Grid>

          {/* Population field (optional) */}
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <PersonIcon fontSize="small" sx={{ mr: 0.5 }} />
              Population (P)
              <Chip size="small" label="Optional" color="default" sx={{ ml: 1 }} variant="outlined" />
            </Typography>
            <TextField
              variant="outlined"
              fullWidth
              placeholder="e.g., Adults over 65"
              value={population}
              onChange={(e) => setPopulation(e.target.value)}
            />
          </Grid>

          {/* Interventions (I) */}
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <BiotechIcon fontSize="small" sx={{ mr: 0.5 }} />
              Interventions (I)
              <Chip size="small" label="Recommended" color="secondary" sx={{ ml: 1 }} />
            </Typography>

            {interventions.map((intervention, index) => (
              <Box key={`intervention-${index}`} sx={{ display: 'flex', mb: 2, gap: 1 }}>
                <Autocomplete
                  freeSolo
                  options={interventionSuggestions}
                  inputValue={intervention}
                  onInputChange={(e, value) => handleInterventionChange(index, value)}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      variant="outlined"
                      fullWidth
                      placeholder="e.g., Amoxicillin, Vaccination, Physical Therapy"
                    />
                  )}
                  sx={{ flex: 1 }}
                />
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  onClick={() => removeIntervention(index)}
                  disabled={interventions.length <= 1}
                  sx={{ minWidth: 'auto', width: 40 }}
                >
                  <RemoveIcon />
                </Button>
              </Box>
            ))}

            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={addIntervention}
              size="small"
            >
              Add Intervention
            </Button>
          </Grid>

          {/* Outcomes (O) */}
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <BarChartIcon fontSize="small" sx={{ mr: 0.5 }} />
              Outcomes (O)
              <Chip size="small" label="Recommended" color="secondary" sx={{ ml: 1 }} />
            </Typography>

            {outcomes.map((outcome, index) => (
              <Box key={`outcome-${index}`} sx={{ display: 'flex', mb: 2, gap: 1 }}>
                <Autocomplete
                  freeSolo
                  options={outcomeSuggestions}
                  inputValue={outcome}
                  onInputChange={(e, value) => handleOutcomeChange(index, value)}
                  renderInput={(params) => (
                    <TextField
                      {...params}
                      variant="outlined"
                      fullWidth
                      placeholder="e.g., Mortality, Recovery Time, Quality of Life"
                    />
                  )}
                  sx={{ flex: 1 }}
                />
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  onClick={() => removeOutcome(index)}
                  disabled={outcomes.length <= 1}
                  sx={{ minWidth: 'auto', width: 40 }}
                >
                  <RemoveIcon />
                </Button>
              </Box>
            ))}

            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={addOutcome}
              size="small"
            >
              Add Outcome
            </Button>
          </Grid>

          {/* Study Design */}
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" gutterBottom>
              Study Design
            </Typography>
            <Autocomplete
              options={studyDesignOptions}
              getOptionLabel={(option) => option.label}
              value={studyDesignOptions.find(option => option.value === studyDesign) || null}
              onChange={(event, newValue) => {
                setStudyDesign(newValue ? newValue.value : '');
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  variant="outlined"
                  fullWidth
                  placeholder="Select study design"
                />
              )}
            />
          </Grid>

          {/* Years */}
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" gutterBottom>
              Publication Years
            </Typography>
            <Autocomplete
              options={yearOptions}
              getOptionLabel={(option) => option.label}
              value={yearOptions.find(option => option.value === years) || null}
              onChange={(event, newValue) => {
                setYears(newValue ? newValue.value : 5);
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  variant="outlined"
                  fullWidth
                  placeholder="Select timeframe"
                />
              )}
            />
          </Grid>

          {/* Maximum Results */}
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" gutterBottom>
              Maximum Results
            </Typography>
            <TextField
              variant="outlined"
              fullWidth
              type="number"
              value={maxResults}
              onChange={(e) => setMaxResults(Math.max(1, Math.min(200, parseInt(e.target.value) || 50)))}
              inputProps={{ min: 1, max: 200 }}
            />
          </Grid>

          {/* Action buttons */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                size="large"
                startIcon={isSearching ? <ButtonLoader size={20} /> : <SearchIcon />}
                onClick={handleSearch}
                disabled={!condition.trim() || isSearching}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </Button>

              <Button
                variant="outlined"
                startIcon={<ClearIcon />}
                onClick={handleReset}
              >
                Reset Form
              </Button>

              {searchResults && (
                <Button
                  variant="outlined"
                  color="secondary"
                  startIcon={<DownloadIcon />}
                >
                  Export Results
                </Button>
              )}
            </Box>
          </Grid>
        </Grid>

        {/* Search history */}
        {searchHistory.length > 0 && !searchResults && (
          <Box sx={{ mt: 4 }}>
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>
                  Search History ({searchHistory.length})
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {searchHistory.map((search) => (
                    <Grid item xs={12} key={search.id}>
                      <Paper
                        variant="outlined"
                        sx={{
                          p: 1.5,
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          cursor: 'pointer',
                          '&:hover': { bgcolor: 'action.hover' }
                        }}
                        onClick={() => loadSearchFromHistory(search)}
                      >
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            {search.condition}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            {new Date(search.timestamp).toLocaleString()} • {search.resultCount} results
                          </Typography>
                        </Box>
                        <Button
                          variant="outlined"
                          size="small"
                          color="primary"
                        >
                          Load
                        </Button>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </AccordionDetails>
            </Accordion>
          </Box>
        )}
      </Paper>

      {/* Search Results */}
      {searchResults && (
        <Paper
          elevation={3}
          sx={{
            p: 3,
            mb: 3,
            borderRadius: 2,
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
          }}
        >
          <Typography variant="h6" gutterBottom>
            Search Results ({searchResults.total_count || 0})
          </Typography>

          <Box sx={{
            p: 2,
            bgcolor: 'primary.light',
            color: 'primary.contrastText',
            borderRadius: 1,
            mb: 3
          }}>
            <Typography variant="subtitle2">PICO Query:</Typography>
            <Typography>{formatPICOQuery()}</Typography>
          </Box>

          {searchResults.articles?.length > 0 ? (
            <StaggeredList staggerDelay={100}>
              {searchResults.articles.map((article, index) => (
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

                    <CardContent sx={{ pt: 0 }}>
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        <strong>Authors:</strong> {Array.isArray(article.authors) ? article.authors.join(', ') : article.authors}
                      </Typography>

                      {/* Show preview of abstract */}
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        {expandedResult === index
                          ? article.abstract
                          : article.abstract?.substring(0, 200) + (article.abstract?.length > 200 ? '...' : '')}
                      </Typography>
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

          {/* Pagination */}
          {searchResults.pagination && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <Button
                disabled={!searchResults.pagination.has_previous}
                onClick={() => {/* Handle previous page */}}
              >
                Previous
              </Button>

              <Box sx={{ mx: 2, display: 'flex', alignItems: 'center' }}>
                Page {searchResults.pagination.page} of {searchResults.pagination.total_pages}
              </Box>

              <Button
                disabled={!searchResults.pagination.has_next}
                onClick={() => {/* Handle next page */}}
              >
                Next
              </Button>
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default PICOSearch;
