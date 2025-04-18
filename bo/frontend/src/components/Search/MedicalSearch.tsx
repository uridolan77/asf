import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  Container, 
  FormControl, 
  Grid, 
  InputLabel, 
  MenuItem, 
  Paper, 
  Select, 
  Tab, 
  Tabs, 
  TextField, 
  Typography, 
  Alert, 
  Chip,
  IconButton,
  Tooltip,
  Autocomplete,
  SelectChangeEvent
} from '@mui/material';
import { 
  Search as SearchIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Info as InfoIcon
} from '@mui/icons-material';

import { useMedicalSearch, StandardSearchParams, PICOSearchParams } from '../../hooks/useMedicalSearch';
import { ButtonLoader } from '../UI/LoadingIndicators';

/**
 * Medical literature search component
 * Supports both standard search and PICO search
 */
const MedicalSearch: React.FC = () => {
  // Use the medical search hook
  const {
    search,
    picoSearch,
    getSearchSuggestions,
    searchMethods,
    useMockData
  } = useMedicalSearch();
  
  // State for tab selection (0 = Standard, 1 = PICO)
  const [tabValue, setTabValue] = useState<number>(0);

  // Standard search state
  const [query, setQuery] = useState<string>('');
  const [maxResults, setMaxResults] = useState<number>(100);
  const [searchMethod, setSearchMethod] = useState<string>('pubmed');
  const [useGraphRag, setUseGraphRag] = useState<boolean>(false);

  // PICO search state
  const [condition, setCondition] = useState<string>('');
  const [interventions, setInterventions] = useState<string[]>(['']);
  const [outcomes, setOutcomes] = useState<string[]>(['']);
  const [population, setPopulation] = useState<string>('');
  const [studyDesign, setStudyDesign] = useState<string>('');
  const [years, setYears] = useState<number>(5);

  // Shared state
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(20);
  const [error, setError] = useState<string>('');

  // Get search suggestions
  const {
    data: suggestionsData
  } = getSearchSuggestions(query);

  // Extract suggestions
  const suggestions = suggestionsData?.suggestions || [];

  // Get the mutation functions
  const {
    mutate: executeStandardSearch,
    isPending: isStandardSearching,
    isError: isStandardSearchError,
    error: standardSearchError,
    data: standardSearchResults
  } = search();

  const {
    mutate: executePicoSearch,
    isPending: isPicoSearching,
    isError: isPicoSearchError,
    error: picoSearchError,
    data: picoSearchResults
  } = picoSearch();

  // Combined loading and error states
  const isSearching = isStandardSearching || isPicoSearching;
  const isSearchError = isStandardSearchError || isPicoSearchError;
  const searchError = standardSearchError || picoSearchError;
  const searchResults = tabValue === 0 ? standardSearchResults : picoSearchResults;

  // Handle tab change
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Add more intervention fields
  const addIntervention = () => {
    setInterventions([...interventions, '']);
  };

  // Remove intervention field
  const removeIntervention = (index: number) => {
    setInterventions(interventions.filter((_, i) => i !== index));
  };

  // Update intervention field
  const updateIntervention = (index: number, value: string) => {
    const newInterventions = [...interventions];
    newInterventions[index] = value;
    setInterventions(newInterventions);
  };

  // Add more outcome fields
  const addOutcome = () => {
    setOutcomes([...outcomes, '']);
  };

  // Remove outcome field
  const removeOutcome = (index: number) => {
    setOutcomes(outcomes.filter((_, i) => i !== index));
  };

  // Update outcome field
  const updateOutcome = (index: number, value: string) => {
    const newOutcomes = [...outcomes];
    newOutcomes[index] = value;
    setOutcomes(newOutcomes);
  };

  // Handle standard search
  const handleStandardSearch = () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setError('');

    const params: StandardSearchParams = {
      query: query.trim(),
      max_results: maxResults,
      page,
      page_size: pageSize,
      search_method: searchMethod,
      use_graph_rag: useGraphRag
    };

    executeStandardSearch(params);
  };

  // Handle PICO search
  const handlePicoSearch = () => {
    if (!condition.trim()) {
      setError('Please enter a medical condition');
      return;
    }

    setError('');

    // Filter out empty interventions and outcomes
    const filteredInterventions = interventions.filter(i => i.trim());
    const filteredOutcomes = outcomes.filter(o => o.trim());
    
    const params: PICOSearchParams = {
      condition: condition.trim(),
      interventions: filteredInterventions,
      outcomes: filteredOutcomes,
      population: population.trim() || undefined,
      study_design: studyDesign.trim() || undefined,
      years: years,
      max_results: maxResults,
      page,
      page_size: pageSize
    };

    executePicoSearch(params);
  };

  // Handle page change
  const handlePageChange = (newPage: number) => {
    setPage(newPage);
    
    // Re-run current search with new page
    if (tabValue === 0) {
      handleStandardSearch();
    } else {
      handlePicoSearch();
    }
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Medical Literature Search
      </Typography>
      
      {useMockData && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
        </Alert>
      )}
      
      <Paper sx={{ p: 2, mb: 4 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          aria-label="search tabs"
          sx={{ mb: 3 }}
        >
          <Tab label="Standard Search" />
          <Tab label="PICO Search" />
        </Tabs>
        
        {/* Standard Search Form */}
        {tabValue === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Autocomplete
                freeSolo
                options={suggestions}
                inputValue={query}
                onInputChange={(_, newValue) => setQuery(newValue)}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Search Query"
                    variant="outlined"
                    fullWidth
                    placeholder="Enter medical terms, conditions, treatments..."
                    required
                  />
                )}
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                label="Max Results"
                type="number"
                variant="outlined"
                fullWidth
                value={maxResults}
                onChange={(e) => setMaxResults(parseInt(e.target.value) || 100)}
                InputProps={{
                  inputProps: { min: 10, max: 1000 }
                }}
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth>
                <InputLabel>Search Method</InputLabel>
                <Select
                  value={searchMethod}
                  label="Search Method"
                  onChange={(e: SelectChangeEvent) => setSearchMethod(e.target.value)}
                >
                  {searchMethods.length > 0 ? (
                    searchMethods.map((method) => (
                      <MenuItem key={method} value={method}>{method}</MenuItem>
                    ))
                  ) : (
                    <>
                      <MenuItem value="pubmed">PubMed</MenuItem>
                      <MenuItem value="clinical_trials">Clinical Trials</MenuItem>
                      <MenuItem value="graph_rag">GraphRAG</MenuItem>
                    </>
                  )}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth>
                <InputLabel>Use GraphRAG</InputLabel>
                <Select
                  value={useGraphRag.toString()}
                  label="Use GraphRAG"
                  onChange={(e: SelectChangeEvent) => setUseGraphRag(e.target.value === 'true')}
                >
                  <MenuItem value="true">Yes</MenuItem>
                  <MenuItem value="false">No</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleStandardSearch}
                disabled={isSearching}
                startIcon={isSearching ? <ButtonLoader size={20} /> : <SearchIcon />}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </Button>
            </Grid>
          </Grid>
        )}
        
        {/* PICO Search Form */}
        {tabValue === 1 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                label="Medical Condition (P)"
                variant="outlined"
                fullWidth
                value={condition}
                onChange={(e) => setCondition(e.target.value)}
                helperText="Enter the medical condition or problem"
                required
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                label="Population"
                variant="outlined"
                fullWidth
                value={population}
                onChange={(e) => setPopulation(e.target.value)}
                helperText="Describe the patient population (e.g., adults, children, elderly)"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Interventions (I)
                <Tooltip title="Enter treatments, drugs, or procedures being studied">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Typography>
              
              {interventions.map((intervention, index) => (
                <Box key={index} sx={{ display: 'flex', mb: 1 }}>
                  <TextField
                    variant="outlined"
                    fullWidth
                    value={intervention}
                    onChange={(e) => updateIntervention(index, e.target.value)}
                    placeholder={`Intervention ${index + 1}`}
                    sx={{ mr: 1 }}
                  />
                  <IconButton 
                    color="error" 
                    onClick={() => removeIntervention(index)}
                    disabled={interventions.length === 1}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
              ))}
              
              <Button
                startIcon={<AddIcon />}
                onClick={addIntervention}
                variant="outlined"
                size="small"
                sx={{ mt: 1 }}
              >
                Add Intervention
              </Button>
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Outcomes (O)
                <Tooltip title="Enter the outcomes or results being measured">
                  <IconButton size="small">
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Typography>
              
              {outcomes.map((outcome, index) => (
                <Box key={index} sx={{ display: 'flex', mb: 1 }}>
                  <TextField
                    variant="outlined"
                    fullWidth
                    value={outcome}
                    onChange={(e) => updateOutcome(index, e.target.value)}
                    placeholder={`Outcome ${index + 1}`}
                    sx={{ mr: 1 }}
                  />
                  <IconButton 
                    color="error" 
                    onClick={() => removeOutcome(index)}
                    disabled={outcomes.length === 1}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Box>
              ))}
              
              <Button
                startIcon={<AddIcon />}
                onClick={addOutcome}
                variant="outlined"
                size="small"
                sx={{ mt: 1 }}
              >
                Add Outcome
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                label="Study Design"
                variant="outlined"
                fullWidth
                value={studyDesign}
                onChange={(e) => setStudyDesign(e.target.value)}
                helperText="E.g., RCT, cohort study, meta-analysis"
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                label="Years"
                type="number"
                variant="outlined"
                fullWidth
                value={years}
                onChange={(e) => setYears(parseInt(e.target.value) || 5)}
                helperText="Limit to studies from the past X years"
                InputProps={{
                  inputProps: { min: 1, max: 50 }
                }}
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                label="Max Results"
                type="number"
                variant="outlined"
                fullWidth
                value={maxResults}
                onChange={(e) => setMaxResults(parseInt(e.target.value) || 100)}
                InputProps={{
                  inputProps: { min: 10, max: 1000 }
                }}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handlePicoSearch}
                disabled={isSearching}
                startIcon={isSearching ? <ButtonLoader size={20} /> : <SearchIcon />}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </Button>
            </Grid>
          </Grid>
        )}
      </Paper>
      
      {/* Error message */}
      {(error || isSearchError) && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error || searchError?.message || 'An error occurred during search'}
        </Alert>
      )}
      
      {/* Search Results */}
      {searchResults && (
        <Box>
          <Typography variant="h5" gutterBottom>
            Search Results
            <Chip 
              label={`${searchResults.total_results || searchResults.total_count || 0} results`} 
              color="primary" 
              size="small" 
              sx={{ ml: 1 }} 
            />
          </Typography>
          
          {((searchResults.articles && searchResults.articles.length > 0) || 
            (searchResults.results && searchResults.results.length > 0)) ? (
            <Grid container spacing={3}>
              {(searchResults.articles || searchResults.results || []).map((article, index) => (
                <Grid item xs={12} key={article.id || index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {article.title}
                      </Typography>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        {article.authors?.join(', ')} - {article.journal} ({article.year})
                      </Typography>
                      <Typography variant="body2" paragraph>
                        {article.abstract}
                      </Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Chip 
                          label={`Relevance: ${(article.relevance_score * 100).toFixed(0)}%`} 
                          color={article.relevance_score > 0.7 ? 'success' : 'primary'} 
                          size="small" 
                        />
                        <Button 
                          size="small" 
                          href={article.source === 'pubmed' ? `https://pubmed.ncbi.nlm.nih.gov/${article.id}` : '#'}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          View Source
                        </Button>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography>No results found</Typography>
          )}
          
          {/* Pagination */}
          {/* Add pagination component here */}
        </Box>
      )}
    </Container>
  );
};

export default MedicalSearch;
