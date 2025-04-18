import React, { useState } from 'react';
import { 
  Box, Button, Card, CardContent, Container, FormControl, 
  Grid, InputLabel, MenuItem, Paper, Select, Tab, Tabs, 
  TextField, Typography, CircularProgress, Alert, SelectChangeEvent
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';

import { useKnowledgeBase, SearchParams } from '../hooks/useKnowledgeBase';
import { usePICOSearch, PICOSearchParams } from '../hooks/usePICOSearch';
import { useFeatureFlags } from '../context/FeatureFlagContext';
import { ButtonLoader } from './UI/LoadingIndicators';

/**
 * Medical literature search component
 * Supports both standard search and PICO search
 */
const MedicalSearch: React.FC = () => {
  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');
  
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

  // Knowledge base hooks
  const {
    searchKnowledgeBase
  } = useKnowledgeBase();

  // PICO search hooks
  const {
    picoSearch
  } = usePICOSearch();

  // Get the mutation functions
  const {
    mutate: executeStandardSearch,
    isPending: isStandardSearching,
    isError: isStandardSearchError,
    error: standardSearchError,
    data: standardSearchResults
  } = searchKnowledgeBase();

  const {
    mutate: executePicoSearch,
    isPending: isPicoSearching,
    isError: isPicoSearchError,
    error: picoSearchError,
    data: picoSearchResults
  } = picoSearch();

  // Handle tab change
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Add more intervention fields
  const addIntervention = () => {
    setInterventions([...interventions, '']);
  };

  // Update intervention at specific index
  const updateIntervention = (index: number, value: string) => {
    const newInterventions = [...interventions];
    newInterventions[index] = value;
    setInterventions(newInterventions);
  };

  // Add more outcome fields
  const addOutcome = () => {
    setOutcomes([...outcomes, '']);
  };

  // Update outcome at specific index
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

    const params: SearchParams = {
      query: query.trim(),
      max_results: maxResults,
      page,
      page_size: pageSize,
      search_method: searchMethod as 'semantic' | 'keyword' | 'hybrid',
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

  // Get current results based on active tab
  const results = tabValue === 0 ? standardSearchResults : picoSearchResults;
  const isSearching = tabValue === 0 ? isStandardSearching : isPicoSearching;
  const isSearchError = tabValue === 0 ? isStandardSearchError : isPicoSearchError;
  const searchError = tabValue === 0 ? standardSearchError : picoSearchError;

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
              <TextField
                label="Search Query"
                variant="outlined"
                fullWidth
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                helperText="Enter keywords to search medical literature"
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                label="Max Results"
                variant="outlined"
                fullWidth
                type="number"
                value={maxResults}
                inputProps={{ min: 1, max: 500 }}
                onChange={(e) => setMaxResults(Math.min(500, Math.max(1, parseInt(e.target.value) || 1)))}
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
                  <MenuItem value="pubmed">PubMed</MenuItem>
                  <MenuItem value="clinical_trials">Clinical Trials</MenuItem>
                  <MenuItem value="graph_rag">GraphRAG</MenuItem>
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
            
            {/* Interventions */}
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Interventions (I)
              </Typography>
              {interventions.map((intervention, index) => (
                <Box key={index} sx={{ mb: 2 }}>
                  <TextField
                    label={`Intervention ${index + 1}`}
                    variant="outlined"
                    fullWidth
                    value={intervention}
                    onChange={(e) => updateIntervention(index, e.target.value)}
                    helperText="Treatment, diagnostic test, exposure, etc."
                  />
                </Box>
              ))}
              <Button 
                variant="outlined" 
                size="small" 
                onClick={addIntervention}
              >
                Add Intervention
              </Button>
            </Grid>
            
            {/* Outcomes */}
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom>
                Outcomes (O)
              </Typography>
              {outcomes.map((outcome, index) => (
                <Box key={index} sx={{ mb: 2 }}>
                  <TextField
                    label={`Outcome ${index + 1}`}
                    variant="outlined"
                    fullWidth
                    value={outcome}
                    onChange={(e) => updateOutcome(index, e.target.value)}
                    helperText="What you want to measure or achieve"
                  />
                </Box>
              ))}
              <Button 
                variant="outlined" 
                size="small" 
                onClick={addOutcome}
              >
                Add Outcome
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                label="Population (P)"
                variant="outlined"
                fullWidth
                value={population}
                onChange={(e) => setPopulation(e.target.value)}
                helperText="Optional: specific patient population"
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                label="Study Design"
                variant="outlined"
                fullWidth
                value={studyDesign}
                onChange={(e) => setStudyDesign(e.target.value)}
                helperText="Optional: RCT, meta-analysis, etc."
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <TextField
                label="Years"
                variant="outlined"
                fullWidth
                type="number"
                value={years}
                inputProps={{ min: 1, max: 50 }}
                onChange={(e) => setYears(Math.min(50, Math.max(1, parseInt(e.target.value) || 1)))}
                helperText="Publication years to include"
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
        
        {error && (
          <Box sx={{ mt: 3, p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
            <Typography color="error">{error}</Typography>
          </Box>
        )}

        {isSearchError && (
          <Alert severity="error" sx={{ mt: 3 }}>
            {searchError?.message || 'An error occurred during search'}
          </Alert>
        )}
      </Paper>
      
      {/* Search Results */}
      {results && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h5" gutterBottom>
            Search Results
          </Typography>
          <Typography variant="subtitle1" gutterBottom>
            Found {results.total_count} results for "{results.query || condition}"
          </Typography>
          
          {results.results?.length > 0 || results.articles?.length > 0 ? (
            <Grid container spacing={3}>
              {(results.results || results.articles || []).map((article: any, index: number) => (
                <Grid item xs={12} key={article.id || article.pmid || index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {article.title}
                      </Typography>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        {article.authors?.join(', ') || article.source} - {article.journal || article.source_type} ({article.year || article.date || 'N/A'})
                      </Typography>
                      <Typography variant="body2" paragraph>
                        {article.abstract || article.content}
                      </Typography>
                      {article.pmid && (
                        <Button 
                          size="small" 
                          href={`https://pubmed.ncbi.nlm.nih.gov/${article.pmid}`}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          View on PubMed
                        </Button>
                      )}
                      {article.url && (
                        <Button 
                          size="small" 
                          href={article.url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          View Source
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography>No results found</Typography>
          )}
          
          {/* Pagination */}
          {results.page && results.page_size && results.total_count && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <Button 
                disabled={results.page <= 1} 
                onClick={() => handlePageChange(page - 1)}
              >
                Previous
              </Button>
              <Box sx={{ mx: 2, display: 'flex', alignItems: 'center' }}>
                Page {results.page} of {Math.ceil(results.total_count / results.page_size)}
              </Box>
              <Button 
                disabled={results.page >= Math.ceil(results.total_count / results.page_size)} 
                onClick={() => handlePageChange(page + 1)}
              >
                Next
              </Button>
            </Box>
          )}
        </Paper>
      )}
    </Container>
  );
};

export default MedicalSearch;
