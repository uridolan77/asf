import React, { useState } from 'react';
import { 
  Box, Button, Card, CardContent, Container, FormControl, 
  Grid, InputLabel, MenuItem, Paper, Select, Tab, Tabs, 
  TextField, Typography, CircularProgress
} from '@mui/material';
import { useAuth } from '../context/AuthContext.jsx';

/**
 * Medical literature search component
 * Supports both standard search and PICO search
 */
const MedicalSearch = () => {
  const { api } = useAuth(); // Use the authenticated API client
  
  // State for tab selection (0 = Standard, 1 = PICO)
  const [tabValue, setTabValue] = useState(0);

  // Standard search state
  const [query, setQuery] = useState('');
  const [maxResults, setMaxResults] = useState(100);
  const [searchMethod, setSearchMethod] = useState('pubmed');
  const [useGraphRag, setUseGraphRag] = useState(false);

  // PICO search state
  const [condition, setCondition] = useState('');
  const [interventions, setInterventions] = useState(['']);
  const [outcomes, setOutcomes] = useState(['']);
  const [population, setPopulation] = useState('');
  const [studyDesign, setStudyDesign] = useState('');
  const [years, setYears] = useState(5);

  // Shared state
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Add more intervention fields
  const addIntervention = () => {
    setInterventions([...interventions, '']);
  };

  // Update intervention at specific index
  const updateIntervention = (index, value) => {
    const newInterventions = [...interventions];
    newInterventions[index] = value;
    setInterventions(newInterventions);
  };

  // Add more outcome fields
  const addOutcome = () => {
    setOutcomes([...outcomes, '']);
  };

  // Update outcome at specific index
  const updateOutcome = (index, value) => {
    const newOutcomes = [...outcomes];
    newOutcomes[index] = value;
    setOutcomes(newOutcomes);
  };

  // Handle standard search
  const handleStandardSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await api.post('/api/knowledge-base/search', {
        query,
        max_results: maxResults,
        page,
        page_size: pageSize,
        search_method: searchMethod,
        use_graph_rag: useGraphRag
      });
      
      setResults(response.data);
    } catch (err) {
      setError(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle PICO search
  const handlePicoSearch = async () => {
    if (!condition.trim()) {
      setError('Please enter a medical condition');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      // Filter out empty interventions and outcomes
      const filteredInterventions = interventions.filter(i => i.trim());
      const filteredOutcomes = outcomes.filter(o => o.trim());
      
      const response = await api.post('/api/knowledge-base/search-pico', {
        condition,
        interventions: filteredInterventions,
        outcomes: filteredOutcomes,
        population,
        study_design: studyDesign,
        years,
        max_results: maxResults,
        page,
        page_size: pageSize
      });
      
      setResults(response.data);
    } catch (err) {
      setError(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle page change
  const handlePageChange = async (newPage) => {
    setPage(newPage);
    
    // Re-run current search with new page
    if (tabValue === 0) {
      await handleStandardSearch();
    } else {
      await handlePicoSearch();
    }
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Medical Literature Search
      </Typography>
      
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
                InputProps={{ inputProps: { min: 1, max: 500 } }}
                onChange={(e) => setMaxResults(Math.min(500, Math.max(1, parseInt(e.target.value) || 1)))}
              />
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <FormControl fullWidth>
                <InputLabel>Search Method</InputLabel>
                <Select
                  value={searchMethod}
                  label="Search Method"
                  onChange={(e) => setSearchMethod(e.target.value)}
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
                  value={useGraphRag}
                  label="Use GraphRAG"
                  onChange={(e) => setUseGraphRag(e.target.value)}
                >
                  <MenuItem value={true}>Yes</MenuItem>
                  <MenuItem value={false}>No</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleStandardSearch}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                Search
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
                InputProps={{ inputProps: { min: 1, max: 50 } }}
                onChange={(e) => setYears(Math.min(50, Math.max(1, parseInt(e.target.value) || 1)))}
                helperText="Publication years to include"
              />
            </Grid>
            
            <Grid item xs={12}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handlePicoSearch}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                Search
              </Button>
            </Grid>
          </Grid>
        )}
        
        {error && (
          <Box sx={{ mt: 3, p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
            <Typography color="error">{error}</Typography>
          </Box>
        )}
      </Paper>
      
      {/* Search Results */}
      {results && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h5" gutterBottom>
            Search Results
          </Typography>
          <Typography variant="subtitle1" gutterBottom>
            Found {results.total_count} results for "{results.query}"
          </Typography>
          
          {results.results.length > 0 ? (
            <Grid container spacing={3}>
              {results.results.map((article, index) => (
                <Grid item xs={12} key={article.pmid || index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        {article.title}
                      </Typography>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        {article.authors} - {article.journal} ({article.publication_date})
                      </Typography>
                      <Typography variant="body2" paragraph>
                        {article.abstract}
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
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography>No results found</Typography>
          )}
          
          {/* Pagination */}
          {results.pagination && (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
              <Button 
                disabled={!results.pagination.has_previous} 
                onClick={() => handlePageChange(page - 1)}
              >
                Previous
              </Button>
              <Box sx={{ mx: 2, display: 'flex', alignItems: 'center' }}>
                Page {results.pagination.page} of {results.pagination.total_pages}
              </Box>
              <Button 
                disabled={!results.pagination.has_next} 
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