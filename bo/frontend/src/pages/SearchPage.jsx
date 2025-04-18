import React, { useState, useEffect } from 'react';
import {
  Box, Container, Grid, Card, CardContent, Typography, TextField, Button,
  Tabs, Tab, FormControl, InputLabel, MenuItem, Select, Paper, Chip,
  List, ListItem, ListItemText, Divider, CircularProgress, Alert, IconButton,
  Accordion, AccordionSummary, AccordionDetails, Rating
} from '@mui/material';
import {
  Search as SearchIcon,
  Bookmark as BookmarkIcon,
  FilterList as FilterListIcon,
  ExpandMore as ExpandMoreIcon,
  History as HistoryIcon,
  Clear as ClearIcon,
  Biotech as BiotechIcon,
  Download as DownloadIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import { useAuth } from '../context/AuthContext.jsx';
import { useNotification } from '../context/NotificationContext.jsx';

// Predefined search fields for PICO format
const PICO_FIELDS = [
  { id: 'population', label: 'Population/Problem', helper: 'e.g., adults with hypertension' },
  { id: 'intervention', label: 'Intervention', helper: 'e.g., ACE inhibitors' },
  { id: 'comparison', label: 'Comparison', helper: 'e.g., ARBs (optional)' },
  { id: 'outcome', label: 'Outcome', helper: 'e.g., reduction in blood pressure' }
];

const SearchPage = () => {
  const { user, api } = useAuth();
  const { showSuccess, showError } = useNotification();
  
  // Tab state
  const [tabValue, setTabValue] = useState(0);
  
  // Search states
  const [standardQuery, setStandardQuery] = useState('');
  const [picoData, setPicoData] = useState({
    population: '',
    intervention: '',
    comparison: '',
    outcome: ''
  });
  const [searchFilters, setSearchFilters] = useState({
    yearStart: 2015,
    yearEnd: new Date().getFullYear(),
    studyTypes: [],
    sources: []
  });
  
  // UI states
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [showFilters, setShowFilters] = useState(false);
  const [recentSearches, setRecentSearches] = useState([]);
  
  // Load recent searches on component mount
  useEffect(() => {
    loadRecentSearches();
  }, []);
  
  // Load recent searches from localStorage or API
  const loadRecentSearches = async () => {
    try {
      const response = await api.get('/api/medical/search/history');
      if (response.data && response.data.searches) {
        setRecentSearches(response.data.searches);
      }
    } catch (error) {
      console.error('Failed to load search history:', error);
      // Fallback to localStorage
      const savedSearches = localStorage.getItem('recentSearches');
      if (savedSearches) {
        setRecentSearches(JSON.parse(savedSearches));
      }
    }
  };
  
  // Save search to history
  const saveToHistory = async (queryText, type = 'standard') => {
    try {
      await api.post('/api/medical/search/history', {
        query: queryText,
        type,
        timestamp: new Date().toISOString()
      });
      
      // Refresh search history
      loadRecentSearches();
    } catch (error) {
      console.error('Failed to save search history:', error);
      // Fallback to localStorage
      const newSearch = {
        id: Date.now(),
        query: queryText,
        type,
        timestamp: new Date().toISOString()
      };
      
      const updatedSearches = [newSearch, ...recentSearches].slice(0, 10);
      setRecentSearches(updatedSearches);
      localStorage.setItem('recentSearches', JSON.stringify(updatedSearches));
    }
  };
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Handle standard search submission
  const handleStandardSearch = async () => {
    if (!standardQuery.trim()) {
      showError('Please enter a search query');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/medical/search', {
        query: standardQuery,
        filters: searchFilters
      });
      
      if (response.data && response.data.results) {
        setSearchResults(response.data.results);
        saveToHistory(standardQuery, 'standard');
      }
    } catch (error) {
      console.error('Search error:', error);
      setError('An error occurred while searching. Please try again.');
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle PICO search submission
  const handlePicoSearch = async () => {
    // Check if at least population and intervention are provided
    if (!picoData.population.trim() || !picoData.intervention.trim()) {
      showError('Please provide at least Population and Intervention fields');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/api/medical/pico-search', {
        ...picoData,
        filters: searchFilters
      });
      
      if (response.data && response.data.results) {
        setSearchResults(response.data.results);
        const picoQuery = `P: ${picoData.population}, I: ${picoData.intervention}, C: ${picoData.comparison}, O: ${picoData.outcome}`;
        saveToHistory(picoQuery, 'pico');
      }
    } catch (error) {
      console.error('PICO search error:', error);
      setError('An error occurred while performing PICO search. Please try again.');
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle PICO field change
  const handlePicoFieldChange = (field, value) => {
    setPicoData({
      ...picoData,
      [field]: value
    });
  };
  
  // Load a previous search
  const loadPreviousSearch = (search) => {
    if (search.type === 'standard') {
      setTabValue(0);
      setStandardQuery(search.query);
    } else if (search.type === 'pico') {
      setTabValue(1);
      // Parse PICO query back into fields
      const picoString = search.query;
      const parts = {
        population: '',
        intervention: '',
        comparison: '',
        outcome: ''
      };
      
      if (picoString.includes('P:')) {
        const match = picoString.match(/P:\s*([^,]*)/);
        if (match) parts.population = match[1].trim();
      }
      
      if (picoString.includes('I:')) {
        const match = picoString.match(/I:\s*([^,]*)/);
        if (match) parts.intervention = match[1].trim();
      }
      
      if (picoString.includes('C:')) {
        const match = picoString.match(/C:\s*([^,]*)/);
        if (match) parts.comparison = match[1].trim();
      }
      
      if (picoString.includes('O:')) {
        const match = picoString.match(/O:\s*([^,]*)/);
        if (match) parts.outcome = match[1].trim();
      }
      
      setPicoData(parts);
    }
  };
  
  // Clear search results
  const clearResults = () => {
    setSearchResults([]);
    setSelectedResult(null);
  };
  
  // Handle filter changes
  const handleFilterChange = (filter, value) => {
    setSearchFilters({
      ...searchFilters,
      [filter]: value
    });
  };
  
  // View result details
  const viewResultDetails = async (resultId) => {
    try {
      setLoading(true);
      const response = await api.get(`/api/medical/article/${resultId}`);
      if (response.data) {
        setSelectedResult(response.data);
      }
    } catch (error) {
      console.error('Failed to load article details:', error);
      showError('Failed to load article details');
    } finally {
      setLoading(false);
    }
  };
  
  // Bookmark/save result
  const bookmarkResult = async (resultId) => {
    try {
      await api.post('/api/medical/bookmarks', { article_id: resultId });
      showSuccess('Article bookmarked successfully');
    } catch (error) {
      console.error('Failed to bookmark article:', error);
      showError('Failed to bookmark article');
    }
  };
  
  // Export results to CSV
  const exportResults = async () => {
    try {
      const response = await api.get('/api/medical/search/export', {
        params: { format: 'csv' },
        responseType: 'blob'
      });
      
      // Create a download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `search-results-${new Date().toISOString().slice(0, 10)}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      showSuccess('Export successful');
    } catch (error) {
      console.error('Failed to export results:', error);
      showError('Failed to export results');
    }
  };
  
  // Study type options
  const studyTypes = [
    'Randomized Controlled Trial',
    'Systematic Review',
    'Meta-Analysis',
    'Cohort Study',
    'Case-Control Study',
    'Case Report',
    'Clinical Trial'
  ];
  
  // Source options
  const sources = [
    'PubMed',
    'Cochrane Library',
    'EMBASE',
    'Clinical Trials',
    'MedRxiv'
  ];
  
  // Action buttons for page header
  const pageActions = (
    <Box>
      {searchResults.length > 0 && (
        <Button
          variant="outlined"
          startIcon={<DownloadIcon />}
          onClick={exportResults}
          sx={{ mr: 1 }}
        >
          Export
        </Button>
      )}
      <Button
        variant="outlined"
        color={showFilters ? 'primary' : 'inherit'}
        startIcon={<FilterListIcon />}
        onClick={() => setShowFilters(!showFilters)}
      >
        Filters
      </Button>
    </Box>
  );
  
  return (
    <PageLayout
      title="Medical Literature Search"
      breadcrumbs={[{ label: 'Search', path: '/search' }]}
      user={user}
      actions={pageActions}
    >
      {/* Search tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          aria-label="search tabs"
          variant="fullWidth"
        >
          <Tab 
            icon={<SearchIcon />} 
            label="Standard Search" 
            id="tab-0" 
            aria-controls="tabpanel-0" 
          />
          <Tab 
            icon={<BiotechIcon />} 
            label="PICO Search" 
            id="tab-1" 
            aria-controls="tabpanel-1" 
          />
        </Tabs>
      </Paper>
      
      {/* Filter panel */}
      {showFilters && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Search Filters
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Year From"
                type="number"
                fullWidth
                value={searchFilters.yearStart}
                onChange={(e) => handleFilterChange('yearStart', e.target.value)}
                InputProps={{ inputProps: { min: 1900, max: new Date().getFullYear() } }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Year To"
                type="number"
                fullWidth
                value={searchFilters.yearEnd}
                onChange={(e) => handleFilterChange('yearEnd', e.target.value)}
                InputProps={{ inputProps: { min: 1900, max: new Date().getFullYear() } }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel id="study-type-label">Study Types</InputLabel>
                <Select
                  labelId="study-type-label"
                  multiple
                  value={searchFilters.studyTypes}
                  onChange={(e) => handleFilterChange('studyTypes', e.target.value)}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {studyTypes.map((type) => (
                    <MenuItem key={type} value={type}>
                      {type}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel id="sources-label">Sources</InputLabel>
                <Select
                  labelId="sources-label"
                  multiple
                  value={searchFilters.sources}
                  onChange={(e) => handleFilterChange('sources', e.target.value)}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {sources.map((source) => (
                    <MenuItem key={source} value={source}>
                      {source}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Paper>
      )}
      
      {/* Standard search panel */}
      <Box role="tabpanel" hidden={tabValue !== 0} id="tabpanel-0" aria-labelledby="tab-0">
        {tabValue === 0 && (
          <Paper sx={{ p: 2, mb: 3 }}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  label="Search medical literature"
                  variant="outlined"
                  fullWidth
                  value={standardQuery}
                  onChange={(e) => setStandardQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleStandardSearch()}
                  placeholder="Enter symptoms, conditions, treatments, or keywords..."
                  InputProps={{
                    endAdornment: (
                      <Button
                        variant="contained"
                        onClick={handleStandardSearch}
                        startIcon={<SearchIcon />}
                        disabled={loading}
                      >
                        Search
                      </Button>
                    ),
                  }}
                />
              </Grid>
            </Grid>
          </Paper>
        )}
      </Box>
      
      {/* PICO search panel */}
      <Box role="tabpanel" hidden={tabValue !== 1} id="tabpanel-1" aria-labelledby="tab-1">
        {tabValue === 1 && (
          <Paper sx={{ p: 2, mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              PICO Framework Search
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Use the PICO framework to structure your clinical question: 
              Population, Intervention, Comparison, and Outcome.
            </Typography>
            
            <Grid container spacing={2}>
              {PICO_FIELDS.map((field) => (
                <Grid item xs={12} md={6} key={field.id}>
                  <TextField
                    label={field.label}
                    variant="outlined"
                    fullWidth
                    value={picoData[field.id]}
                    onChange={(e) => handlePicoFieldChange(field.id, e.target.value)}
                    helperText={field.helper}
                    required={field.id === 'population' || field.id === 'intervention'}
                  />
                </Grid>
              ))}
              
              <Grid item xs={12} sx={{ mt: 1 }}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handlePicoSearch}
                  startIcon={<SearchIcon />}
                  disabled={loading}
                  fullWidth
                >
                  Run PICO Search
                </Button>
              </Grid>
            </Grid>
          </Paper>
        )}
      </Box>
      
      {/* Recent searches panel */}
      {recentSearches.length > 0 && !searchResults.length && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="h6">
              <HistoryIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
              Recent Searches
            </Typography>
          </Box>
          <List>
            {recentSearches.slice(0, 5).map((search, index) => (
              <React.Fragment key={search.id || index}>
                <ListItem button onClick={() => loadPreviousSearch(search)}>
                  <ListItemText
                    primary={search.query}
                    secondary={`${search.type === 'pico' ? 'PICO Search' : 'Standard Search'} • ${new Date(search.timestamp).toLocaleString()}`}
                  />
                </ListItem>
                {index < recentSearches.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </Paper>
      )}
      
      {/* Loading indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}
      
      {/* Error message */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {/* Search results */}
      {searchResults.length > 0 && !loading && (
        <Box sx={{ mt: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Search Results ({searchResults.length})
            </Typography>
            <Button
              startIcon={<ClearIcon />}
              onClick={clearResults}
              size="small"
            >
              Clear Results
            </Button>
          </Box>
          
          <Grid container spacing={3}>
            {/* Results list */}
            <Grid item xs={12} md={selectedResult ? 6 : 12}>
              <List sx={{ bgcolor: 'background.paper' }}>
                {searchResults.map((result, index) => (
                  <React.Fragment key={result.id}>
                    <ListItem
                      alignItems="flex-start"
                      button
                      selected={selectedResult && selectedResult.id === result.id}
                      onClick={() => viewResultDetails(result.id)}
                    >
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="subtitle1" component="div">
                              {result.title}
                            </Typography>
                            <Chip
                              label={result.type}
                              size="small"
                              color={result.type.includes('Meta-Analysis') || result.type.includes('Systematic Review') ? 'success' : 'default'}
                            />
                          </Box>
                        }
                        secondary={
                          <>
                            <Typography variant="body2" color="text.primary" component="span">
                              {result.authors.join(', ')} • {result.journal}
                            </Typography>
                            <Typography variant="body2" component="div">
                              {result.year} • {result.citations} citations
                            </Typography>
                            <Typography variant="body2" component="div" sx={{ mt: 1 }}>
                              {result.abstract?.substring(0, 150)}...
                            </Typography>
                          </>
                        }
                      />
                    </ListItem>
                    {index < searchResults.length - 1 && <Divider />}
                  </React.Fragment>
                ))}
              </List>
            </Grid>
            
            {/* Selected result details */}
            {selectedResult && (
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Typography variant="h5">
                      {selectedResult.title}
                    </Typography>
                    <IconButton onClick={() => bookmarkResult(selectedResult.id)}>
                      <BookmarkIcon />
                    </IconButton>
                  </Box>
                  
                  <Chip
                    label={selectedResult.type}
                    sx={{ mb: 2 }}
                    color={selectedResult.type.includes('Meta-Analysis') || selectedResult.type.includes('Systematic Review') ? 'success' : 'default'}
                  />
                  
                  <Typography variant="subtitle1" gutterBottom>
                    {selectedResult.authors.join(', ')}
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {selectedResult.journal} • {selectedResult.year} • DOI: {selectedResult.doi}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', my: 2 }}>
                    <Typography variant="body2" component="span" sx={{ mr: 1 }}>
                      Quality:
                    </Typography>
                    <Rating value={selectedResult.quality_score || 0} readOnly precision={0.5} />
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="h6" gutterBottom>
                    Abstract
                  </Typography>
                  <Typography variant="body1" paragraph>
                    {selectedResult.abstract}
                  </Typography>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="h6" gutterBottom>
                    Key Findings
                  </Typography>
                  <List dense>
                    {selectedResult.key_findings?.map((finding, idx) => (
                      <ListItem key={idx}>
                        <ListItemText primary={finding} />
                      </ListItem>
                    ))}
                  </List>
                  
                  {selectedResult.methodology && (
                    <>
                      <Divider sx={{ my: 2 }} />
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography variant="h6">Methodology</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Typography variant="body1">
                            {selectedResult.methodology}
                          </Typography>
                        </AccordionDetails>
                      </Accordion>
                    </>
                  )}
                  
                  {selectedResult.statistical_data && (
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography variant="h6">Statistical Data</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Typography variant="body1">
                          {selectedResult.statistical_data}
                        </Typography>
                      </AccordionDetails>
                    </Accordion>
                  )}
                  
                  <Box sx={{ mt: 3 }}>
                    <Button variant="outlined" href={selectedResult.url} target="_blank">
                      View Full Article
                    </Button>
                  </Box>
                </Paper>
              </Grid>
            )}
          </Grid>
        </Box>
      )}
    </PageLayout>
  );
};

export default SearchPage;