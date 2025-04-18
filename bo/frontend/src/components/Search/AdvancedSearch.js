import React, { useState, useEffect } from 'react';
import { 
  Box, Paper, Typography, TextField, Button, Grid, Chip, 
  IconButton, Autocomplete, FormControl, InputLabel, 
  Select, MenuItem, Divider, Drawer, Slider, Switch,
  FormControlLabel, CircularProgress
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterAlt as FilterIcon,
  Close as CloseIcon,
  Add as AddIcon,
  Clear as ClearIcon,
  Download as DownloadIcon,
  Save as SaveIcon,
  Tune as TuneIcon
} from '@mui/icons-material';
import axios from 'axios';
import debounce from 'lodash/debounce';

/**
 * Enhanced medical search component with filters, autocomplete, and type-ahead suggestions
 */
const AdvancedSearch = ({ onSearchResults }) => {
  // Main search state
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [recentSearches, setRecentSearches] = useState([]);
  const [error, setError] = useState('');
  
  // Filters state
  const [filters, setFilters] = useState({
    yearRange: [1950, 2025],
    sources: [],
    articleTypes: [],
    authors: [],
    journals: [],
    includeFreeFullText: false,
    includeTrials: true,
    excludeRetractions: true,
    selectedLanguages: ['en']
  });
  
  // Available filter options (normally would come from API)
  const filterOptions = {
    articleTypes: [
      'Clinical Trial', 'Review', 'Meta-Analysis', 'Randomized Controlled Trial',
      'Systematic Review', 'Case Report', 'Observational Study', 'Guideline'
    ],
    languages: [
      { code: 'en', label: 'English' },
      { code: 'fr', label: 'French' },
      { code: 'es', label: 'Spanish' },
      { code: 'de', label: 'German' },
      { code: 'it', label: 'Italian' },
      { code: 'ja', label: 'Japanese' },
      { code: 'zh', label: 'Chinese' }
    ],
    sources: [
      'PubMed', 'MEDLINE', 'ClinicalTrials.gov', 'MedRxiv', 'Cochrane', 'Embase'
    ]
  };
  
  // Load recent searches from localStorage
  useEffect(() => {
    const savedSearches = localStorage.getItem('recentSearches');
    if (savedSearches) {
      setRecentSearches(JSON.parse(savedSearches));
    }
  }, []);
  
  // Save recent searches to localStorage
  const saveRecentSearch = (searchQuery) => {
    const updatedSearches = [
      searchQuery,
      ...recentSearches.filter(s => s !== searchQuery).slice(0, 9)
    ];
    setRecentSearches(updatedSearches);
    localStorage.setItem('recentSearches', JSON.stringify(updatedSearches));
  };
  
  // Debounced function to fetch suggestions as the user types
  const fetchSuggestions = debounce(async (input) => {
    if (!input || input.length < 2) {
      setSuggestions([]);
      return;
    }
    
    try {
      const response = await axios.get(`http://localhost:8000/api/medical/terminology/search?query=${input}&max_results=10`);
      if (response.data.success) {
        // Extract term suggestions from terminology service
        const terms = response.data.data.results.map(result => result.preferredTerm);
        setSuggestions(terms);
      }
    } catch (error) {
      console.error('Error fetching suggestions:', error);
      setSuggestions([]);
    }
  }, 300);
  
  // Handle query input change
  const handleQueryChange = (event) => {
    const newQuery = event.target.value;
    setQuery(newQuery);
    fetchSuggestions(newQuery);
  };
  
  // Handle filter changes
  const handleFilterChange = (filterName, value) => {
    setFilters({
      ...filters,
      [filterName]: value
    });
  };
  
  // Build search params from query and filters
  const buildSearchParams = () => {
    const params = {
      query,
      max_results: 100,
      page: 1,
      page_size: 20
    };
    
    // Add filters
    if (filters.yearRange[0] !== 1950 || filters.yearRange[1] !== 2025) {
      params.year_range = filters.yearRange;
    }
    
    if (filters.sources.length > 0) {
      params.sources = filters.sources;
    }
    
    if (filters.articleTypes.length > 0) {
      params.article_types = filters.articleTypes;
    }
    
    if (filters.authors.length > 0) {
      params.authors = filters.authors;
    }
    
    if (filters.journals.length > 0) {
      params.journals = filters.journals;
    }
    
    if (filters.includeFreeFullText) {
      params.free_full_text = true;
    }
    
    if (!filters.includeTrials) {
      params.exclude_trials = true;
    }
    
    if (filters.excludeRetractions) {
      params.exclude_retractions = true;
    }
    
    if (filters.selectedLanguages.length > 0 && 
       !(filters.selectedLanguages.length === 1 && filters.selectedLanguages[0] === 'en')) {
      params.languages = filters.selectedLanguages;
    }
    
    return params;
  };
  
  // Execute search
  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }
    
    setIsSearching(true);
    setError('');
    saveRecentSearch(query);
    
    try {
      const params = buildSearchParams();
      const token = localStorage.getItem('token');
      const response = await axios.post('http://localhost:8000/api/medical/search', params, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.data.success) {
        setSearchResults(response.data.data);
        
        // If callback provided, send results up
        if (onSearchResults) {
          onSearchResults(response.data.data);
        }
      } else {
        setError('Search failed: ' + response.data.message);
      }
    } catch (error) {
      console.error('Error executing search:', error);
      setError('Error executing search: ' + error.message);
    } finally {
      setIsSearching(false);
    }
  };
  
  // Clear search
  const handleClearSearch = () => {
    setQuery('');
    setSearchResults(null);
    setSuggestions([]);
    setError('');
  };
  
  // Reset filters to defaults
  const handleResetFilters = () => {
    setFilters({
      yearRange: [1950, 2025],
      sources: [],
      articleTypes: [],
      authors: [],
      journals: [],
      includeFreeFullText: false,
      includeTrials: true,
      excludeRetractions: true,
      selectedLanguages: ['en']
    });
  };
  
  return (
    <Box sx={{ width: '100%' }}>
      {/* Main search field */}
      <Paper 
        elevation={3} 
        sx={{ 
          p: 2, 
          mb: 2, 
          borderRadius: 2,
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}
      >
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={9}>
            <Autocomplete
              freeSolo
              options={suggestions}
              inputValue={query}
              onInputChange={(event, newValue) => {
                setQuery(newValue);
                fetchSuggestions(newValue);
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Search Medical Literature"
                  variant="outlined"
                  fullWidth
                  placeholder="Search for medical conditions, treatments, or research topics..."
                  InputProps={{
                    ...params.InputProps,
                    startAdornment: (
                      <SearchIcon color="action" sx={{ ml: 1, mr: 0.5 }} />
                    )
                  }}
                />
              )}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="contained"
                color="primary"
                fullWidth
                size="large"
                disabled={!query.trim() || isSearching}
                onClick={handleSearch}
                startIcon={isSearching ? <CircularProgress size={18} color="inherit" /> : null}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </Button>
              <Button
                variant="outlined"
                color="secondary"
                size="large"
                onClick={() => setFiltersOpen(true)}
                sx={{ minWidth: 'auto', px: 1 }}
              >
                <FilterIcon />
              </Button>
            </Box>
          </Grid>
        </Grid>

        {error && (
          <Box sx={{ mt: 2 }}>
            <Typography color="error">{error}</Typography>
          </Box>
        )}
        
        {/* Search controls - chips area */}
        <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {/* Recent searches */}
          {recentSearches.length > 0 && !query && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2" color="textSecondary">
                Recent:
              </Typography>
              {recentSearches.slice(0, 5).map((search, index) => (
                <Chip
                  key={index}
                  label={search}
                  size="small"
                  onClick={() => {
                    setQuery(search);
                    handleSearch();
                  }}
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Box>
          )}
          
          {/* Active filters */}
          {(filters.sources.length > 0 || 
            filters.articleTypes.length > 0 || 
            filters.yearRange[0] !== 1950 || 
            filters.yearRange[1] !== 2025) && (
            <>
              <Divider orientation="vertical" flexItem />
              <Typography variant="body2" color="textSecondary">
                Filters:
              </Typography>
              
              {filters.yearRange[0] !== 1950 || filters.yearRange[1] !== 2025 ? (
                <Chip 
                  size="small"
                  label={`Years: ${filters.yearRange[0]}-${filters.yearRange[1]}`}
                  onDelete={() => handleFilterChange('yearRange', [1950, 2025])}
                />
              ) : null}
              
              {filters.sources.map((source, index) => (
                <Chip
                  key={index}
                  size="small"
                  label={`Source: ${source}`}
                  onDelete={() => handleFilterChange(
                    'sources', 
                    filters.sources.filter(s => s !== source)
                  )}
                />
              ))}
              
              {filters.articleTypes.map((type, index) => (
                <Chip
                  key={index}
                  size="small"
                  label={`Type: ${type}`}
                  onDelete={() => handleFilterChange(
                    'articleTypes', 
                    filters.articleTypes.filter(t => t !== type)
                  )}
                />
              ))}
              
              {(filters.sources.length > 0 || 
                filters.articleTypes.length > 0 || 
                filters.yearRange[0] !== 1950 || 
                filters.yearRange[1] !== 2025) && (
                <Chip
                  size="small"
                  label="Reset Filters"
                  color="secondary"
                  onClick={handleResetFilters}
                />
              )}
            </>
          )}
          
          {/* Clear search */}
          {query && (
            <>
              <Box sx={{ flex: 1 }} />
              <Chip
                icon={<ClearIcon />}
                label="Clear Search"
                size="small"
                onClick={handleClearSearch}
              />
            </>
          )}
        </Box>
      </Paper>
      
      {/* Filter drawer */}
      <Drawer
        anchor="right"
        open={filtersOpen}
        onClose={() => setFiltersOpen(false)}
        sx={{
          '& .MuiDrawer-paper': { 
            width: { xs: '100%', sm: 400 },
            p: 3
          },
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6">Search Filters</Typography>
          <IconButton onClick={() => setFiltersOpen(false)}>
            <CloseIcon />
          </IconButton>
        </Box>
        
        <Divider sx={{ mb: 3 }} />
        
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* Year range filter */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Publication Years: {filters.yearRange[0]} - {filters.yearRange[1]}
            </Typography>
            <Slider
              value={filters.yearRange}
              onChange={(e, newValue) => handleFilterChange('yearRange', newValue)}
              min={1950}
              max={2025}
              step={1}
              valueLabelDisplay="auto"
            />
          </Box>
          
          {/* Article type filter */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Article Types
            </Typography>
            <Autocomplete
              multiple
              options={filterOptions.articleTypes}
              value={filters.articleTypes}
              onChange={(e, newValue) => handleFilterChange('articleTypes', newValue)}
              renderInput={(params) => (
                <TextField 
                  {...params} 
                  variant="outlined" 
                  placeholder="Select article types..."
                />
              )}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    label={option}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Box>
          
          {/* Sources filter */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Sources
            </Typography>
            <Autocomplete
              multiple
              options={filterOptions.sources}
              value={filters.sources}
              onChange={(e, newValue) => handleFilterChange('sources', newValue)}
              renderInput={(params) => (
                <TextField 
                  {...params} 
                  variant="outlined" 
                  placeholder="Select sources..."
                />
              )}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    label={option}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Box>
          
          {/* Authors filter */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Authors
            </Typography>
            <Autocomplete
              multiple
              freeSolo
              options={[]}
              value={filters.authors}
              onChange={(e, newValue) => handleFilterChange('authors', newValue)}
              renderInput={(params) => (
                <TextField 
                  {...params} 
                  variant="outlined" 
                  placeholder="Enter author names..."
                />
              )}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    label={option}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Box>
          
          {/* Journals filter */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Journals
            </Typography>
            <Autocomplete
              multiple
              freeSolo
              options={[]}
              value={filters.journals}
              onChange={(e, newValue) => handleFilterChange('journals', newValue)}
              renderInput={(params) => (
                <TextField 
                  {...params} 
                  variant="outlined" 
                  placeholder="Enter journal names..."
                />
              )}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    label={option}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Box>
          
          {/* Switch filters */}
          <Box>
            <FormControlLabel
              control={
                <Switch
                  checked={filters.includeFreeFullText}
                  onChange={(e) => handleFilterChange('includeFreeFullText', e.target.checked)}
                />
              }
              label="Free Full Text Only"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={filters.includeTrials}
                  onChange={(e) => handleFilterChange('includeTrials', e.target.checked)}
                />
              }
              label="Include Clinical Trials"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={filters.excludeRetractions}
                  onChange={(e) => handleFilterChange('excludeRetractions', e.target.checked)}
                />
              }
              label="Exclude Retractions"
            />
          </Box>
          
          {/* Languages filter */}
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Languages
            </Typography>
            <Autocomplete
              multiple
              options={filterOptions.languages}
              value={filterOptions.languages.filter(lang => 
                filters.selectedLanguages.includes(lang.code)
              )}
              onChange={(e, newValue) => handleFilterChange(
                'selectedLanguages', 
                newValue.map(lang => lang.code)
              )}
              getOptionLabel={(option) => option.label}
              renderInput={(params) => (
                <TextField 
                  {...params} 
                  variant="outlined" 
                  placeholder="Select languages..."
                />
              )}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    label={option.label}
                    size="small"
                    {...getTagProps({ index })}
                  />
                ))
              }
            />
          </Box>
          
          <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
            <Button 
              variant="outlined" 
              onClick={handleResetFilters}
              startIcon={<ClearIcon />}
            >
              Reset Filters
            </Button>
            <Button 
              variant="contained" 
              color="primary"
              onClick={() => {
                setFiltersOpen(false);
                if (query) handleSearch();
              }}
              startIcon={<TuneIcon />}
            >
              Apply Filters
            </Button>
          </Box>
        </Box>
      </Drawer>

      {/* Search Results would go here */}
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
          
          {/* Results would be displayed here */}
          <Typography>
            {searchResults.total_count > 0 
              ? `Found ${searchResults.total_count} results for "${query}"`
              : `No results found for "${query}"`}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default AdvancedSearch;
