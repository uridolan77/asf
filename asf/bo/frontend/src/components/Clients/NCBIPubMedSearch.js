import React, { useState } from 'react';
import {
  Box, Paper, Typography, TextField, Button, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, IconButton, Tooltip, Alert, Pagination, Card, CardContent
} from '@mui/material';
import {
  Search as SearchIcon,
  OpenInNew as OpenInNewIcon,
  ContentCopy as ContentCopyIcon,
  Download as DownloadIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import apiService from '../../services/api';
import { useNotification } from '../../context/NotificationContext';

/**
 * NCBI PubMed Search Component
 * 
 * This component provides a search interface for PubMed using the NCBI API.
 */
const NCBIPubMedSearch = () => {
  const { showSuccess, showError } = useNotification();
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [page, setPage] = useState(1);
  const [selectedArticle, setSelectedArticle] = useState(null);
  
  // Handle search
  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!query.trim()) {
      showError('Please enter a search query');
      return;
    }
    
    setLoading(true);
    setResults(null);
    setSelectedArticle(null);
    
    try {
      const result = await apiService.clients.ncbi.searchPubMed(query, 20);
      
      if (result.success) {
        setResults(result.data);
        setPage(1);
        showSuccess(`Found ${result.data.total_results || 0} results`);
      } else {
        showError(`Search failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Error searching PubMed:', error);
      showError(`Error searching PubMed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle article selection
  const handleSelectArticle = async (pmid) => {
    setLoading(true);
    
    try {
      const result = await apiService.clients.ncbi.getArticle(pmid);
      
      if (result.success) {
        setSelectedArticle(result.data);
      } else {
        showError(`Failed to get article details: ${result.error}`);
      }
    } catch (error) {
      console.error('Error getting article details:', error);
      showError(`Error getting article details: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle page change
  const handlePageChange = (event, value) => {
    setPage(value);
  };
  
  // Handle copy citation
  const handleCopyCitation = () => {
    if (!selectedArticle) return;
    
    const citation = `${selectedArticle.authors.join(', ')}. (${selectedArticle.year}). ${selectedArticle.title}. ${selectedArticle.journal}, ${selectedArticle.volume}(${selectedArticle.issue}), ${selectedArticle.pages}. doi: ${selectedArticle.doi}`;
    
    navigator.clipboard.writeText(citation)
      .then(() => showSuccess('Citation copied to clipboard'))
      .catch(() => showError('Failed to copy citation'));
  };
  
  // Render article details
  const renderArticleDetails = () => {
    if (!selectedArticle) return null;
    
    return (
      <Card variant="outlined" sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {selectedArticle.title}
          </Typography>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
            {selectedArticle.authors.map((author, index) => (
              <Chip key={index} label={author} size="small" />
            ))}
          </Box>
          
          <Typography variant="body2" color="text.secondary" gutterBottom>
            <strong>Journal:</strong> {selectedArticle.journal}, {selectedArticle.year}
            {selectedArticle.volume && `, Volume ${selectedArticle.volume}`}
            {selectedArticle.issue && `, Issue ${selectedArticle.issue}`}
            {selectedArticle.pages && `, Pages ${selectedArticle.pages}`}
          </Typography>
          
          {selectedArticle.doi && (
            <Typography variant="body2" color="text.secondary" gutterBottom>
              <strong>DOI:</strong> {selectedArticle.doi}
            </Typography>
          )}
          
          {selectedArticle.pmid && (
            <Typography variant="body2" color="text.secondary" gutterBottom>
              <strong>PMID:</strong> {selectedArticle.pmid}
            </Typography>
          )}
          
          {selectedArticle.abstract && (
            <>
              <Typography variant="subtitle1" sx={{ mt: 2, mb: 1 }}>
                Abstract
              </Typography>
              <Typography variant="body2" paragraph>
                {selectedArticle.abstract}
              </Typography>
            </>
          )}
          
          <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
            <Button
              size="small"
              startIcon={<OpenInNewIcon />}
              onClick={() => window.open(`https://pubmed.ncbi.nlm.nih.gov/${selectedArticle.pmid}/`, '_blank')}
            >
              View on PubMed
            </Button>
            <Button
              size="small"
              startIcon={<ContentCopyIcon />}
              onClick={handleCopyCitation}
            >
              Copy Citation
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  };
  
  // Render search results
  const renderSearchResults = () => {
    if (!results) return null;
    
    const { articles, total_results } = results;
    
    if (!articles || articles.length === 0) {
      return (
        <Alert severity="info" sx={{ mt: 3 }}>
          No results found for "{query}". Try a different search term.
        </Alert>
      );
    }
    
    // Calculate pagination
    const itemsPerPage = 10;
    const startIndex = (page - 1) * itemsPerPage;
    const endIndex = Math.min(startIndex + itemsPerPage, articles.length);
    const displayedArticles = articles.slice(startIndex, endIndex);
    const totalPages = Math.ceil(articles.length / itemsPerPage);
    
    return (
      <Box sx={{ mt: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Found {total_results} results for "{query}"
        </Typography>
        
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Title</TableCell>
                <TableCell>Authors</TableCell>
                <TableCell>Journal</TableCell>
                <TableCell>Year</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {displayedArticles.map((article) => (
                <TableRow key={article.pmid} hover>
                  <TableCell>{article.title}</TableCell>
                  <TableCell>{article.authors.slice(0, 3).join(', ')}{article.authors.length > 3 ? ', et al.' : ''}</TableCell>
                  <TableCell>{article.journal}</TableCell>
                  <TableCell>{article.year}</TableCell>
                  <TableCell>
                    <Tooltip title="View Details">
                      <IconButton size="small" onClick={() => handleSelectArticle(article.pmid)}>
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Open in PubMed">
                      <IconButton 
                        size="small" 
                        onClick={() => window.open(`https://pubmed.ncbi.nlm.nih.gov/${article.pmid}/`, '_blank')}
                      >
                        <OpenInNewIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        {totalPages > 1 && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Pagination 
              count={totalPages} 
              page={page} 
              onChange={handlePageChange} 
              color="primary" 
            />
          </Box>
        )}
      </Box>
    );
  };
  
  return (
    <Box>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          PubMed Search
        </Typography>
        
        <form onSubmit={handleSearch}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              fullWidth
              label="Search PubMed"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., covid-19 treatment efficacy"
              variant="outlined"
              size="small"
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <SearchIcon />}
            >
              Search
            </Button>
          </Box>
        </form>
        
        {loading && !results && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
            <CircularProgress />
          </Box>
        )}
        
        {renderSearchResults()}
        {renderArticleDetails()}
      </Paper>
    </Box>
  );
};

export default NCBIPubMedSearch;
