// Medical Research Synthesizer - Claim Visualization Component
// Interactive component for visualizing and analyzing extracted scientific claims

import React, { useState, useEffect } from 'react';
import {
  Box, Container, Typography, Paper, Button, Card, CardContent,
  Divider, Grid, CircularProgress, Alert, Tabs, Tab, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, LinearProgress, Tooltip, IconButton, TextField, MenuItem,
  FormControl, InputLabel, Select, Switch, FormControlLabel
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DownloadIcon from '@mui/icons-material/Download';
import InfoIcon from '@mui/icons-material/Info';
import { useAuth } from '../contexts/AuthContext';
import { callApi } from '../utils/api';

// Color mapping for claim types
const claimTypeColors = {
  finding: '#4caf50', // green
  methodology: '#2196f3', // blue
  interpretation: '#ff9800', // orange
  background: '#9e9e9e', // gray
  implication: '#673ab7', // purple
  other: '#795548' // brown
};

const ClaimVisualizer = () => {
  const { token } = useAuth();
  
  // State for extracted claims
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [text, setText] = useState('');
  const [processedText, setProcessedText] = useState('');
  const [claims, setClaims] = useState([]);
  const [documentId, setDocumentId] = useState('');
  const [tabValue, setTabValue] = useState(0);
  
  // State for filters
  const [filters, setFilters] = useState({
    claimTypes: [],
    minConfidence: 0.5,
    showHedged: true,
    showNegated: true
  });
  
  // State for selected claim
  const [selectedClaim, setSelectedClaim] = useState(null);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  // Extract claims from text
  const extractClaims = async () => {
    if (!text.trim()) {
      setError('Please enter some text to extract claims from.');
      return;
    }
    
    setLoading(true);
    setError('');
    
    try {
      const response = await callApi('post', '/api/extract-claims', {
        text,
        doc_id: documentId || undefined
      }, token);
      
      setClaims(response.claims || []);
      setProcessedText(text);
      
    } catch (err) {
      setError('Failed to extract claims: ' + (err.message || 'Unknown error'));
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  // Apply filters to claims
  const filteredClaims = claims.filter(claim => {
    // Filter by claim type
    if (filters.claimTypes.length > 0 && !filters.claimTypes.includes(claim.claim_type)) {
      return false;
    }
    
    // Filter by confidence
    if (claim.confidence < filters.minConfidence) {
      return false;
    }
    
    // Filter hedged claims
    if (!filters.showHedged && claim.hedge_level > 0.5) {
      return false;
    }
    
    // Filter negated claims
    if (!filters.showNegated && claim.negated) {
      return false;
    }
    
    return true;
  });
  
  // Handle claim selection
  const handleClaimSelect = (claim) => {
    setSelectedClaim(selectedClaim && selectedClaim.text === claim.text ? null : claim);
  };
  
  // Render a claim in the text
  const renderProcessedTextWithHighlights = () => {
    if (!processedText) return null;
    
    let lastIndex = 0;
    const textParts = [];
    
    // Sort claims by start position to ensure proper rendering
    const sortedClaims = [...filteredClaims].sort((a, b) => a.start_char - b.start_char);
    
    sortedClaims.forEach((claim, index) => {
      // Add text before the current claim
      if (claim.start_char > lastIndex) {
        textParts.push(
          <span key={`text-${index}`}>
            {processedText.substring(lastIndex, claim.start_char)}
          </span>
        );
      }
      
      // Add the highlighted claim
      const isSelected = selectedClaim && selectedClaim.text === claim.text;
      const backgroundColor = claim.claim_type ? claimTypeColors[claim.claim_type] || '#795548' : '#795548';
      
      textParts.push(
        <Tooltip
          key={`claim-${index}`}
          title={
            <React.Fragment>
              <Typography variant="subtitle2">
                {claim.claim_type.charAt(0).toUpperCase() + claim.claim_type.slice(1)} 
                {claim.negated && ' (Negated)'}
                {claim.hedge_level > 0.3 && ' (Hedged)'}
              </Typography>
              <Typography variant="body2">
                Confidence: {(claim.confidence * 100).toFixed(1)}%
              </Typography>
            </React.Fragment>
          }
        >
          <span
            onClick={() => handleClaimSelect(claim)}
            style={{
              backgroundColor: isSelected ? 'rgba(255, 255, 0, 0.4)' : `${backgroundColor}30`,
              border: `2px solid ${backgroundColor}`,
              padding: '2px 0',
              cursor: 'pointer',
              borderRadius: '3px',
              textDecoration: claim.negated ? 'line-through' : 'none',
              fontStyle: claim.hedge_level > 0.3 ? 'italic' : 'normal',
              fontWeight: isSelected ? 'bold' : 'normal'
            }}
          >
            {processedText.substring(claim.start_char, claim.end_char)}
          </span>
        </Tooltip>
      );
      
      lastIndex = claim.end_char;
    });
    
    // Add any remaining text
    if (lastIndex < processedText.length) {
      textParts.push(
        <span key="text-end">
          {processedText.substring(lastIndex)}
        </span>
      );
    }
    
    return textParts;
  };
  
  // Render the claim details panel
  const renderClaimDetails = () => {
    if (!selectedClaim) {
      return (
        <Paper elevation={2} sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="subtitle1" color="text.secondary">
            Select a claim to see details
          </Typography>
        </Paper>
      );
    }
    
    const claimType = selectedClaim.claim_type.charAt(0).toUpperCase() + selectedClaim.claim_type.slice(1);
    const claimColor = claimTypeColors[selectedClaim.claim_type] || '#795548';
    
    return (
      <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
        <Typography variant="h6" gutterBottom sx={{ color: claimColor }}>
          {claimType} Claim
          {selectedClaim.negated && <Chip size="small" label="Negated" sx={{ ml: 1, backgroundColor: '#f44336', color: 'white' }} />}
          {selectedClaim.hedge_level > 0.3 && (
            <Chip 
              size="small" 
              label={`Hedged ${(selectedClaim.hedge_level * 100).toFixed(0)}%`} 
              sx={{ ml: 1, backgroundColor: '#ff9800', color: 'white' }} 
            />
          )}
        </Typography>
        
        <Divider sx={{ mb: 2 }} />
        
        <Typography variant="body1" paragraph>
          "{selectedClaim.text}"
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="caption" component="div" color="text.secondary">
              Confidence
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Box sx={{ width: '100%', mr: 1 }}>
                <LinearProgress 
                  variant="determinate" 
                  value={selectedClaim.confidence * 100} 
                  color={selectedClaim.confidence > 0.8 ? 'success' : selectedClaim.confidence > 0.6 ? 'primary' : 'warning'} 
                />
              </Box>
              <Typography variant="body2" color="text.secondary">
                {(selectedClaim.confidence * 100).toFixed(0)}%
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={6}>
            <Typography variant="caption" component="div" color="text.secondary">
              Position
            </Typography>
            <Typography variant="body2">
              Chars {selectedClaim.start_char}-{selectedClaim.end_char}
            </Typography>
          </Grid>
          
          {selectedClaim.related_entities && selectedClaim.related_entities.length > 0 && (
            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ mt: 2 }}>
                Related Entities
              </Typography>
              
              <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Entity</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Position</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {selectedClaim.related_entities.map((entity, idx) => (
                      <TableRow key={idx}>
                        <TableCell>{entity.text}</TableCell>
                        <TableCell>{entity.label}</TableCell>
                        <TableCell>{entity.start_char}-{entity.end_char}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          )}
        </Grid>
      </Paper>
    );
  };
  
  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Scientific Claim Visualizer
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Input Text
        </Typography>
        
        <TextField
          fullWidth
          multiline
          rows={6}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter medical abstract or text to extract claims..."
          variant="outlined"
          sx={{ mb: 2 }}
        />
        
        <Box sx={{ display: 'flex', mb: 2 }}>
          <TextField
            label="Document ID (optional)"
            value={documentId}
            onChange={(e) => setDocumentId(e.target.value)}
            sx={{ mr: 2 }}
          />
          
          <Button
            variant="contained"
            color="primary"
            onClick={extractClaims}
            disabled={loading || !text.trim()}
          >
            {loading ? <CircularProgress size={24} /> : 'Extract Claims'}
          </Button>
        </Box>
      </Paper>
      
      {claims.length > 0 && (
        <Paper elevation={3} sx={{ p: 3 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="Highlighted Text" />
              <Tab label="Claims Table" />
              <Tab label="Statistics" />
            </Tabs>
          </Box>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle1">
              Found {filteredClaims.length} claims {filteredClaims.length !== claims.length && `(filtered from ${claims.length})`}
            </Typography>
            
            <Box>
              <FormControl sx={{ minWidth: 120, mr: 2 }}>
                <InputLabel>Claim Type</InputLabel>
                <Select
                  multiple
                  value={filters.claimTypes}
                  onChange={(e) => setFilters({ ...filters, claimTypes: e.target.value })}
                  label="Claim Type"
                  renderValue={(selected) => selected.map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(', ')}
                >
                  <MenuItem value="finding">Finding</MenuItem>
                  <MenuItem value="methodology">Methodology</MenuItem>
                  <MenuItem value="interpretation">Interpretation</MenuItem>
                  <MenuItem value="background">Background</MenuItem>
                  <MenuItem value="implication">Implication</MenuItem>
                  <MenuItem value="other">Other</MenuItem>
                </Select>
              </FormControl>
              
              <FormControl sx={{ minWidth: 120, mr: 2 }}>
                <InputLabel>Min. Confidence</InputLabel>
                <Select
                  value={filters.minConfidence}
                  onChange={(e) => setFilters({ ...filters, minConfidence: e.target.value })}
                  label="Min. Confidence"
                >
                  <MenuItem value={0.3}>30%</MenuItem>
                  <MenuItem value={0.5}>50%</MenuItem>
                  <MenuItem value={0.7}>70%</MenuItem>
                  <MenuItem value={0.9}>90%</MenuItem>
                </Select>
              </FormControl>
              
              <FormControlLabel
                control={
                  <Switch 
                    checked={filters.showHedged}
                    onChange={(e) => setFilters({ ...filters, showHedged: e.target.checked })}
                  />
                }
                label="Hedged"
              />
              
              <FormControlLabel
                control={
                  <Switch 
                    checked={filters.showNegated}
                    onChange={(e) => setFilters({ ...filters, showNegated: e.target.checked })}
                  />
                }
                label="Negated"
              />
            </Box>
          </Box>
          
          {tabValue === 0 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Paper elevation={2} sx={{ p: 3, minHeight: 300 }}>
                  <Typography variant="body1" component="div" sx={{ lineHeight: 1.8 }}>
                    {renderProcessedTextWithHighlights()}
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={4}>
                {renderClaimDetails()}
              </Grid>
            </Grid>
          )}
          
          {tabValue === 1 && (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Claim Text</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Confidence</TableCell>
                    <TableCell>Hedged</TableCell>
                    <TableCell>Negated</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredClaims.map((claim, index) => (
                    <TableRow 
                      key={index}
                      selected={selectedClaim && selectedClaim.text === claim.text}
                      sx={{ 
                        cursor: 'pointer',
                        '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.04)' }
                      }}
                      onClick={() => handleClaimSelect(claim)}
                    >
                      <TableCell>
                        <Typography 
                          sx={{ 
                            maxWidth: 300,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap'
                          }}
                        >
                          {claim.text}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={claim.claim_type.charAt(0).toUpperCase() + claim.claim_type.slice(1)} 
                          size="small"
                          sx={{ 
                            backgroundColor: `${claimTypeColors[claim.claim_type]}20`,
                            color: claimTypeColors[claim.claim_type],
                            borderColor: claimTypeColors[claim.claim_type],
                            borderWidth: 1,
                            borderStyle: 'solid'
                          }}
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Box sx={{ width: 60, mr: 1 }}>
                            <LinearProgress 
                              variant="determinate" 
                              value={claim.confidence * 100} 
                              color={
                                claim.confidence > 0.8 ? 'success' :
                                claim.confidence > 0.6 ? 'primary' : 'warning'
                              }
                            />
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            {(claim.confidence * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        {claim.hedge_level > 0.3 ? 
                          <Chip size="small" label={`${(claim.hedge_level * 100).toFixed(0)}%`} color="warning" /> : 
                          'No'
                        }
                      </TableCell>
                      <TableCell>
                        {claim.negated ? 
                          <Chip size="small" label="Yes" color="error" /> : 
                          'No'
                        }
                      </TableCell>
                      <TableCell>
                        <IconButton size="small" onClick={(e) => {
                          e.stopPropagation();
                          handleClaimSelect(claim);
                        }}>
                          <VisibilityIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          
          {tabValue === 2 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Claim Type Distribution
                    </Typography>
                    
                    <TableContainer component={Paper} variant="outlined">
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Type</TableCell>
                            <TableCell>Count</TableCell>
                            <TableCell>%</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(
                            claims.reduce((acc, claim) => {
                              acc[claim.claim_type] = (acc[claim.claim_type] || 0) + 1;
                              return acc;
                            }, {})
                          ).sort((a, b) => b[1] - a[1]).map(([type, count]) => (
                            <TableRow key={type}>
                              <TableCell>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Box 
                                    sx={{ 
                                      width: 12, 
                                      height: 12, 
                                      borderRadius: '50%', 
                                      backgroundColor: claimTypeColors[type] || '#795548',
                                      mr: 1 
                                    }} 
                                  />
                                  {type.charAt(0).toUpperCase() + type.slice(1)}
                                </Box>
                              </TableCell>
                              <TableCell>{count}</TableCell>
                              <TableCell>{((count / claims.length) * 100).toFixed(1)}%</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Claim Characteristics
                    </Typography>
                    
                    <Grid container spacing={2} sx={{ mb: 2 }}>
                      <Grid item xs={6}>
                        <Paper elevation={1} sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="body2" color="text.secondary">
                            Hedged Claims
                          </Typography>
                          <Typography variant="h4">
                            {claims.filter(c => c.hedge_level > 0.3).length}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            ({((claims.filter(c => c.hedge_level > 0.3).length / claims.length) * 100).toFixed(1)}%)
                          </Typography>
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Paper elevation={1} sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="body2" color="text.secondary">
                            Negated Claims
                          </Typography>
                          <Typography variant="h4">
                            {claims.filter(c => c.negated).length}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            ({((claims.filter(c => c.negated).length / claims.length) * 100).toFixed(1)}%)
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                    
                    <Paper elevation={1} sx={{ p: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Confidence Distribution
                      </Typography>
                      
                      <Box sx={{ mt: 2 }}>
                        {[
                          { range: '90-100%', color: 'success' },
                          { range: '70-89%', color: 'primary' },
                          { range: '50-69%', color: 'warning' },
                          { range: '0-49%', color: 'error' }
                        ].map((bracket) => {
                          let count;
                          if (bracket.range === '90-100%') {
                            count = claims.filter(c => c.confidence >= 0.9).length;
                          } else if (bracket.range === '70-89%') {
                            count = claims.filter(c => c.confidence >= 0.7 && c.confidence < 0.9).length;
                          } else if (bracket.range === '50-69%') {
                            count = claims.filter(c => c.confidence >= 0.5 && c.confidence < 0.7).length;
                          } else {
                            count = claims.filter(c => c.confidence < 0.5).length;
                          }
                          
                          const percentage = (count / claims.length) * 100;
                          
                          return (
                            <Box key={bracket.range} sx={{ mb: 1 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                <Typography variant="body2">{bracket.range}</Typography>
                                <Typography variant="body2">{count} ({percentage.toFixed(1)}%)</Typography>
                              </Box>
                              <LinearProgress 
                                variant="determinate" 
                                value={percentage} 
                                color={bracket.color} 
                                sx={{ height: 10, borderRadius: 5 }}
                              />
                            </Box>
                          );
                        })}
                      </Box>
                    </Paper>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}
        </Paper>
      )}
    </Container>
  );
};

export default ClaimVisualizer;