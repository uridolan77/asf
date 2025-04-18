// Medical Research Synthesizer - Web Interface Components
// Key React components for the frontend application

// src/components/QueryBuilder.jsx
import React, { useState, useEffect } from 'react';
import {
  Box, Container, Typography, Paper, Button, TextField, MenuItem, Select,
  FormControl, InputLabel, Chip, Divider, Grid, IconButton, FormGroup, 
  FormControlLabel, Checkbox, CircularProgress, Alert, Autocomplete
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import BuildIcon from '@mui/icons-material/Build';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { callApi } from '../utils/api';

const QueryBuilder = () => {
  const { token } = useAuth();
  const navigate = useNavigate();
  
  // State for query builder
  const [queryType, setQueryType] = useState('pico');
  const [useMesh, setUseMesh] = useState(true);
  const [conditions, setConditions] = useState([{ name: '', synonyms: [], subtypes: [] }]);
  const [interventions, setInterventions] = useState([{ name: '', type: 'treatment', alternatives: [], specific_forms: [] }]);
  const [outcomes, setOutcomes] = useState([{ name: '', type: 'efficacy', synonyms: [], related_metrics: [] }]);
  const [studyDesigns, setStudyDesigns] = useState([]);
  const [yearsBack, setYearsBack] = useState(5);
  const [englishOnly, setEnglishOnly] = useState(true);
  const [highQualityOnly, setHighQualityOnly] = useState(false);
  const [humansOnly, setHumansOnly] = useState(true);
  
  // State for templates
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [generatedQuery, setGeneratedQuery] = useState('');
  
  // Fetch templates on mount
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const response = await callApi('get', '/api/query/templates', null, token);
        setTemplates(response);
      } catch (err) {
        setError('Failed to load query templates');
        console.error(err);
      }
    };
    
    fetchTemplates();
  }, [token]);
  
  // Load template data
  const handleTemplateChange = async (templateId) => {
    if (!templateId) return;
    
    setSelectedTemplate(templateId);
    setLoading(true);
    
    try {
      const response = await callApi('get', `/api/query/template/${templateId}`, null, token);
      
      // Parse the components
      const components = response.components;
      
      if (components.conditions) {
        setConditions(components.conditions.map(name => ({ name, synonyms: [], subtypes: [] })));
      }
      
      if (components.interventions) {
        setInterventions(components.interventions.map(name => ({ 
          name, type: 'treatment', alternatives: [], specific_forms: [] 
        })));
      }
      
      if (components.outcomes) {
        setOutcomes(components.outcomes.map(name => ({ 
          name, type: 'efficacy', synonyms: [], related_metrics: [] 
        })));
      }
      
      if (components.study_designs) {
        setStudyDesigns(components.study_designs.map(type => ({ 
          type, related_designs: [], characteristics: [] 
        })));
      }
      
      setGeneratedQuery(response.query);
      
    } catch (err) {
      setError('Failed to load template data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  // Add/Remove functions for array fields
  const addCondition = () => {
    setConditions([...conditions, { name: '', synonyms: [], subtypes: [] }]);
  };
  
  const removeCondition = (index) => {
    const newConditions = [...conditions];
    newConditions.splice(index, 1);
    setConditions(newConditions);
  };
  
  const addIntervention = () => {
    setInterventions([...interventions, { name: '', type: 'treatment', alternatives: [], specific_forms: [] }]);
  };
  
  const removeIntervention = (index) => {
    const newInterventions = [...interventions];
    newInterventions.splice(index, 1);
    setInterventions(newInterventions);
  };
  
  const addOutcome = () => {
    setOutcomes([...outcomes, { name: '', type: 'efficacy', synonyms: [], related_metrics: [] }]);
  };
  
  const removeOutcome = (index) => {
    const newOutcomes = [...outcomes];
    newOutcomes.splice(index, 1);
    setOutcomes(newOutcomes);
  };
  
  const addStudyDesign = () => {
    setStudyDesigns([...studyDesigns, { type: '', related_designs: [], characteristics: [] }]);
  };
  
  const removeStudyDesign = (index) => {
    const newStudyDesigns = [...studyDesigns];
    newStudyDesigns.splice(index, 1);
    setStudyDesigns(newStudyDesigns);
  };
  
  // Handle form submission
  const handleSubmit = async () => {
    setLoading(true);
    setError('');
    
    try {
      const queryData = {
        query_type: queryType,
        use_mesh: useMesh,
        conditions,
        interventions,
        outcomes,
        study_designs: studyDesigns,
        years: yearsBack,
        filters: {
          language: englishOnly ? 'english_only' : null,
          quality: highQualityOnly ? 'high_quality_only' : null,
          humans_only: humansOnly
        }
      };
      
      const response = await callApi('post', '/api/query/create', queryData, token);
      
      setGeneratedQuery(response.query);
      
      // Redirect to search page with the query ID
      navigate(`/search?queryId=${response.query_id}`);
      
    } catch (err) {
      setError('Failed to create query: ' + (err.message || 'Unknown error'));
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Medical Query Builder
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Select a Template (Optional)
        </Typography>
        
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel id="template-select-label">Template</InputLabel>
          <Select
            labelId="template-select-label"
            value={selectedTemplate}
            onChange={(e) => handleTemplateChange(e.target.value)}
            label="Template"
          >
            <MenuItem value="">
              <em>None (Build from scratch)</em>
            </MenuItem>
            {templates.map((template) => (
              <MenuItem key={template.id} value={template.id}>
                {template.name} - {template.description}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        
        <Divider sx={{ mb: 3 }} />
        
        <Typography variant="h6" gutterBottom>
          Query Settings
        </Typography>
        
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel id="query-type-label">Query Type</InputLabel>
              <Select
                labelId="query-type-label"
                value={queryType}
                onChange={(e) => setQueryType(e.target.value)}
                label="Query Type"
              >
                <MenuItem value="pico">PICO (Population, Intervention, Comparison, Outcome)</MenuItem>
                <MenuItem value="simple">Simple Query</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={6}>
            <FormGroup>
              <FormControlLabel 
                control={<Checkbox checked={useMesh} onChange={(e) => setUseMesh(e.target.checked)} />}
                label="Use MeSH Terms (Recommended)"
              />
            </FormGroup>
          </Grid>
        </Grid>
        
        <Divider sx={{ mb: 3 }} />
        
        {/* Conditions Section */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Medical Conditions
          </Typography>
          
          {conditions.map((condition, index) => (
            <Paper elevation={2} key={index} sx={{ p: 2, mb: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={10}>
                  <TextField
                    fullWidth
                    label="Condition Name"
                    value={condition.name}
                    onChange={(e) => {
                      const newConditions = [...conditions];
                      newConditions[index].name = e.target.value;
                      setConditions(newConditions);
                    }}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={2} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                  <IconButton 
                    color="error" 
                    onClick={() => removeCondition(index)}
                    disabled={conditions.length === 1}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Grid>
                
                <Grid item xs={12}>
                  <Autocomplete
                    multiple
                    freeSolo
                    options={[]}
                    value={condition.synonyms}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => (
                        <Chip label={option} {...getTagProps({ index })} />
                      ))
                    }
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Synonyms (e.g., alternative names)"
                        placeholder="Add synonym"
                      />
                    )}
                    onChange={(e, newValue) => {
                      const newConditions = [...conditions];
                      newConditions[index].synonyms = newValue;
                      setConditions(newConditions);
                    }}
                  />
                </Grid>
              </Grid>
            </Paper>
          ))}
          
          <Button
            startIcon={<AddIcon />}
            onClick={addCondition}
            variant="outlined"
            sx={{ mt: 1 }}
          >
            Add Condition
          </Button>
        </Box>
        
        {/* Interventions Section */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Interventions or Treatments
          </Typography>
          
          {interventions.map((intervention, index) => (
            <Paper elevation={2} key={index} sx={{ p: 2, mb: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={8}>
                  <TextField
                    fullWidth
                    label="Intervention Name"
                    value={intervention.name}
                    onChange={(e) => {
                      const newInterventions = [...interventions];
                      newInterventions[index].name = e.target.value;
                      setInterventions(newInterventions);
                    }}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={2}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Type</InputLabel>
                    <Select
                      value={intervention.type}
                      label="Type"
                      onChange={(e) => {
                        const newInterventions = [...interventions];
                        newInterventions[index].type = e.target.value;
                        setInterventions(newInterventions);
                      }}
                    >
                      <MenuItem value="treatment">Treatment</MenuItem>
                      <MenuItem value="diagnostic">Diagnostic</MenuItem>
                      <MenuItem value="prevention">Prevention</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={2} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                  <IconButton 
                    color="error" 
                    onClick={() => removeIntervention(index)}
                    disabled={interventions.length === 1}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Grid>
                
                <Grid item xs={12}>
                  <Autocomplete
                    multiple
                    freeSolo
                    options={[]}
                    value={intervention.alternatives}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => (
                        <Chip label={option} {...getTagProps({ index })} />
                      ))
                    }
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Alternative Interventions"
                        placeholder="Add alternative"
                      />
                    )}
                    onChange={(e, newValue) => {
                      const newInterventions = [...interventions];
                      newInterventions[index].alternatives = newValue;
                      setInterventions(newInterventions);
                    }}
                  />
                </Grid>
              </Grid>
            </Paper>
          ))}
          
          <Button
            startIcon={<AddIcon />}
            onClick={addIntervention}
            variant="outlined"
            sx={{ mt: 1 }}
          >
            Add Intervention
          </Button>
        </Box>
        
        {/* Outcomes Section */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Outcomes or Metrics
          </Typography>
          
          {outcomes.map((outcome, index) => (
            <Paper elevation={2} key={index} sx={{ p: 2, mb: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={8}>
                  <TextField
                    fullWidth
                    label="Outcome Metric"
                    value={outcome.name}
                    onChange={(e) => {
                      const newOutcomes = [...outcomes];
                      newOutcomes[index].name = e.target.value;
                      setOutcomes(newOutcomes);
                    }}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={2}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Type</InputLabel>
                    <Select
                      value={outcome.type}
                      label="Type"
                      onChange={(e) => {
                        const newOutcomes = [...outcomes];
                        newOutcomes[index].type = e.target.value;
                        setOutcomes(newOutcomes);
                      }}
                    >
                      <MenuItem value="efficacy">Efficacy</MenuItem>
                      <MenuItem value="safety">Safety</MenuItem>
                      <MenuItem value="cost">Cost</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={2} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                  <IconButton 
                    color="error" 
                    onClick={() => removeOutcome(index)}
                    disabled={outcomes.length === 1}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Grid>
                
                <Grid item xs={12}>
                  <Autocomplete
                    multiple
                    freeSolo
                    options={[]}
                    value={outcome.synonyms}
                    renderTags={(value, getTagProps) =>
                      value.map((option, index) => (
                        <Chip label={option} {...getTagProps({ index })} />
                      ))
                    }
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        label="Synonyms"
                        placeholder="Add synonym"
                      />
                    )}
                    onChange={(e, newValue) => {
                      const newOutcomes = [...outcomes];
                      newOutcomes[index].synonyms = newValue;
                      setOutcomes(newOutcomes);
                    }}
                  />
                </Grid>
              </Grid>
            </Paper>
          ))}
          
          <Button
            startIcon={<AddIcon />}
            onClick={addOutcome}
            variant="outlined"
            sx={{ mt: 1 }}
          >
            Add Outcome
          </Button>
        </Box>
        
        {/* Study Design Section */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Study Designs (Optional)
          </Typography>
          
          {studyDesigns.map((design, index) => (
            <Paper elevation={2} key={index} sx={{ p: 2, mb: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={10}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Study Design</InputLabel>
                    <Select
                      value={design.type}
                      label="Study Design"
                      onChange={(e) => {
                        const newDesigns = [...studyDesigns];
                        newDesigns[index].type = e.target.value;
                        setStudyDesigns(newDesigns);
                      }}
                    >
                      <MenuItem value="randomized controlled trial">Randomized Controlled Trial</MenuItem>
                      <MenuItem value="systematic review">Systematic Review</MenuItem>
                      <MenuItem value="meta-analysis">Meta-Analysis</MenuItem>
                      <MenuItem value="cohort study">Cohort Study</MenuItem>
                      <MenuItem value="case-control study">Case-Control Study</MenuItem>
                      <MenuItem value="observational study">Observational Study</MenuItem>
                      <MenuItem value="clinical trial">Clinical Trial</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={2} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                  <IconButton 
                    color="error" 
                    onClick={() => removeStudyDesign(index)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Grid>
              </Grid>
            </Paper>
          ))}
          
          <Button
            startIcon={<AddIcon />}
            onClick={addStudyDesign}
            variant="outlined"
            sx={{ mt: 1 }}
          >
            Add Study Design
          </Button>
        </Box>
        
        {/* Additional Filters */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Additional Filters
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Years Back"
                type="number"
                value={yearsBack}
                onChange={(e) => setYearsBack(parseInt(e.target.value) || 5)}
                inputProps={{ min: 1, max: 100 }}
                helperText="Limit to publications within this many years"
              />
            </Grid>
            <Grid item xs={12}>
              <FormGroup>
                <FormControlLabel 
                  control={<Checkbox checked={englishOnly} onChange={(e) => setEnglishOnly(e.target.checked)} />}
                  label="English Only"
                />
                <FormControlLabel 
                  control={<Checkbox checked={highQualityOnly} onChange={(e) => setHighQualityOnly(e.target.checked)} />}
                  label="High Quality Evidence Only (RCTs, Meta-Analyses, Systematic Reviews)"
                />
                <FormControlLabel 
                  control={<Checkbox checked={humansOnly} onChange={(e) => setHumansOnly(e.target.checked)} />}
                  label="Human Studies Only"
                />
              </FormGroup>
            </Grid>
          </Grid>
        </Box>
        
        {/* Generated Query */}
        {generatedQuery && (
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Generated Query
            </Typography>
            <Paper elevation={2} sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
              <Typography component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {generatedQuery}
              </Typography>
            </Paper>
          </Box>
        )}
        
        {/* Submit Button */}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 4 }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            startIcon={<BuildIcon />}
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Build Query & Search'}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default QueryBuilder;

// src/components/SearchResults.jsx
import React, { useState, useEffect } from 'react';
import {
  Box, Container, Typography, Paper, Chip, Button, Card, CardContent, 
  CardActions, Divider, Grid, CircularProgress, Alert, IconButton,
  Menu, MenuItem, Dialog, DialogTitle, DialogContent, DialogActions,
  List, ListItem, ListItemText, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DownloadIcon from '@mui/icons-material/Download';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import { useSearchParams } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { callApi } from '../utils/api';

const SearchResults = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const queryId = searchParams.get('queryId');
  const rawQuery = searchParams.get('query');
  
  // State
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState([]);
  const [query, setQuery] = useState('');
  const [resultId, setResultId] = useState('');
  const [totalResults, setTotalResults] = useState(0);
  const [contradictions, setContradictions] = useState([]);
  const [showContradictions, setShowContradictions] = useState(false);
  
  // Export menu
  const [exportAnchorEl, setExportAnchorEl] = useState(null);
  const exportOpen = Boolean(exportAnchorEl);
  
  useEffect(() => {
    const fetchResults = async () => {
      if (!queryId && !rawQuery) return;
      
      setLoading(true);
      setError('');
      
      try {
        const payload = queryId 
          ? { query_id: queryId, max_results: 20 }
          : { query: rawQuery, max_results: 20 };
        
        const response = await callApi('post', '/api/search/execute', payload, token);
        
        setResults(response.results);
        setQuery(response.query);
        setResultId(response.result_id);
        setTotalResults(response.total_results);
        
      } catch (err) {
        setError('Failed to fetch results: ' + (err.message || 'Unknown error'));
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchResults();
  }, [queryId, rawQuery, token]);
  
  const handleExportClick = (event) => {
    setExportAnchorEl(event.currentTarget);
  };
  
  const handleExportClose = () => {
    setExportAnchorEl(null);
  };
  
  const handleExport = async (format) => {
    handleExportClose();
    
    if (!resultId) {
      setError('No results to export');
      return;
    }
    
    try {
      // For PDF, JSON, CSV, Excel - we'll call the API and download the file
      const payload = { result_id: resultId };
      
      // These endpoints will return file downloads
      window.location.href = `/api/export/${format}`;
      
      // In a real application, you would handle the download properly
      // For now, we just show a message
      alert(`Exporting to ${format.toUpperCase()}... This would download a file in a real application.`);
      
    } catch (err) {
      setError(`Failed to export as ${format.toUpperCase()}: ` + (err.message || 'Unknown error'));
      console.error(err);
    }
  };
  
  const analyzeContradictions = async () => {
    if (!resultId) {
      setError('No results to analyze');
      return;
    }
    
    setAnalyzing(true);
    setError('');
    
    try {
      const payload = { result_id: resultId };
      const response = await callApi('post', '/api/analysis/contradictions', payload, token);
      
      setContradictions(response.contradictions || []);
      setShowContradictions(true);
      
    } catch (err) {
      setError('Failed to analyze contradictions: ' + (err.message || 'Unknown error'));
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };
  
  const renderAuthorityBadge = (score) => {
    let color = 'default';
    let label = 'Unknown';
    
    if (score >= 80) {
      color = 'success';
      label = 'High';
    } else if (score >= 60) {
      color = 'primary';
      label = 'Good';
    } else if (score >= 40) {
      color = 'warning';
      label = 'Moderate';
    } else if (score >= 0) {
      color = 'error';
      label = 'Low';
    }
    
    return <Chip size="small" color={color} label={`Authority: ${label}`} />;
  };
  
  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Search Results
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Query
        </Typography>
        <Paper elevation={1} sx={{ p: 2, backgroundColor: '#f5f5f5', mb: 2 }}>
          <Typography component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {query}
          </Typography>
        </Paper>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="subtitle1">
            Found {totalResults} results
          </Typography>
          
          <Box>
            <Button
              variant="outlined"
              color="primary"
              startIcon={<AnalyticsIcon />}
              onClick={analyzeContradictions}
              disabled={analyzing || loading || results.length === 0}
              sx={{ mr: 1 }}
            >
              {analyzing ? <CircularProgress size={24} /> : 'Analyze Contradictions'}
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              onClick={handleExportClick}
              disabled={loading || results.length === 0}
            >
              Export
            </Button>
            
            <Menu
              anchorEl={exportAnchorEl}
              open={exportOpen}
              onClose={handleExportClose}
            >
              <MenuItem onClick={() => handleExport('pdf')}>PDF</MenuItem>
              <MenuItem onClick={() => handleExport('excel')}>Excel</MenuItem>
              <MenuItem onClick={() => handleExport('csv')}>CSV</MenuItem>
              <MenuItem onClick={() => handleExport('json')}>JSON</MenuItem>
              <MenuItem onClick={() => handleExport('markdown')}>Markdown</MenuItem>
            </Menu>
          </Box>
        </Box>
        
        <Divider sx={{ mb: 3 }} />
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        ) : results.length === 0 ? (
          <Alert severity="info">No results found.</Alert>
        ) : (
          <>
            {/* Results List */}
            {results.map((article, index) => (
              <Card key={index} sx={{ mb: 3 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <Typography variant="h6" component="h2" gutterBottom>
                      {article.title}
                    </Typography>
                    
                    <Box>
                      {renderAuthorityBadge(article.authority_score)}
                    </Box>
                  </Box>
                  
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    {article.authors ? article.authors.join(', ') : 'Unknown authors'}
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {article.journal || 'Unknown journal'} | {article.human_date || article.publication_date || 'Unknown date'}
                  </Typography>
                  
                  <Box sx={{ mt: 2 }}>
                    <Grid container spacing={2}>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" component="div" color="text.secondary">
                          Impact Factor
                        </Typography>
                        <Typography variant="body2">
                          {article.impact_factor ? article.impact_factor.toFixed(2) : 'Unknown'}
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" component="div" color="text.secondary">
                          Citations
                        </Typography>
                        <Typography variant="body2">
                          {article.citation_count !== undefined ? article.citation_count : 'Unknown'}
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" component="div" color="text.secondary">
                          Publication Type
                        </Typography>
                        <Typography variant="body2">
                          {article.publication_types && article.publication_types.length > 0 
                            ? article.publication_types.join(', ') 
                            : 'Journal Article'}
                        </Typography>
                      </Grid>
                      <Grid item xs={6} md={3}>
                        <Typography variant="caption" component="div" color="text.secondary">
                          PMID
                        </Typography>
                        <Typography variant="body2">
                          {article.pmid || 'Unknown'}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="body1" component="div" sx={{ mt: 2 }}>
                    {article.abstract 
                      ? article.abstract.length > 400 
                        ? article.abstract.substring(0, 400) + '...' 
                        : article.abstract
                      : 'No abstract available'
                    }
                  </Typography>
                </CardContent>
                <CardActions sx={{ justifyContent: 'flex-end' }}>
                  <Button 
                    size="small" 
                    component="a"
                    href={`https://pubmed.ncbi.nlm.nih.gov/${article.pmid}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    View on PubMed
                  </Button>
                </CardActions>
              </Card>
            ))}
          </>
        )}
      </Paper>
      
      {/* Contradictions Dialog */}
      <Dialog 
        open={showContradictions} 
        onClose={() => setShowContradictions(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Contradictory Findings Analysis</DialogTitle>
        <DialogContent>
          {contradictions.length === 0 ? (
            <Alert severity="info" sx={{ mt: 2 }}>
              No contradictions found in the analyzed articles.
            </Alert>
          ) : (
            <List>
              {contradictions.map((contradiction, index) => (
                <ListItem key={index} disablePadding>
                  <Accordion sx={{ width: '100%', mb: 1 }}>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">
                        Contradiction {index + 1} 
                        <Chip 
                          size="small" 
                          color={contradiction.confidence === 'high' ? 'error' : 'warning'}
                          label={`${contradiction.confidence} confidence`}
                          sx={{ ml: 1 }}
                        />
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle2" gutterBottom>
                            Publication 1 
                            {renderAuthorityBadge(contradiction.publication1.authority_score)}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Title:</strong> {contradiction.publication1.title}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>PMID:</strong> {contradiction.publication1.pmid}
                          </Typography>
                          <Typography variant="body2">
                            <strong>Abstract:</strong> {contradiction.publication1.abstract_snippet}
                          </Typography>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle2" gutterBottom>
                            Publication 2
                            {renderAuthorityBadge(contradiction.publication2.authority_score)}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>Title:</strong> {contradiction.publication2.title}
                          </Typography>
                          <Typography variant="body2" gutterBottom>
                            <strong>PMID:</strong> {contradiction.publication2.pmid}
                          </Typography>
                          <Typography variant="body2">
                            <strong>Abstract:</strong> {contradiction.publication2.abstract_snippet}
                          </Typography>
                        </Grid>
                        <Grid item xs={12}>
                          <Paper elevation={1} sx={{ p: 2, backgroundColor: '#f8f8f8', mt: 1 }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Authority Analysis:
                            </Typography>
                            <Typography variant="body2">
                              {contradiction.authority_comparison.higher_authority === 'publication1' && (
                                <>Publication 1 has higher authority (+{Math.abs(contradiction.authority_comparison.authority_difference).toFixed(1)} points)</>
                              )}
                              {contradiction.authority_comparison.higher_authority === 'publication2' && (
                                <>Publication 2 has higher authority (+{Math.abs(contradiction.authority_comparison.authority_difference).toFixed(1)} points)</>
                              )}
                              {contradiction.authority_comparison.higher_authority === 'equal' && (
                                <>Both publications have similar authority</>
                              )}
                            </Typography>
                          </Paper>
                        </Grid>
                      </Grid>
                    </AccordionDetails>
                  </Accordion>
                </ListItem>
              ))}
            </List>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowContradictions(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default SearchResults;

// src/components/KnowledgeBase.jsx
import React, { useState, useEffect } from 'react';
import {
  Box, Container, Typography, Paper, Button, TextField, MenuItem, Select,
  FormControl, InputLabel, CircularProgress, Alert, Table, TableBody,
  TableCell, TableContainer, TableHead, TableRow, Dialog, DialogTitle,
  DialogContent, DialogActions, Grid, Card, CardContent, CardHeader, IconButton,
  Tooltip, List, ListItem, ListItemText
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import AddIcon from '@mui/icons-material/Add';
import DownloadIcon from '@mui/icons-material/Download';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { useAuth } from '../contexts/AuthContext';
import { callApi } from '../utils/api';
import SearchResults from './SearchResults';

const KnowledgeBase = () => {
  const { token } = useAuth();
  
  // State for knowledge bases
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [selectedKb, setSelectedKb] = useState(null);
  const [kbArticles, setKbArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [error, setError] = useState('');
  
  // State for creating a new KB
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newKbName, setNewKbName] = useState('');
  const [newKbQuery, setNewKbQuery] = useState('');
  const [newKbSchedule, setNewKbSchedule] = useState('weekly');
  const [creating, setCreating] = useState(false);
  
  // State for viewing KB
  const [showViewDialog, setShowViewDialog] = useState(false);
  
  // Fetch knowledge bases on mount
  useEffect(() => {
    const fetchKnowledgeBases = async () => {
      setLoading(true);
      setError('');
      
      try {
        const response = await callApi('get', '/api/kb/list', null, token);
        setKnowledgeBases(response);
      } catch (err) {
        setError('Failed to load knowledge bases: ' + (err.message || 'Unknown error'));
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchKnowledgeBases();
  }, [token]);
  
  const handleSelectKb = async (kb) => {
    setSelectedKb(kb);
    setKbArticles([]);
    setLoading(true);
    setError('');
    
    try {
      const response = await callApi('get', `/api/kb/${kb.name}`, null, token);
      setKbArticles(response.articles);
    } catch (err) {
      setError('Failed to load knowledge base: ' + (err.message || 'Unknown error'));
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const handleUpdateKb = async () => {
    if (!selectedKb) return;
    
    setUpdating(true);
    setError('');
    
    try {
      const response = await callApi('post', `/api/kb/${selectedKb.name}/update`, null, token);
      
      // Refresh the KB articles
      await handleSelectKb(selectedKb);
      
      // Show success message
      alert(`Knowledge base updated: ${response.new_count} new articles added.`);
      
    } catch (err) {
      setError('Failed to update knowledge base: ' + (err.message || 'Unknown error'));
      console.error(err);
    } finally {
      setUpdating(false);
    }
  };
  
  const handleCreateKb = async () => {
    if (!newKbName || !newKbQuery) {
      setError('Name and query are required');
      return;
    }
    
    setCreating(true);
    setError('');
    
    try {
      const payload = {
        name: newKbName,
        query: newKbQuery,
        schedule: newKbSchedule,
        max_results: 100
      };
      
      const response = await callApi('post', '/api/kb/create', payload, token);
      
      // Add the new KB to the list
      setKnowledgeBases([...knowledgeBases, response]);
      
      // Select the new KB
      handleSelectKb(response);
      
      // Close the dialog
      setShowCreateDialog(false);
      
      // Reset form
      setNewKbName('');
      setNewKbQuery('');
      setNewKbSchedule('weekly');
      
    } catch (err) {
      setError('Failed to create knowledge base: ' + (err.message || 'Unknown error'));
      console.error(err);
    } finally {
      setCreating(false);
    }
  };
  
  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Knowledge Bases
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">Available Knowledge Bases</Typography>
              <Button
                variant="contained"
                color="primary"
                startIcon={<AddIcon />}
                onClick={() => setShowCreateDialog(true)}
              >
                Create New
              </Button>
            </Box>
            
            {loading && !selectedKb ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
              </Box>
            ) : knowledgeBases.length === 0 ? (
              <Alert severity="info">No knowledge bases found. Create one to get started.</Alert>
            ) : (
              <List sx={{ bgcolor: 'background.paper' }}>
                {knowledgeBases.map((kb) => (
                  <ListItem
                    key={kb.kb_id}
                    button
                    selected={selectedKb && selectedKb.kb_id === kb.kb_id}
                    onClick={() => handleSelectKb(kb)}
                  >
                    <ListItemText
                      primary={kb.name}
                      secondary={`Created: ${new Date(kb.created_date).toLocaleDateString()}`}
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ p: 3 }}>
            {selectedKb ? (
              <>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6">{selectedKb.name}</Typography>
                  <Box>
                    <Button
                      variant="outlined"
                      color="primary"
                      startIcon={<RefreshIcon />}
                      onClick={handleUpdateKb}
                      disabled={updating}
                      sx={{ mr: 1 }}
                    >
                      {updating ? <CircularProgress size={24} /> : 'Update'}
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<VisibilityIcon />}
                      onClick={() => setShowViewDialog(true)}
                      disabled={loading || kbArticles.length === 0}
                    >
                      View Articles
                    </Button>
                  </Box>
                </Box>
                
                <Typography variant="subtitle1" gutterBottom>
                  Query
                </Typography>
                <Paper elevation={1} sx={{ p: 2, backgroundColor: '#f5f5f5', mb: 3 }}>
                  <Typography component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {selectedKb.query}
                  </Typography>
                </Paper>
                
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={6} md={4}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Update Schedule
                    </Typography>
                    <Typography variant="body1">
                      {selectedKb.update_schedule.charAt(0).toUpperCase() + selectedKb.update_schedule.slice(1)}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} md={4}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Articles
                    </Typography>
                    <Typography variant="body1">
                      {loading ? <CircularProgress size={16} sx={{ mr: 1 }} /> : kbArticles.length}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} md={4}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Created Date
                    </Typography>
                    <Typography variant="body1">
                      {new Date(selectedKb.created_date).toLocaleDateString()}
                    </Typography>
                  </Grid>
                </Grid>
                
                <Typography variant="subtitle1" gutterBottom>
                  Recent Updates
                </Typography>
                <Alert severity="info">
                  This knowledge base will automatically update {selectedKb.update_schedule}.
                  You can also update it manually by clicking the Update button.
                </Alert>
              </>
            ) : (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 300 }}>
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  Select a Knowledge Base
                </Typography>
                <Typography variant="body1" color="text.secondary" align="center">
                  Choose a knowledge base from the list to view its details,
                  or create a new one to start collecting literature.
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
      
      {/* Create KB Dialog */}
      <Dialog 
        open={showCreateDialog} 
        onClose={() => setShowCreateDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Create New Knowledge Base</DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Knowledge Base Name"
                value={newKbName}
                onChange={(e) => setNewKbName(e.target.value)}
                required
                helperText="A descriptive name for this knowledge base"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Search Query"
                value={newKbQuery}
                onChange={(e) => setNewKbQuery(e.target.value)}
                required
                multiline
                rows={4}
                helperText="Enter a PubMed query to collect articles"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Update Schedule</InputLabel>
                <Select
                  value={newKbSchedule}
                  label="Update Schedule"
                  onChange={(e) => setNewKbSchedule(e.target.value)}
                >
                  <MenuItem value="daily">Daily</MenuItem>
                  <MenuItem value="weekly">Weekly</MenuItem>
                  <MenuItem value="monthly">Monthly</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCreateDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleCreateKb}
            variant="contained" 
            color="primary"
            disabled={creating || !newKbName || !newKbQuery}
          >
            {creating ? <CircularProgress size={24} /> : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* View KB Articles Dialog */}
      <Dialog 
        open={showViewDialog} 
        onClose={() => setShowViewDialog(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          {selectedKb ? `Articles in "${selectedKb.name}"` : 'Knowledge Base Articles'}
        </DialogTitle>
        <DialogContent>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          ) : kbArticles.length === 0 ? (
            <Alert severity="info">No articles in this knowledge base.</Alert>
          ) : (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Title</TableCell>
                    <TableCell>Authors</TableCell>
                    <TableCell>Journal</TableCell>
                    <TableCell>Date</TableCell>
                    <TableCell>Authority</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {kbArticles.map((article) => (
                    <TableRow key={article.pmid}>
                      <TableCell>
                        <Typography variant="body2" sx={{ maxWidth: 300, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {article.title}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {article.authors ? article.authors.join(', ') : 'Unknown'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" sx={{ maxWidth: 150, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {article.journal || 'Unknown'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {article.human_date || article.publication_date || 'Unknown'}
                      </TableCell>
                      <TableCell>
                        {article.authority_score !== undefined ? article.authority_score : 'Unknown'}
                      </TableCell>
                      <TableCell>
                        <Tooltip title="View on PubMed">
                          <IconButton 
                            size="small"
                            component="a"
                            href={`https://pubmed.ncbi.nlm.nih.gov/${article.pmid}`}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            <VisibilityIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </DialogContent>
        <DialogActions>
          <Button 
            startIcon={<DownloadIcon />}
            disabled={kbArticles.length === 0}
            onClick={() => alert('Export functionality would be here in a real application')}
          >
            Export
          </Button>
          <Button onClick={() => setShowViewDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default KnowledgeBase;

// src/components/ContradictionAnalysis.jsx
import React, { useState, useEffect } from 'react';
import {
  Box, Container, Typography, Paper, Button, Card, CardContent,
  Divider, Grid, CircularProgress, Alert, Tabs, Tab, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, LinearProgress
} from '@mui/material';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import DownloadIcon from '@mui/icons-material/Download';
import { useAuth } from '../contexts/AuthContext';
import { callApi } from '../utils/api';

const ContradictionAnalysis = () => {
  const { token } = useAuth();
  
  // State
  const [loading, setLoading] = useState(false);
  const [detailedLoading, setDetailedLoading] = useState(false);
  const [error, setError] = useState('');
  const [capAnalysis, setCapAnalysis] = useState(null);
  const [detailedAnalysis, setDetailedAnalysis] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  
  // Fetch CAP analysis on mount
  useEffect(() => {
    const fetchCapAnalysis = async () => {
      setLoading(true);
      setError('');
      
      try {
        const response = await callApi('get', '/api/analysis/cap', null, token);
        setCapAnalysis(response);
      } catch (err) {
        setError('Failed to analyze CAP treatments: ' + (err.message || 'Unknown error'));
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchCapAnalysis();
  }, [token]);
  
  const fetchDetailedAnalysis = async () => {
    if (detailedAnalysis) return; // Already loaded
    
    setDetailedLoading(true);
    setError('');
    
    try {
      const response = await callApi('get', '/api/analysis/cap/detailed', null, token);
      setDetailedAnalysis(response);
    } catch (err) {
      setError('Failed to fetch detailed analysis: ' + (err.message || 'Unknown error'));
      console.error(err);
    } finally {
      setDetailedLoading(false);
    }
  };
  
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    
    // Load detailed analysis when switching to the detailed tab
    if (newValue === 1) {
      fetchDetailedAnalysis();
    }
  };
  
  const renderAuthorityChip = (strength) => {
    let color = 'default';
    
    switch(strength) {
      case 'strong':
        color = 'success';
        break;
      case 'moderate':
        color = 'primary';
        break;
      case 'weak':
        color = 'warning';
        break;
      case 'very weak':
        color = 'error';
        break;
      default:
        color = 'default';
    }
    
    return (
      <Chip 
        size="small" 
        color={color} 
        label={`${strength.charAt(0).toUpperCase() + strength.slice(1)} consensus`} 
      />
    );
  };
  
  return (
    <Container maxWidth="lg">
      <Typography variant="h4" component="h1" gutterBottom>
        Community-Acquired Pneumonia (CAP) Treatment Analysis
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="Contradiction Overview" />
        <Tab label="Detailed Treatment Analysis" />
      </Tabs>
      
      {tabValue === 0 && (
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>
            Contradictory Treatment Findings
          </Typography>
          
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          ) : !capAnalysis ? (
            <Alert severity="info">No analysis data available.</Alert>
          ) : (
            <>
              <Typography variant="subtitle1" gutterBottom>
                Found {capAnalysis.num_contradictions} contradictions in {capAnalysis.total_articles} articles
              </Typography>
              
              <Grid container spacing={3} sx={{ mt: 2 }}>
                {Object.entries(capAnalysis.contradictions_by_intervention).map(([intervention, count]) => (
                  count > 0 && (
                    <Grid item xs={12} md={6} lg={4} key={intervention}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            {intervention === 'unspecified' 
                              ? 'Other Antibiotics' 
                              : intervention.charAt(0).toUpperCase() + intervention.slice(1)}
                          </Typography>
                          
                          <Box sx={{ mb: 2 }}>
                            <Typography variant="body2" color="text.secondary" gutterBottom>
                              Contradictions Found
                            </Typography>
                            <Typography variant="h4">
                              {count}
                            </Typography>
                          </Box>
                          
                          <Divider sx={{ mb: 2 }} />
                          
                          <Typography variant="body2" color="text.secondary" gutterBottom>
                            Authority Analysis
                          </Typography>
                          
                          {capAnalysis.authority_analysis[intervention] && (
                            <Box>
                              <Grid container spacing={1}>
                                <Grid item xs={4}>
                                  <Typography variant="caption" component="div" color="text.secondary">
                                    Higher Quality
                                  </Typography>
                                  <Typography variant="body2">
                                    {capAnalysis.authority_analysis[intervention].publication1}
                                  </Typography>
                                </Grid>
                                <Grid item xs={4}>
                                  <Typography variant="caption" component="div" color="text.secondary">
                                    Lower Quality
                                  </Typography>
                                  <Typography variant="body2">
                                    {capAnalysis.authority_analysis[intervention].publication2}
                                  </Typography>
                                </Grid>
                                <Grid item xs={4}>
                                  <Typography variant="caption" component="div" color="text.secondary">
                                    Equal
                                  </Typography>
                                  <Typography variant="body2">
                                    {capAnalysis.authority_analysis[intervention].equal}
                                  </Typography>
                                </Grid>
                              </Grid>
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  )
                ))}
              </Grid>
              
              <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
                <Button
                  variant="outlined"
                  color="primary"
                  startIcon={<AnalyticsIcon />}
                  onClick={() => setTabValue(1)}
                >
                  View Detailed Analysis
                </Button>
              </Box>
            </>
          )}
        </Paper>
      )}
      
      {tabValue === 1 && (
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>
            Treatment Duration vs. Agent Analysis
          </Typography>
          
          {detailedLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
              <CircularProgress />
            </Box>
          ) : !detailedAnalysis ? (
            <Alert severity="info">No detailed analysis data available.</Alert>
          ) : (
            <>
              <Grid container spacing={4}>
                {/* Duration Analysis */}
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Treatment Duration
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Based on {detailedAnalysis.duration_articles} articles with {detailedAnalysis.duration_contradictions} contradictions
                      </Typography>
                      
                      <TableContainer component={Paper} variant="outlined" sx={{ mt: 2 }}>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Duration</TableCell>
                              <TableCell>Consensus</TableCell>
                              <TableCell>Score</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(detailedAnalysis.duration_consensus).map(([duration, consensus]) => (
                              <TableRow key={duration}>
                                <TableCell>
                                  {duration === 'short_course' ? 'Short Course' : 'Long Course'}
                                </TableCell>
                                <TableCell>
                                  {renderAuthorityChip(consensus.strength)}
                                </TableCell>
                                <TableCell>
                                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Box sx={{ width: '100%', mr: 1 }}>
                                      <LinearProgress 
                                        variant="determinate" 
                                        value={Math.min(consensus.score, 100)} 
                                        color={
                                          consensus.strength === 'strong' ? 'success' :
                                          consensus.strength === 'moderate' ? 'primary' :
                                          consensus.strength === 'weak' ? 'warning' : 'error'
                                        }
                                      />
                                    </Box>
                                    <Typography variant="body2" color="text.secondary">
                                      {consensus.score.toFixed(1)}
                                    </Typography>
                                  </Box>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                      
                      <Typography variant="body1" sx={{ mt: 3 }}>
                        Key Finding: {
                          detailedAnalysis.duration_consensus.short_course.score > 
                          detailedAnalysis.duration_consensus.long_course.score ?
                          'Short-course antibiotic therapy has stronger evidence support.' :
                          'Long-course antibiotic therapy has stronger evidence support.'
                        }
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                {/* Agent Analysis */}
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Antibiotic Agent Choice
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Based on {detailedAnalysis.agent_articles} articles with {detailedAnalysis.agent_contradictions} contradictions
                      </Typography>
                      
                      <TableContainer component={Paper} variant="outlined" sx={{ mt: 2 }}>
                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Agent</TableCell>
                              <TableCell>Consensus</TableCell>
                              <TableCell>Score</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(detailedAnalysis.agent_consensus).map(([agent, consensus]) => (
                              <TableRow key={agent}>
                                <TableCell>
                                  {agent === 'macrolide' ? 'Macrolide' : 
                                   agent === 'fluoroquinolone' ? 'Fluoroquinolone' : 'Beta Lactam'}
                                </TableCell>
                                <TableCell>
                                  {renderAuthorityChip(consensus.strength)}
                                </TableCell>
                                <TableCell>
                                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Box sx={{ width: '100%', mr: 1 }}>
                                      <LinearProgress 
                                        variant="determinate" 
                                        value={Math.min(consensus.score, 100)} 
                                        color={
                                          consensus.strength === 'strong' ? 'success' :
                                          consensus.strength === 'moderate' ? 'primary' :
                                          consensus.strength === 'weak' ? 'warning' : 'error'
                                        }
                                      />
                                    </Box>
                                    <Typography variant="body2" color="text.secondary">
                                      {consensus.score.toFixed(1)}
                                    </Typography>
                                  </Box>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                      
                      <Typography variant="body1" sx={{ mt: 3 }}>
                        Key Finding: {
                          (() => {
                            const scores = Object.entries(detailedAnalysis.agent_consensus)
                              .map(([agent, consensus]) => ({ agent, score: consensus.score }));
                            scores.sort((a, b) => b.score - a.score);
                            const topAgent = scores[0].agent;
                            return `${topAgent.charAt(0).toUpperCase() + topAgent.slice(1).replace('_', ' ')} antibiotics have the strongest evidence support.`;
                          })()
                        }
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                {/* Summary Table */}
                <Grid item xs={12}>
                  <Paper elevation={2} sx={{ p: 3, mt: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Evidence Synthesis
                    </Typography>
                    
                    <Typography variant="body1" paragraph>
                      Based on our analysis of the contradictory evidence in current literature on CAP treatments, 
                      we can draw the following conclusions:
                    </Typography>
                    
                    <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Treatment Aspect</TableCell>
                            <TableCell>Recommendation</TableCell>
                            <TableCell>Evidence Strength</TableCell>
                            <TableCell>Contradictions</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          <TableRow>
                            <TableCell>Duration</TableCell>
                            <TableCell>{
                              detailedAnalysis.duration_consensus.short_course.score > 
                              detailedAnalysis.duration_consensus.long_course.score ?
                              'Short-course therapy (≤ 5 days)' :
                              'Long-course therapy (> 5 days)'
                            }</TableCell>
                            <TableCell>{
                              Math.max(
                                detailedAnalysis.duration_consensus.short_course.score,
                                detailedAnalysis.duration_consensus.long_course.score
                              ) > 70 ? 'Strong' :
                              Math.max(
                                detailedAnalysis.duration_consensus.short_course.score,
                                detailedAnalysis.duration_consensus.long_course.score
                              ) > 50 ? 'Moderate' : 'Low'
                            }</TableCell>
                            <TableCell>{detailedAnalysis.duration_contradictions}</TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell>First-line Agent</TableCell>
                            <TableCell>{
                              (() => {
                                const scores = Object.entries(detailedAnalysis.agent_consensus)
                                  .map(([agent, consensus]) => ({ agent, score: consensus.score }));
                                scores.sort((a, b) => b.score - a.score);
                                const topAgent = scores[0].agent;
                                return topAgent.charAt(0).toUpperCase() + topAgent.slice(1).replace('_', ' ');
                              })()
                            }</TableCell>
                            <TableCell>{
                              Math.max(
                                ...Object.values(detailedAnalysis.agent_consensus).map(c => c.score)
                              ) > 70 ? 'Strong' :
                              Math.max(
                                ...Object.values(detailedAnalysis.agent_consensus).map(c => c.score)
                              ) > 50 ? 'Moderate' : 'Low'
                            }</TableCell>
                            <TableCell>{detailedAnalysis.agent_contradictions}</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                    
                    <Typography variant="body1">
                      The evidence synthesis shows {detailedAnalysis.duration_contradictions + detailedAnalysis.agent_contradictions} total 
                      contradictory findings across {detailedAnalysis.duration_articles + detailedAnalysis.agent_articles} analyzed articles. 
                      Higher authority scores indicate more reliable evidence based on publication metrics, study design, and citation patterns.
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </>
          )}
        </Paper>
      )}
    </Container>
  );
};

export default ContradictionAnalysis;