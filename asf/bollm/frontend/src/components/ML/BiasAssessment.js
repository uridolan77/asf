import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, TextField, Button, Grid, Chip,
  FormControlLabel, Switch, Divider, CircularProgress,
  Card, CardContent, CardHeader, CardActions, Tooltip, Alert,
  LinearProgress, FormControl, InputLabel, Select, MenuItem,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Rating, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Science as ScienceIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  WarningAmber as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HelpOutline as HelpIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext';
// Import apiService as a fallback
import defaultApiService from '../../services/api';
import { ButtonLoader, ContentLoader } from '../UI/LoadingIndicators.js';
import { FadeIn, HoverAnimation } from '../UI/Animations';

/**
 * Bias Assessment component
 *
 * This component allows users to assess bias in medical articles
 * using various bias assessment tools like ROBINS-I, RoB 2, etc.
 */
const BiasAssessment = ({ onExport, apiService }) => {
  // Use provided apiService or fall back to the default
  const api = apiService || defaultApiService;
  const { showSuccess, showError } = useNotification();

  // Form state
  const [articleId, setArticleId] = useState('');
  const [title, setTitle] = useState('');
  const [abstract, setAbstract] = useState('');
  const [fullText, setFullText] = useState('');
  const [assessmentType, setAssessmentType] = useState('robins-i');

  // UI state
  const [isAssessing, setIsAssessing] = useState(false);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [biasTools, setBiasTools] = useState([]);

  // Load bias assessment tools - only when explicitly called
  const loadBiasTools = async () => {
    if (isLoadingTools) return;

    setIsLoadingTools(true);

    try {
      const result = await api.ml.getBiasAssessmentTools();

      if (result.success) {
        setBiasTools(result.data.tools);
        showSuccess('Bias assessment tools loaded successfully');
      } else {
        showError(`Failed to load bias assessment tools: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading bias assessment tools:', error);
      showError(`Error loading bias assessment tools: ${error.message}`);
    } finally {
      setIsLoadingTools(false);
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!articleId.trim() && !abstract.trim() && !fullText.trim()) {
      showError('Please provide either an article ID, abstract, or full text');
      setError('Please provide either an article ID, abstract, or full text');
      return;
    }

    setIsAssessing(true);
    setError('');

    try {
      const params = {
        article_id: articleId.trim() || null,
        title: title.trim() || null,
        abstract: abstract.trim() || null,
        full_text: fullText.trim() || null,
        assessment_type: assessmentType
      };

      const result = await api.ml.assessBias(params);

      if (result.success) {
        setResult(result.data);
        showSuccess('Bias assessment completed successfully');
      } else {
        setError(`Assessment failed: ${result.error}`);
        showError(`Assessment failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Error assessing bias:', error);
      setError(`Assessment error: ${error.message}`);
      showError(`Assessment error: ${error.message}`);
    } finally {
      setIsAssessing(false);
    }
  };

  // Handle export
  const handleExport = (format) => {
    if (!result) return;

    if (onExport) {
      onExport(format, {
        analysis_id: result.assessment_id
      });
    }
  };

  // Get risk of bias color
  const getBiasRiskColor = (risk) => {
    switch (risk.toLowerCase()) {
      case 'high':
        return 'error';
      case 'moderate':
      case 'some concerns':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'info';
    }
  };

  // Get risk of bias icon
  const getBiasRiskIcon = (risk) => {
    switch (risk.toLowerCase()) {
      case 'high':
        return <ErrorIcon color="error" />;
      case 'moderate':
      case 'some concerns':
        return <WarningIcon color="warning" />;
      case 'low':
        return <CheckCircleIcon color="success" />;
      default:
        return <HelpIcon color="info" />;
    }
  };

  // Get domain score
  const getDomainScore = (score) => {
    if (score >= 0.8) return 5;
    if (score >= 0.6) return 4;
    if (score >= 0.4) return 3;
    if (score >= 0.2) return 2;
    return 1;
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <WarningIcon sx={{ mr: 1 }} />
          Bias Assessment
          <Tooltip title="Assess bias in medical articles using various bias assessment tools">
            <InfoIcon fontSize="small" sx={{ ml: 1, color: 'text.secondary' }} />
          </Tooltip>
        </Typography>

        <Divider sx={{ mb: 3 }} />

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Box sx={{ mb: 3, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            startIcon={isLoadingTools ? <ButtonLoader size={20} /> : <RefreshIcon />}
            onClick={loadBiasTools}
            disabled={isLoadingTools}
            sx={{ mb: 2 }}
          >
            {isLoadingTools ? 'Loading...' : 'Load Assessment Tools'}
          </Button>
        </Box>

        <form onSubmit={handleSubmit}>
          <Grid container spacing={3}>
            {/* Article ID */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Article ID"
                variant="outlined"
                placeholder="e.g., PMID:12345678"
                value={articleId}
                onChange={(e) => setArticleId(e.target.value)}
                helperText="Enter a PubMed ID (PMID) or DOI"
              />
            </Grid>

            {/* Assessment Type */}
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel id="assessment-type-label">Assessment Tool</InputLabel>
                <Select
                  labelId="assessment-type-label"
                  value={assessmentType}
                  label="Assessment Tool"
                  onChange={(e) => setAssessmentType(e.target.value)}
                >
                  {biasTools.map((tool) => (
                    <MenuItem key={tool.id} value={tool.id}>
                      {tool.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Title */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Article Title (Optional)"
                variant="outlined"
                placeholder="Enter the article title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </Grid>

            {/* Abstract */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Abstract (Optional)"
                variant="outlined"
                placeholder="Enter the article abstract"
                value={abstract}
                onChange={(e) => setAbstract(e.target.value)}
                multiline
                rows={4}
                helperText="Required if article ID is not provided"
              />
            </Grid>

            {/* Full Text */}
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Full Text (Optional)"
                variant="outlined"
                placeholder="Enter the article full text"
                value={fullText}
                onChange={(e) => setFullText(e.target.value)}
                multiline
                rows={6}
                helperText="Provides more accurate assessment if available"
              />
            </Grid>

            {/* Action buttons */}
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={isAssessing ? <ButtonLoader size={20} /> : <ScienceIcon />}
                  disabled={(!articleId.trim() && !abstract.trim() && !fullText.trim()) || isAssessing}
                >
                  {isAssessing ? 'Assessing...' : 'Assess Bias'}
                </Button>

                <Button
                  variant="outlined"
                  onClick={() => {
                    setArticleId('');
                    setTitle('');
                    setAbstract('');
                    setFullText('');
                    setResult(null);
                    setError('');
                  }}
                >
                  Clear
                </Button>
              </Box>
            </Grid>
          </Grid>
        </form>
      </Paper>

      {/* Results */}
      {result && (
        <FadeIn>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Bias Assessment Results
              </Typography>

              <Box>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={() => handleExport('pdf')}
                  sx={{ mr: 1 }}
                >
                  Export as PDF
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={() => handleExport('json')}
                >
                  Export as JSON
                </Button>
              </Box>
            </Box>

            <Divider sx={{ mb: 3 }} />

            <Grid container spacing={3}>
              <Grid item xs={12}>
                <HoverAnimation>
                  <Card variant="outlined">
                    <CardHeader
                      title={result.article_title || 'Article Assessment'}
                      subheader={`Assessment ID: ${result.assessment_id}`}
                      action={
                        <Chip
                          label={`Overall: ${result.overall_risk}`}
                          color={getBiasRiskColor(result.overall_risk)}
                          icon={getBiasRiskIcon(result.overall_risk)}
                        />
                      }
                    />
                    <CardContent>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle2" gutterBottom>
                            Article ID:
                          </Typography>
                          <Typography variant="body2" sx={{ mb: 2 }}>
                            {result.article_id || 'Not provided'}
                          </Typography>

                          <Typography variant="subtitle2" gutterBottom>
                            Assessment Tool:
                          </Typography>
                          <Typography variant="body2">
                            {result.assessment_type}
                          </Typography>
                        </Grid>

                        <Grid item xs={12} md={6}>
                          <Typography variant="subtitle2" gutterBottom>
                            Assessment Date:
                          </Typography>
                          <Typography variant="body2" sx={{ mb: 2 }}>
                            {new Date(result.assessment_date).toLocaleString()}
                          </Typography>

                          <Typography variant="subtitle2" gutterBottom>
                            Study Type:
                          </Typography>
                          <Typography variant="body2">
                            {result.study_type || 'Not determined'}
                          </Typography>
                        </Grid>

                        <Grid item xs={12}>
                          <Typography variant="subtitle2" gutterBottom>
                            Overall Risk of Bias:
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                            {getBiasRiskIcon(result.overall_risk)}
                            <Typography variant="body1" sx={{ ml: 1 }}>
                              {result.overall_risk}
                            </Typography>
                          </Box>

                          {result.summary && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="subtitle2" gutterBottom>
                                Summary:
                              </Typography>
                              <Typography variant="body2">
                                {result.summary}
                              </Typography>
                            </Box>
                          )}
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </HoverAnimation>
              </Grid>

              {/* Domain Assessments */}
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Domain Assessments
                </Typography>

                {result.domains.map((domain, index) => (
                  <Accordion key={index} sx={{ mb: 1 }}>
                    <AccordionSummary
                      expandIcon={<ExpandMoreIcon />}
                      aria-controls={`domain-${index}-content`}
                      id={`domain-${index}-header`}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
                        <Typography variant="subtitle1">{domain.name}</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Rating
                            value={getDomainScore(domain.score)}
                            readOnly
                            size="small"
                            sx={{ mr: 2 }}
                          />
                          <Chip
                            label={domain.risk}
                            color={getBiasRiskColor(domain.risk)}
                            size="small"
                            icon={getBiasRiskIcon(domain.risk)}
                          />
                        </Box>
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="body2" gutterBottom>
                        {domain.description}
                      </Typography>

                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Assessment:
                        </Typography>
                        <Typography variant="body2" sx={{ mb: 2 }}>
                          {domain.assessment}
                        </Typography>

                        <Typography variant="subtitle2" gutterBottom>
                          Score: {(domain.score * 100).toFixed(1)}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={domain.score * 100}
                          color={getBiasRiskColor(domain.risk)}
                          sx={{ height: 10, borderRadius: 5, mb: 2 }}
                        />

                        {domain.criteria && domain.criteria.length > 0 && (
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>
                              Criteria:
                            </Typography>
                            <TableContainer component={Paper} variant="outlined">
                              <Table size="small">
                                <TableHead>
                                  <TableRow>
                                    <TableCell>Criterion</TableCell>
                                    <TableCell>Judgment</TableCell>
                                    <TableCell>Support</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {domain.criteria.map((criterion, idx) => (
                                    <TableRow key={idx}>
                                      <TableCell>{criterion.name}</TableCell>
                                      <TableCell>
                                        <Chip
                                          label={criterion.judgment}
                                          color={getBiasRiskColor(criterion.judgment)}
                                          size="small"
                                        />
                                      </TableCell>
                                      <TableCell>{criterion.support}</TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </Box>
                        )}
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                ))}
              </Grid>

              {/* Recommendations */}
              {result.recommendations && result.recommendations.length > 0 && (
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    Recommendations
                  </Typography>
                  <HoverAnimation>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="body2" sx={{ mb: 2 }}>
                          Based on the bias assessment, the following recommendations are provided:
                        </Typography>

                        <Box component="ul" sx={{ pl: 2 }}>
                          {result.recommendations.map((recommendation, index) => (
                            <Box component="li" key={index} sx={{ mb: 1 }}>
                              <Typography variant="body2">
                                {recommendation}
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      </CardContent>
                    </Card>
                  </HoverAnimation>
                </Grid>
              )}
            </Grid>
          </Paper>
        </FadeIn>
      )}
    </Box>
  );
};

export default BiasAssessment;
