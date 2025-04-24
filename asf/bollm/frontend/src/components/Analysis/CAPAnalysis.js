import React, { useState, useEffect } from 'react';
import {
  Box, Paper, Typography, Button, Grid, Chip,
  Divider, CircularProgress, Alert, Tooltip,
  Card, CardContent, CardHeader, Tab, Tabs,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, LinearProgress
} from '@mui/material';
import {
  Science as ScienceIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  BiotechOutlined as BiotechIcon,
  PersonOutline as PersonIcon,
  BarChart as BarChartIcon,
  MedicalInformation as MedicalIcon
} from '@mui/icons-material';

import { useNotification } from '../../context/NotificationContext';
import apiService from '../../services/api';
import { ContentLoader } from '../UI/LoadingIndicators';
import { FadeIn, SlideIn } from '../UI/Animations';

/**
 * CAP (Community-Acquired Pneumonia) Analysis component
 * 
 * This component displays analysis of CAP literature, including
 * treatment patterns, patient populations, and outcomes.
 */
const CAPAnalysis = ({ onExport }) => {
  const { showSuccess, showError } = useNotification();
  
  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState(0);
  
  // Load CAP analysis on mount
  useEffect(() => {
    loadCAPAnalysis();
  }, []);
  
  // Load CAP analysis
  const loadCAPAnalysis = async () => {
    setIsLoading(true);
    setError('');
    
    try {
      const result = await apiService.analysis.cap();
      
      if (result.success) {
        setResults(result.data.data);
        showSuccess('CAP analysis loaded successfully');
      } else {
        setError(`Failed to load CAP analysis: ${result.error}`);
        showError(`Failed to load CAP analysis: ${result.error}`);
      }
    } catch (error) {
      console.error('Error loading CAP analysis:', error);
      setError(`Error loading CAP analysis: ${error.message}`);
      showError(`Error loading CAP analysis: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle tab change
  const handleTabChange = (_, newValue) => {
    setActiveTab(newValue);
  };
  
  // Handle export
  const handleExport = (format) => {
    if (!results) return;
    
    if (onExport) {
      onExport(format, {
        analysis_id: results.analysis_id
      });
    }
  };
  
  // Get effectiveness color
  const getEffectivenessColor = (effectiveness) => {
    if (effectiveness >= 80) return 'success';
    if (effectiveness >= 50) return 'primary';
    if (effectiveness >= 30) return 'warning';
    return 'error';
  };
  
  if (isLoading) {
    return <ContentLoader height={400} message="Loading CAP analysis..." />;
  }
  
  return (
    <Box sx={{ width: '100%' }}>
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <MedicalIcon sx={{ mr: 1 }} />
          Community-Acquired Pneumonia (CAP) Analysis
          <Tooltip title="Comprehensive analysis of CAP literature, including treatments, populations, and outcomes">
            <InfoIcon fontSize="small" sx={{ ml: 1, color: 'text.secondary' }} />
          </Tooltip>
        </Typography>
        
        <Divider sx={{ mb: 3 }} />
        
        {error ? (
          <Alert 
            severity="error" 
            action={
              <Button color="inherit" size="small" onClick={loadCAPAnalysis}>
                Retry
              </Button>
            }
          >
            {error}
          </Alert>
        ) : results ? (
          <FadeIn>
            <Box sx={{ mb: 3 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={8}>
                  <Typography variant="subtitle1">
                    Analysis ID: {results.analysis_id}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Last updated: {new Date(results.timestamp).toLocaleString()}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4} sx={{ display: 'flex', justifyContent: 'flex-end' }}>
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
                    onClick={() => handleExport('excel')}
                  >
                    Export as Excel
                  </Button>
                </Grid>
              </Grid>
            </Box>
            
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs 
                value={activeTab} 
                onChange={handleTabChange} 
                aria-label="CAP analysis tabs"
                variant="fullWidth"
              >
                <Tab 
                  icon={<BiotechIcon />} 
                  label="Treatments" 
                  id="tab-0" 
                  aria-controls="tabpanel-0" 
                />
                <Tab 
                  icon={<PersonIcon />} 
                  label="Populations" 
                  id="tab-1" 
                  aria-controls="tabpanel-1" 
                />
                <Tab 
                  icon={<BarChartIcon />} 
                  label="Outcomes" 
                  id="tab-2" 
                  aria-controls="tabpanel-2" 
                />
              </Tabs>
            </Box>
            
            {/* Treatments Tab */}
            <Box role="tabpanel" hidden={activeTab !== 0} id="tabpanel-0" aria-labelledby="tab-0" sx={{ py: 3 }}>
              {activeTab === 0 && (
                <SlideIn>
                  <Typography variant="h6" gutterBottom>
                    Treatment Effectiveness
                  </Typography>
                  
                  <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Treatment</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Effectiveness</TableCell>
                          <TableCell>Studies</TableCell>
                          <TableCell>Patient Count</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {results.treatments.map((treatment, index) => (
                          <TableRow key={index} hover>
                            <TableCell>{treatment.name}</TableCell>
                            <TableCell>{treatment.type}</TableCell>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                <Box sx={{ width: '100%', mr: 1 }}>
                                  <LinearProgress
                                    variant="determinate"
                                    value={treatment.effectiveness}
                                    color={getEffectivenessColor(treatment.effectiveness)}
                                    sx={{ height: 10, borderRadius: 5 }}
                                  />
                                </Box>
                                <Box sx={{ minWidth: 35 }}>
                                  <Typography variant="body2" color="text.secondary">
                                    {`${Math.round(treatment.effectiveness)}%`}
                                  </Typography>
                                </Box>
                              </Box>
                            </TableCell>
                            <TableCell>{treatment.studies}</TableCell>
                            <TableCell>{treatment.patient_count.toLocaleString()}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  
                  <Typography variant="h6" gutterBottom>
                    Treatment Combinations
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {results.treatment_combinations.map((combo, index) => (
                      <Grid item xs={12} md={6} lg={4} key={index}>
                        <Card variant="outlined">
                          <CardHeader
                            title={combo.name}
                            subheader={`${combo.studies} studies, ${combo.patient_count.toLocaleString()} patients`}
                          />
                          <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                              Effectiveness: {combo.effectiveness}%
                            </Typography>
                            <LinearProgress
                              variant="determinate"
                              value={combo.effectiveness}
                              color={getEffectivenessColor(combo.effectiveness)}
                              sx={{ height: 10, borderRadius: 5, mb: 2 }}
                            />
                            <Typography variant="body2">
                              {combo.description}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </SlideIn>
              )}
            </Box>
            
            {/* Populations Tab */}
            <Box role="tabpanel" hidden={activeTab !== 1} id="tabpanel-1" aria-labelledby="tab-1" sx={{ py: 3 }}>
              {activeTab === 1 && (
                <SlideIn>
                  <Typography variant="h6" gutterBottom>
                    Patient Populations
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {results.populations.map((population, index) => (
                      <Grid item xs={12} md={6} key={index}>
                        <Card variant="outlined">
                          <CardHeader
                            title={population.name}
                            subheader={`${population.studies} studies, ${population.patient_count.toLocaleString()} patients`}
                          />
                          <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                              Key Characteristics:
                            </Typography>
                            <Box component="ul" sx={{ pl: 2 }}>
                              {population.characteristics.map((char, idx) => (
                                <Box component="li" key={idx}>
                                  <Typography variant="body2">{char}</Typography>
                                </Box>
                              ))}
                            </Box>
                            
                            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                              Most Effective Treatments:
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                              {population.effective_treatments.map((treatment, idx) => (
                                <Chip
                                  key={idx}
                                  label={`${treatment.name} (${treatment.effectiveness}%)`}
                                  color={getEffectivenessColor(treatment.effectiveness)}
                                  size="small"
                                />
                              ))}
                            </Box>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </SlideIn>
              )}
            </Box>
            
            {/* Outcomes Tab */}
            <Box role="tabpanel" hidden={activeTab !== 2} id="tabpanel-2" aria-labelledby="tab-2" sx={{ py: 3 }}>
              {activeTab === 2 && (
                <SlideIn>
                  <Typography variant="h6" gutterBottom>
                    Clinical Outcomes
                  </Typography>
                  
                  <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Outcome</TableCell>
                          <TableCell>Overall Rate</TableCell>
                          <TableCell>Best Treatment</TableCell>
                          <TableCell>Worst Treatment</TableCell>
                          <TableCell>Studies</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {results.outcomes.map((outcome, index) => (
                          <TableRow key={index} hover>
                            <TableCell>{outcome.name}</TableCell>
                            <TableCell>{outcome.overall_rate}%</TableCell>
                            <TableCell>
                              <Chip
                                label={`${outcome.best_treatment.name} (${outcome.best_treatment.rate}%)`}
                                color="success"
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={`${outcome.worst_treatment.name} (${outcome.worst_treatment.rate}%)`}
                                color="error"
                                size="small"
                              />
                            </TableCell>
                            <TableCell>{outcome.studies}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                  
                  <Typography variant="h6" gutterBottom>
                    Outcome by Population
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {results.outcome_by_population.map((item, index) => (
                      <Grid item xs={12} md={6} lg={4} key={index}>
                        <Card variant="outlined">
                          <CardHeader
                            title={item.population}
                            subheader={`${item.studies} studies`}
                          />
                          <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                              Key Outcomes:
                            </Typography>
                            <TableContainer>
                              <Table size="small">
                                <TableHead>
                                  <TableRow>
                                    <TableCell>Outcome</TableCell>
                                    <TableCell align="right">Rate</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {item.outcomes.map((outcome, idx) => (
                                    <TableRow key={idx}>
                                      <TableCell>{outcome.name}</TableCell>
                                      <TableCell align="right">{outcome.rate}%</TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </SlideIn>
              )}
            </Box>
          </FadeIn>
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<ScienceIcon />}
              onClick={loadCAPAnalysis}
            >
              Load CAP Analysis
            </Button>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default CAPAnalysis;
