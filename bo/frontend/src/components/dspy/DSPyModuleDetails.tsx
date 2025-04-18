import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Divider,
  Button,
  Chip,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
  Alert,
  Tabs,
  Tab
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Assessment as AssessmentIcon,
  InsertDriveFile as FileIcon,
  Code as CodeIcon,
  BubbleChart as BubbleChartIcon
} from '@mui/icons-material';
import ClientService from '../../services/ClientService';
import { formatDateTime } from '../../utils/formatters';
import { TabPanel, a11yProps } from '../common/TabPanel';

interface DSPyModuleDetailsProps {
  module: any;
  onRefresh: () => void;
}

const DSPyModuleDetails: React.FC<DSPyModuleDetailsProps> = ({ module, onRefresh }) => {
  const [metrics, setMetrics] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [currentTab, setCurrentTab] = useState<number>(0);

  // Sample structured data to display in the UI
  // In a real implementation, this would come from the API
  const sampleInputs = [
    { name: 'text', type: 'string', description: 'Medical text to analyze' },
    { name: 'claim', type: 'string', description: 'The claim to find evidence for' }
  ];

  const sampleOutputs = [
    { name: 'evidence', type: 'string', description: 'Extracted evidence from the text' },
    { name: 'relation', type: 'string', description: 'Relation between evidence and claim' },
    { name: 'confidence', type: 'number', description: 'Confidence score between 0 and 1' },
    { name: 'reasoning', type: 'string', description: 'Reasoning process for determining the relation' }
  ];

  const sampleCode = `
# Example usage of the ${module.name} module
from asf.medical.ml.dspy import get_enhanced_client

async def main():
    client = await get_enhanced_client()
    
    # Call the module
    result = await client.call_module(
        module_name="${module.name}",
        text="The patient has a history of hypertension and diabetes.",
        claim="The patient has diabetes."
    )
    
    print(f"Evidence: {result['evidence']}")
    print(f"Relation: {result['relation']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
`;

  useEffect(() => {
    loadMetrics();
  }, [module]);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      // In a real implementation, you would fetch metrics from your API
      // const data = await ClientService.getModuleMetrics(module.module_id);
      // setMetrics(data);
      
      // For now, use mock data
      setTimeout(() => {
        const mockData = generateMockMetrics();
        setMetrics(mockData);
        setLoading(false);
      }, 500);
    } catch (err) {
      console.error('Error loading metrics:', err);
      setError('Failed to load module metrics. Please try again.');
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  // Generate some mock metrics for display purposes
  const generateMockMetrics = () => {
    const now = new Date();
    const mockMetrics = [];
    
    for (let i = 0; i < 10; i++) {
      const date = new Date(now);
      date.setMinutes(now.getMinutes() - i * 10);
      
      mockMetrics.push({
        metric_id: `mock-${i}`,
        timestamp: date.toISOString(),
        response_time: Math.random() * 2 + 0.5, // 0.5 to 2.5 seconds
        success: Math.random() > 0.2, // 80% success rate
        cached: Math.random() > 0.7, // 30% cache hits
        tokens_used: Math.floor(Math.random() * 1000) + 200 // 200 to 1200 tokens
      });
    }
    
    return mockMetrics;
  };

  return (
    <Box>
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box>
            <Typography variant="h5" gutterBottom>
              {module.name}
              <Chip
                label={module.module_type || module.class_name || "Module"}
                color="primary"
                size="small"
                variant="outlined"
                sx={{ ml: 1 }}
              />
            </Typography>
            <Typography variant="body1" color="textSecondary" paragraph>
              {module.description || 'No description provided'}
            </Typography>
          </Box>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={onRefresh}
          >
            Refresh
          </Button>
        </Box>

        <Divider sx={{ mb: 2 }} />
        
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2">Module ID</Typography>
            <Typography variant="body2">{module.module_id}</Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2">Registered At</Typography>
            <Typography variant="body2">{formatDateTime(module.registered_at)}</Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2">Updated At</Typography>
            <Typography variant="body2">{formatDateTime(module.updated_at)}</Typography>
          </Grid>
        </Grid>

        <Box sx={{ width: '100%' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={currentTab} onChange={handleTabChange} aria-label="module details tabs">
              <Tab icon={<AssessmentIcon />} label="Metrics" {...a11yProps(0)} />
              <Tab icon={<FileIcon />} label="Schema" {...a11yProps(1)} />
              <Tab icon={<CodeIcon />} label="Examples" {...a11yProps(2)} />
              <Tab icon={<BubbleChartIcon />} label="Dependencies" {...a11yProps(3)} />
            </Tabs>
          </Box>
          
          <TabPanel value={currentTab} index={0}>
            <Typography variant="h6" gutterBottom>Performance Metrics</Typography>
            
            {loading ? (
              <LinearProgress />
            ) : error ? (
              <Alert severity="error">{error}</Alert>
            ) : metrics.length === 0 ? (
              <Alert severity="info">No metrics available for this module.</Alert>
            ) : (
              <>
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={12} md={3}>
                    <Card elevation={1}>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>Average Response Time</Typography>
                        <Typography variant="h4" color="primary">
                          {(metrics.reduce((acc, m) => acc + m.response_time, 0) / metrics.length).toFixed(2)}s
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card elevation={1}>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>Success Rate</Typography>
                        <Typography variant="h4" color="primary">
                          {(metrics.filter(m => m.success).length / metrics.length * 100).toFixed(0)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card elevation={1}>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>Cache Hit Rate</Typography>
                        <Typography variant="h4" color="primary">
                          {(metrics.filter(m => m.cached).length / metrics.length * 100).toFixed(0)}%
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} md={3}>
                    <Card elevation={1}>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom>Avg Tokens Used</Typography>
                        <Typography variant="h4" color="primary">
                          {Math.round(metrics.reduce((acc, m) => acc + m.tokens_used, 0) / metrics.length)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              
                <Typography variant="subtitle1" gutterBottom>Recent Executions</Typography>
                <Paper variant="outlined">
                  <List dense>
                    {metrics.map((metric) => (
                      <ListItem key={metric.metric_id} divider>
                        <Grid container spacing={2}>
                          <Grid item xs={4}>
                            <ListItemText
                              primary={formatDateTime(metric.timestamp)}
                              secondary={`Response Time: ${metric.response_time.toFixed(2)}s`}
                            />
                          </Grid>
                          <Grid item xs={3}>
                            <ListItemText
                              primary={
                                <Chip
                                  label={metric.success ? 'Success' : 'Failed'}
                                  color={metric.success ? 'success' : 'error'}
                                  size="small"
                                />
                              }
                              secondary={metric.cached ? 'Cached Response' : 'LLM Response'}
                            />
                          </Grid>
                          <Grid item xs={5}>
                            <ListItemText
                              primary={`Tokens: ${metric.tokens_used}`}
                              secondary={`Cost est: $${(metric.tokens_used * 0.00002).toFixed(4)}`}
                            />
                          </Grid>
                        </Grid>
                      </ListItem>
                    ))}
                  </List>
                </Paper>
              </>
            )}
          </TabPanel>
          
          <TabPanel value={currentTab} index={1}>
            <Typography variant="h6" gutterBottom>Module Schema</Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Input Fields
                </Typography>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  {sampleInputs.length === 0 ? (
                    <Typography variant="body2" color="textSecondary">No input fields available</Typography>
                  ) : (
                    <List dense>
                      {sampleInputs.map((input, index) => (
                        <ListItem key={index} divider={index < sampleInputs.length - 1}>
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                {input.name}
                                <Chip 
                                  label={input.type} 
                                  size="small" 
                                  variant="outlined"
                                  sx={{ ml: 1 }}
                                />
                              </Box>
                            }
                            secondary={input.description}
                          />
                        </ListItem>
                      ))}
                    </List>
                  )}
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Output Fields
                </Typography>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  {sampleOutputs.length === 0 ? (
                    <Typography variant="body2" color="textSecondary">No output fields available</Typography>
                  ) : (
                    <List dense>
                      {sampleOutputs.map((output, index) => (
                        <ListItem key={index} divider={index < sampleOutputs.length - 1}>
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                {output.name}
                                <Chip 
                                  label={output.type} 
                                  size="small" 
                                  variant="outlined"
                                  sx={{ ml: 1 }}
                                />
                              </Box>
                            }
                            secondary={output.description}
                          />
                        </ListItem>
                      ))}
                    </List>
                  )}
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>
          
          <TabPanel value={currentTab} index={2}>
            <Typography variant="h6" gutterBottom>Example Usage</Typography>
            <Paper 
              variant="outlined" 
              sx={{ 
                p: 2, 
                backgroundColor: '#f5f5f5',
                fontFamily: 'monospace',
                fontSize: '0.9rem',
                whiteSpace: 'pre-wrap',
                overflowX: 'auto'
              }}
            >
              {sampleCode}
            </Paper>
          </TabPanel>
          
          <TabPanel value={currentTab} index={3}>
            <Typography variant="h6" gutterBottom>Module Dependencies</Typography>
            <Alert severity="info">
              Dependencies information is not available in this preview version.
            </Alert>
          </TabPanel>
        </Box>
      </Paper>
    </Box>
  );
};

export default DSPyModuleDetails;