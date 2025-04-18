import React from 'react';
import { Box, Typography, Paper, Button, Grid, Card, CardContent, CardActions } from '@mui/material';
import { Link } from 'react-router-dom';
import PageLayout from '../components/Layout/PageLayout';

/**
 * MCP Dashboard Page
 *
 * This is a standalone page for the MCP dashboard that can be accessed directly.
 */
const MCPDashboard = () => {
  return (
    <PageLayout
      title="MCP Dashboard"
      breadcrumbs={[
        { label: 'Home', path: '/' },
        { label: 'LLM Management', path: '/llm-management' },
        { label: 'MCP Dashboard', path: '/mcp-dashboard' }
      ]}
    >
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Model Context Protocol (MCP) Dashboard
        </Typography>
        <Typography paragraph>
          The Model Context Protocol (MCP) is a standardized protocol for interacting with large language models.
          This dashboard allows you to manage MCP providers, monitor their status, and view usage statistics.
        </Typography>

        <Button
          variant="contained"
          color="primary"
          component={Link}
          to="/llm-management?tab=mcp"
          sx={{ mb: 2 }}
        >
          Go to LLM Management MCP Tab
        </Button>

        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Provider Management
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Configure and manage MCP providers with different transport types (stdio, gRPC, HTTP).
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" component={Link} to="/llm-management">
                  Go to LLM Management
                </Button>
              </CardActions>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Status Monitoring
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Monitor the status of your MCP providers and view detailed metrics.
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" component={Link} to="/llm-management">
                  Go to LLM Management
                </Button>
              </CardActions>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Usage Statistics
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  View detailed usage statistics for your MCP providers.
                </Typography>
              </CardContent>
              <CardActions>
                <Button size="small" component={Link} to="/llm-management">
                  Go to LLM Management
                </Button>
              </CardActions>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          About MCP
        </Typography>
        <Typography paragraph>
          The Model Context Protocol (MCP) provides a standardized way to interact with large language models,
          with support for multiple transport options, streaming, and advanced resilience features.
        </Typography>

        <Typography variant="h6" gutterBottom>
          Key Features:
        </Typography>
        <Box component="ul" sx={{ pl: 4 }}>
          <Box component="li">
            <Typography>Multiple transport options (stdio, gRPC, HTTP/REST)</Typography>
          </Box>
          <Box component="li">
            <Typography>Streaming & non-streaming support</Typography>
          </Box>
          <Box component="li">
            <Typography>Advanced resilience with circuit breaker pattern</Typography>
          </Box>
          <Box component="li">
            <Typography>Comprehensive observability with metrics and tracing</Typography>
          </Box>
        </Box>
      </Paper>
    </PageLayout>
  );
};

export default MCPDashboard;
