import React from 'react';
import {
  Card, CardHeader, CardContent, Divider, Grid, Button
} from '@mui/material';
import {
  Search as SearchIcon,
  Book as BookIcon,
  Biotech as BiotechIcon,
  Science as ScienceIcon,
  SmartToy as SmartToyIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

/**
 * Research tools grid component for the dashboard
 */
const ResearchTools = () => {
  const navigate = useNavigate();

  // Define tools with their properties
  const tools = [
    {
      name: 'PICO Search',
      icon: <SearchIcon />,
      color: 'primary',
      path: '/pico-search',
      description: 'Search medical literature using PICO framework'
    },
    {
      name: 'Knowledge Base',
      icon: <BookIcon />,
      color: 'secondary',
      path: '/knowledge-base',
      description: 'Access curated medical knowledge'
    },
    {
      name: 'Contradiction Analysis',
      icon: <BiotechIcon />,
      color: 'info',
      path: '/contradiction-analysis',
      description: 'Analyze contradictions in medical literature'
    },
    {
      name: 'Medical Terminology',
      icon: <ScienceIcon />,
      color: 'warning',
      path: '/terminology',
      description: 'Look up medical terms and definitions'
    },
    {
      name: 'LLM Management',
      icon: <SmartToyIcon />,
      color: 'success',
      path: '/llm-management',
      description: 'Manage LLM services and models'
    }
  ];

  return (
    <Card>
      <CardHeader title="Research Tools" />
      <Divider />
      <CardContent>
        <Grid container spacing={3}>
          {tools.map((tool, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Button
                variant="outlined"
                color={tool.color}
                size="large"
                startIcon={tool.icon}
                fullWidth
                sx={{ p: 2, height: '100%', borderRadius: 2 }}
                onClick={() => navigate(tool.path)}
              >
                {tool.name}
              </Button>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default ResearchTools;
