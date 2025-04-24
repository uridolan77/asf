import React from 'react';
import { Box, Typography, Paper, Button, Container } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import ConstructionIcon from '@mui/icons-material/Construction';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

/**
 * A placeholder page component for routes that are not yet implemented
 * @param {Object} props - Component props
 * @param {string} props.title - Page title
 * @param {string} props.description - Optional description
 */
const PlaceholderPage = ({ title, description }) => {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Extract title from path if not provided
  const derivedTitle = title || location.pathname.split('/').pop().replace(/-/g, ' ');
  
  // Format title for display
  const formattedTitle = derivedTitle
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
  
  return (
    <Container maxWidth="lg">
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          {formattedTitle}
        </Typography>
        
        <Paper 
          elevation={3} 
          sx={{ 
            p: 4, 
            mt: 3, 
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '300px'
          }}
        >
          <ConstructionIcon sx={{ fontSize: 80, color: 'warning.main', mb: 3 }} />
          
          <Typography variant="h5" gutterBottom>
            This Feature is Under Construction
          </Typography>
          
          <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mb: 4 }}>
            {description || `The ${formattedTitle} feature is currently being developed and will be available soon. Please check back later.`}
          </Typography>
          
          <Button
            variant="outlined"
            startIcon={<ArrowBackIcon />}
            onClick={() => navigate('/llm/dashboard')}
          >
            Return to Dashboard
          </Button>
        </Paper>
      </Box>
    </Container>
  );
};

export default PlaceholderPage;