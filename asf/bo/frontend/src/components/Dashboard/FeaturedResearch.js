import React from 'react';
import { 
  Card, CardHeader, CardContent, Divider, Grid, 
  Paper, Typography, Button, Box 
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import RecentUpdates from './RecentUpdates';

/**
 * Featured research section for the dashboard
 * 
 * @param {Object} props - Component props
 * @param {Array} props.updates - Array of recent updates to display
 */
const FeaturedResearch = ({ updates = [] }) => {
  const navigate = useNavigate();
  
  return (
    <Card>
      <CardHeader 
        title="Featured Medical Research" 
        action={
          <Button 
            variant="contained" 
            color="primary"
            startIcon={<SearchIcon />}
            onClick={() => navigate('/pico-search')}
          >
            Search Literature
          </Button>
        }
      />
      <Divider />
      <CardContent>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper 
              elevation={0}
              sx={{ 
                p: 2, 
                borderRadius: 2, 
                bgcolor: 'primary.light', 
                color: 'primary.contrastText',
                display: 'flex',
                flexDirection: 'column',
                height: '100%'
              }}
            >
              <Typography variant="h6" gutterBottom>
                Community Acquired Pneumonia (CAP) Research
              </Typography>
              <Typography variant="body2" paragraph>
                Access the latest research on Community Acquired Pneumonia treatments, 
                diagnostic criteria, and emerging evidence.
              </Typography>
              <Box sx={{ mt: 'auto', display: 'flex', gap: 1 }}>
                <Button 
                  variant="contained" 
                  size="small"
                  sx={{ bgcolor: 'primary.dark' }}
                  onClick={() => navigate('/pico-search')}
                >
                  CAP Treatment Research
                </Button>
                <Button 
                  variant="outlined" 
                  size="small"
                  sx={{ color: 'white', borderColor: 'white' }}
                >
                  View Guidelines
                </Button>
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <RecentUpdates updates={updates} />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default FeaturedResearch;
