// Example page showing React Query and Zustand integration
import React, { useState } from 'react';
import { 
  Box, Typography, Paper, Grid, Button, 
  CircularProgress, Chip, IconButton
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import FavoriteIcon from '@mui/icons-material/Favorite';
import FavoriteBorderIcon from '@mui/icons-material/FavoriteBorder';
import { useApiQuery } from '../hooks/useApi';
import { useAppConfigStore } from '../store/appConfigStore';

// Types for the data we're fetching
interface ClaimData {
  id: string;
  text: string;
  source: string;
  confidence: number;
}

// Example claim list component using React Query + Zustand
const ClaimsExamplePage: React.FC = () => {
  // Get feature flags from Zustand store
  const { featureFlags, toggleFeature } = useAppConfigStore();
  const showBetaVisualizer = featureFlags.find(f => f.id === 'beta-visualizer')?.enabled || false;
  
  // State for selected claim
  const [selectedClaimId, setSelectedClaimId] = useState<string | null>(null);
  
  // Fetch claims data with React Query
  const { 
    data, 
    isLoading, 
    isError, 
    error, 
    refetch 
  } = useApiQuery<{ claims: ClaimData[] }>(
    '/api/claims', 
    ['claims'],
    {
      // Only fetch if beta visualizer is enabled
      enabled: showBetaVisualizer,
      // Stale time of 5 minutes
      staleTime: 5 * 60 * 1000,
      // Show error toast on failure
      onError: (err) => {
        console.error('Failed to fetch claims:', err);
      }
    }
  );
  
  // Favorites system (just an example, could be in Zustand too)
  const [favorites, setFavorites] = useState<string[]>([]);
  
  const toggleFavorite = (claimId: string) => {
    if (favorites.includes(claimId)) {
      setFavorites(favorites.filter(id => id !== claimId));
    } else {
      setFavorites([...favorites, claimId]);
    }
  };
  
  if (!showBetaVisualizer) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>Claims Visualizer</Typography>
        <Paper sx={{ p: 3, maxWidth: 600, mb: 3 }}>
          <Typography variant="body1" paragraph>
            The beta claim visualizer is currently disabled.
          </Typography>
          <Button 
            variant="contained" 
            onClick={() => toggleFeature('beta-visualizer')}
          >
            Enable Beta Visualizer
          </Button>
        </Paper>
      </Box>
    );
  }
  
  if (isLoading) {
    return (
      <Box sx={{ p: 3, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (isError) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography color="error" variant="h6">
          Error loading claims: {(error as any)?.message || 'Unknown error'}
        </Typography>
        <Button 
          variant="outlined" 
          onClick={() => refetch()} 
          startIcon={<RefreshIcon />}
          sx={{ mt: 2 }}
        >
          Retry
        </Button>
      </Box>
    );
  }
  
  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Claims Visualizer (Beta)</Typography>
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />} 
          onClick={() => refetch()}
        >
          Refresh Data
        </Button>
      </Box>
      
      <Grid container spacing={3}>
        {data?.claims.map((claim) => (
          <Grid item xs={12} sm={6} md={4} key={claim.id}>
            <Paper 
              sx={{ 
                p: 2, 
                height: '100%',
                border: selectedClaimId === claim.id ? '2px solid #2196f3' : 'none',
                cursor: 'pointer'
              }}
              onClick={() => setSelectedClaimId(claim.id)}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Chip 
                  label={`Confidence: ${Math.round(claim.confidence * 100)}%`}
                  color={claim.confidence > 0.7 ? 'success' : claim.confidence > 0.4 ? 'warning' : 'error'}
                  size="small"
                />
                <IconButton 
                  size="small" 
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleFavorite(claim.id);
                  }}
                  color="primary"
                >
                  {favorites.includes(claim.id) ? <FavoriteIcon /> : <FavoriteBorderIcon />}
                </IconButton>
              </Box>
              
              <Typography variant="body1" sx={{ mb: 2 }}>
                {claim.text}
              </Typography>
              
              <Typography variant="caption" color="text.secondary">
                Source: {claim.source}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>
      
      {data?.claims.length === 0 && (
        <Paper sx={{ p: 3, mt: 2 }}>
          <Typography variant="body1" align="center">
            No claims found. Try adjusting your search criteria.
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default ClaimsExamplePage;