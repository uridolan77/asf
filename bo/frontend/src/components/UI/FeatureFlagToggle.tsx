import React from 'react';
import {
  Box,
  FormControlLabel,
  Switch,
  Typography,
  Paper,
  Divider,
  IconButton,
  Tooltip
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import { useFeatureFlags } from '../../context/FeatureFlagContext';

interface FeatureFlagToggleProps {
  showTitle?: boolean;
}

/**
 * FeatureFlagToggle component
 * 
 * Displays toggles for feature flags with descriptions.
 * Can be used in development mode to easily toggle features.
 */
const FeatureFlagToggle: React.FC<FeatureFlagToggleProps> = ({ showTitle = true }) => {
  const { flags, setFlag } = useFeatureFlags();
  
  // Feature flag descriptions
  const flagDescriptions: Record<string, string> = {
    useMockData: 'Use mock data instead of real API calls',
    enableReactQuery: 'Use React Query for data fetching',
    enableNewUI: 'Enable new UI components and layouts',
    enableBetaFeatures: 'Enable beta features that are still in development',
    enablePerformanceMonitoring: 'Enable performance monitoring and logging'
  };
  
  // Handle toggle change
  const handleToggle = (flag: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setFlag(flag as keyof typeof flags, event.target.checked);
  };
  
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      {showTitle && (
        <>
          <Typography variant="h6" gutterBottom>
            Feature Flags
            <Tooltip title="These flags control various features of the application. They are only visible in development mode.">
              <IconButton size="small" sx={{ ml: 1 }}>
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Typography>
          <Divider sx={{ mb: 2 }} />
        </>
      )}
      
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {Object.entries(flags).map(([flag, value]) => (
          <FormControlLabel
            key={flag}
            control={
              <Switch
                checked={value}
                onChange={handleToggle(flag)}
                color="primary"
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography variant="body2">
                  {flag}
                </Typography>
                <Tooltip title={flagDescriptions[flag] || 'No description available'}>
                  <IconButton size="small" sx={{ ml: 0.5 }}>
                    <InfoIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            }
          />
        ))}
      </Box>
    </Paper>
  );
};

export default FeatureFlagToggle;
