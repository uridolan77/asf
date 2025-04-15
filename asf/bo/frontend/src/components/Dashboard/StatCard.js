import React from 'react';
import { Card, CardContent, Typography, Button, Box } from '@mui/material';

/**
 * A reusable card component for displaying statistics on the dashboard
 * 
 * @param {Object} props - Component props
 * @param {string} props.title - Card title
 * @param {string|number} props.value - Main statistic value to display
 * @param {string} props.actionText - Optional text for action button
 * @param {Function} props.onAction - Optional callback for action button
 * @param {React.ReactNode} props.icon - Optional icon to display
 * @param {Object} props.sx - Optional additional styles
 */
const StatCard = ({ title, value, actionText, onAction, icon, sx = {} }) => {
  return (
    <Card sx={{ height: '100%', ...sx }}>
      <CardContent>
        <Typography color="textSecondary" gutterBottom>
          {title}
        </Typography>
        
        {icon ? (
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            {icon}
            <Typography variant="h3" component="div" sx={{ ml: 1 }}>
              {value || '--'}
            </Typography>
          </Box>
        ) : (
          <Typography variant="h3" component="div">
            {value || '--'}
          </Typography>
        )}
        
        {actionText && onAction && (
          <Button 
            variant="outlined" 
            size="small" 
            sx={{ mt: 2 }}
            onClick={onAction}
          >
            {actionText}
          </Button>
        )}
      </CardContent>
    </Card>
  );
};

export default StatCard;
