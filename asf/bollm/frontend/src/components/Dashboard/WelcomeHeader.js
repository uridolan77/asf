import React from 'react';
import { Paper, Box, Typography, Chip } from '@mui/material';

/**
 * Welcome header component for the dashboard
 * 
 * @param {Object} props - Component props
 * @param {Object} props.user - User object with username and role_id
 */
const WelcomeHeader = ({ user }) => {
  if (!user) return null;
  
  return (
    <Paper sx={{ p: 3, borderRadius: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Welcome back, {user.username}
        </Typography>
        <Chip 
          label={user.role_id === 2 ? 'Admin' : 'User'} 
          color={user.role_id === 2 ? 'secondary' : 'primary'}
          sx={{ fontWeight: 'bold' }}
        />
      </Box>
    </Paper>
  );
};

export default WelcomeHeader;
