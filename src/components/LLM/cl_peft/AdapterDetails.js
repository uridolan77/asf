import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Chip,
  Divider,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  useTheme
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Code as CodeIcon,
  Settings as SettingsIcon,
  CalendarToday as CalendarTodayIcon,
  Update as UpdateIcon,
  Label as LabelIcon
} from '@mui/icons-material';
import { format } from 'date-fns';

/**
 * AdapterDetails component displays detailed information about a CL-PEFT adapter
 */
const AdapterDetails = ({ adapter }) => {
  const theme = useTheme();
  
  if (!adapter) {
    return null;
  }
  
  // Format date
  const formatDate = (dateString) => {
    try {
      return format(new Date(dateString), 'MMM d, yyyy HH:mm:ss');
    } catch (error) {
      return dateString;
    }
  };
  
  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'ready':
        return theme.palette.success.main;
      case 'error':
        return theme.palette.error.main;
      case 'initializing':
      case 'training':
        return theme.palette.warning.main;
      default:
        return theme.palette.grey[500];
    }
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Adapter Details
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Basic Information
              </Typography>
              <List dense disablePadding>
                <ListItem>
                  <ListItemIcon>
                    <MemoryIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary="Adapter ID"
                    secondary={adapter.adapter_id}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <StorageIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary="Base Model"
                    secondary={adapter.base_model_name}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CodeIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary="PEFT Method"
                    secondary={adapter.peft_method}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <SettingsIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary="CL Strategy"
                    secondary={adapter.cl_strategy}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CalendarTodayIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary="Created At"
                    secondary={formatDate(adapter.created_at)}
                  />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <UpdateIcon />
                  </ListItemIcon>
                  <ListItemText
                    primary="Last Updated"
                    secondary={formatDate(adapter.updated_at)}
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Status & Statistics
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Current Status:
                </Typography>
                <Chip 
                  label={adapter.status} 
                  sx={{ 
                    bgcolor: getStatusColor(adapter.status),
                    color: 'white'
                  }}
                />
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="body2" gutterBottom>
                Task History:
              </Typography>
              <Typography variant="body1">
                {adapter.task_history?.length || 0} tasks
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="body2" gutterBottom>
                Tags:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {adapter.tags && adapter.tags.length > 0 ? (
                  adapter.tags.map((tag, index) => (
                    <Chip 
                      key={index}
                      label={tag} 
                      size="small" 
                      icon={<LabelIcon />}
                      variant="outlined"
                    />
                  ))
                ) : (
                  <Typography variant="body2" color="textSecondary">
                    No tags
                  </Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Description
              </Typography>
              <Typography variant="body1">
                {adapter.description || "No description provided."}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AdapterDetails;
