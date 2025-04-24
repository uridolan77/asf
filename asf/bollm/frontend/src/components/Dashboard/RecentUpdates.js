import React from 'react';
import { 
  Typography, List, ListItem, ListItemIcon, ListItemText 
} from '@mui/material';
import { TrendingUp as TrendingUpIcon } from '@mui/icons-material';

/**
 * Recent medical research updates component
 * 
 * @param {Object} props - Component props
 * @param {Array} props.updates - Array of update objects with title and date
 */
const RecentUpdates = ({ updates = [] }) => {
  // Default updates if none provided
  const defaultUpdates = [
    {
      title: "Procalcitonin-guided antibiotic therapy in CAP shows promising results",
      date: "Apr 2025"
    },
    {
      title: "New data on antibiotic resistance patterns in Streptococcus pneumoniae",
      date: "Mar 2025"
    },
    {
      title: "Post-COVID patterns in respiratory infections suggest modified treatment approaches",
      date: "Feb 2025"
    }
  ];

  const displayUpdates = updates.length > 0 ? updates : defaultUpdates;

  return (
    <>
      <Typography variant="subtitle1" gutterBottom fontWeight="bold">
        Recent Medical Research Updates:
      </Typography>
      <List dense>
        {displayUpdates.map((update, index) => (
          <ListItem key={index}>
            <ListItemIcon sx={{ minWidth: 36 }}>
              <TrendingUpIcon color="secondary" />
            </ListItemIcon>
            <ListItemText 
              primary={update.title} 
              secondary={update.date}
            />
          </ListItem>
        ))}
      </List>
    </>
  );
};

export default RecentUpdates;
