import React, { useState } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Checkbox,
  TextField,
  InputAdornment,
  CircularProgress,
  Tabs,
  Tab
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';
import { useMetrics, useMetricCategories } from '../../hooks/useMetrics';

interface MetricSelectorProps {
  selectedMetrics: string[];
  onMetricToggle: (metricId: string) => void;
}

const MetricSelector: React.FC<MetricSelectorProps> = ({
  selectedMetrics,
  onMetricToggle
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  const { metrics, loading: metricsLoading } = useMetrics();
  const { categories, loading: categoriesLoading } = useMetricCategories();
  
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
  };
  
  const handleCategoryChange = (_event: React.SyntheticEvent, newValue: string | null) => {
    setSelectedCategory(newValue);
  };
  
  const filteredMetrics = metrics.filter(metric => {
    const matchesSearch = searchTerm === '' || 
      metric.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (metric.description && metric.description.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesCategory = selectedCategory === null || metric.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Metrics
      </Typography>
      
      <TextField
        fullWidth
        placeholder="Search metrics..."
        value={searchTerm}
        onChange={handleSearchChange}
        margin="normal"
        variant="outlined"
        size="small"
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
        }}
      />
      
      {!categoriesLoading && categories.length > 0 && (
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mt: 2 }}>
          <Tabs
            value={selectedCategory}
            onChange={handleCategoryChange}
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab label="All" value={null} />
            {categories.map(category => (
              <Tab key={category.id} label={category.name} value={category.id} />
            ))}
          </Tabs>
        </Box>
      )}
      
      {metricsLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress size={24} />
        </Box>
      ) : (
        <List sx={{ maxHeight: 300, overflow: 'auto' }}>
          {filteredMetrics.map(metric => (
            <ListItem key={metric.id} disablePadding>
              <ListItemButton
                dense
                onClick={() => onMetricToggle(metric.id)}
              >
                <ListItemIcon>
                  <Checkbox
                    edge="start"
                    checked={selectedMetrics.includes(metric.id)}
                    tabIndex={-1}
                    disableRipple
                  />
                </ListItemIcon>
                <ListItemText
                  primary={metric.name}
                  secondary={metric.description}
                  primaryTypographyProps={{
                    variant: 'body2',
                  }}
                  secondaryTypographyProps={{
                    variant: 'caption',
                    noWrap: true,
                  }}
                />
              </ListItemButton>
            </ListItem>
          ))}
          
          {filteredMetrics.length === 0 && (
            <ListItem>
              <ListItemText
                primary="No metrics found"
                secondary="Try adjusting your search or category filter"
                primaryTypographyProps={{
                  variant: 'body2',
                  align: 'center',
                }}
                secondaryTypographyProps={{
                  variant: 'caption',
                  align: 'center',
                }}
              />
            </ListItem>
          )}
        </List>
      )}
    </Box>
  );
};

export default MetricSelector;
