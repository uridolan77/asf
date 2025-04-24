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
import { useDimensions, useDimensionCategories } from '../../hooks/useDimensions';

interface DimensionSelectorProps {
  selectedDimensions: string[];
  onDimensionToggle: (dimensionId: string) => void;
}

const DimensionSelector: React.FC<DimensionSelectorProps> = ({
  selectedDimensions,
  onDimensionToggle
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  const { dimensions, loading: dimensionsLoading } = useDimensions();
  const { categories, loading: categoriesLoading } = useDimensionCategories();
  
  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
  };
  
  const handleCategoryChange = (_event: React.SyntheticEvent, newValue: string | null) => {
    setSelectedCategory(newValue);
  };
  
  const filteredDimensions = dimensions.filter(dimension => {
    const matchesSearch = searchTerm === '' || 
      dimension.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (dimension.description && dimension.description.toLowerCase().includes(searchTerm.toLowerCase()));
    
    const matchesCategory = selectedCategory === null || dimension.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Dimensions
      </Typography>
      
      <TextField
        fullWidth
        placeholder="Search dimensions..."
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
      
      {dimensionsLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress size={24} />
        </Box>
      ) : (
        <List sx={{ maxHeight: 300, overflow: 'auto' }}>
          {filteredDimensions.map(dimension => (
            <ListItem key={dimension.id} disablePadding>
              <ListItemButton
                dense
                onClick={() => onDimensionToggle(dimension.id)}
              >
                <ListItemIcon>
                  <Checkbox
                    edge="start"
                    checked={selectedDimensions.includes(dimension.id)}
                    tabIndex={-1}
                    disableRipple
                  />
                </ListItemIcon>
                <ListItemText
                  primary={dimension.name}
                  secondary={dimension.description}
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
          
          {filteredDimensions.length === 0 && (
            <ListItem>
              <ListItemText
                primary="No dimensions found"
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

export default DimensionSelector;
