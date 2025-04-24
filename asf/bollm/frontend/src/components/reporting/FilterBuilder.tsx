import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Chip,
  IconButton,
  Divider,
  Paper,
  SelectChangeEvent
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';

import { useDimensions, useDimensionValues } from '../../hooks/useDimensions';
import { useMetrics } from '../../hooks/useMetrics';
import { Dimension, Metric } from '../../types/reporting';

interface FilterBuilderProps {
  filters: Record<string, any>;
  onFiltersChange: (filters: Record<string, any>) => void;
}

interface Filter {
  id: string;
  field: string;
  operator: string;
  value: any;
}

const FilterBuilder: React.FC<FilterBuilderProps> = ({
  filters,
  onFiltersChange
}) => {
  const [filtersList, setFiltersList] = useState<Filter[]>([]);
  const [selectedField, setSelectedField] = useState<string>('');
  const [selectedOperator, setSelectedOperator] = useState<string>('equals');
  const [filterValue, setFilterValue] = useState<string>('');
  
  const { dimensions } = useDimensions();
  const { metrics } = useMetrics();
  const { values: fieldValues } = useDimensionValues(selectedField);
  
  // Convert filters object to filters list on mount
  useEffect(() => {
    const newFiltersList: Filter[] = [];
    
    Object.entries(filters).forEach(([field, condition]) => {
      if (typeof condition === 'object') {
        Object.entries(condition).forEach(([operator, value]) => {
          newFiltersList.push({
            id: `${field}_${operator}_${Date.now()}`,
            field,
            operator: mapOperatorFromAPI(operator),
            value
          });
        });
      } else {
        newFiltersList.push({
          id: `${field}_equals_${Date.now()}`,
          field,
          operator: 'equals',
          value: condition
        });
      }
    });
    
    setFiltersList(newFiltersList);
  }, []);
  
  // Convert filters list to filters object when list changes
  useEffect(() => {
    const newFilters: Record<string, any> = {};
    
    filtersList.forEach(filter => {
      const apiOperator = mapOperatorToAPI(filter.operator);
      
      if (!newFilters[filter.field]) {
        if (filter.operator === 'equals') {
          newFilters[filter.field] = filter.value;
        } else {
          newFilters[filter.field] = { [apiOperator]: filter.value };
        }
      } else {
        if (typeof newFilters[filter.field] !== 'object') {
          // Convert simple equality to object
          const oldValue = newFilters[filter.field];
          newFilters[filter.field] = { 'eq': oldValue };
        }
        
        // Add new operator
        newFilters[filter.field][apiOperator] = filter.value;
      }
    });
    
    onFiltersChange(newFilters);
  }, [filtersList]);
  
  const mapOperatorToAPI = (operator: string): string => {
    switch (operator) {
      case 'equals': return 'eq';
      case 'not_equals': return 'ne';
      case 'greater_than': return 'gt';
      case 'greater_than_or_equals': return 'gte';
      case 'less_than': return 'lt';
      case 'less_than_or_equals': return 'lte';
      case 'contains': return 'contains';
      case 'starts_with': return 'startswith';
      case 'ends_with': return 'endswith';
      case 'in': return 'in';
      default: return operator;
    }
  };
  
  const mapOperatorFromAPI = (operator: string): string => {
    switch (operator) {
      case 'eq': return 'equals';
      case 'ne': return 'not_equals';
      case 'gt': return 'greater_than';
      case 'gte': return 'greater_than_or_equals';
      case 'lt': return 'less_than';
      case 'lte': return 'less_than_or_equals';
      case 'contains': return 'contains';
      case 'startswith': return 'starts_with';
      case 'endswith': return 'ends_with';
      case 'in': return 'in';
      default: return operator;
    }
  };
  
  const getFieldType = (fieldId: string): string => {
    const dimension = dimensions.find(d => d.id === fieldId);
    if (dimension) return dimension.data_type;
    
    const metric = metrics.find(m => m.id === fieldId);
    if (metric) return metric.data_type;
    
    return 'string';
  };
  
  const getFieldName = (fieldId: string): string => {
    const dimension = dimensions.find(d => d.id === fieldId);
    if (dimension) return dimension.name;
    
    const metric = metrics.find(m => m.id === fieldId);
    if (metric) return metric.name;
    
    return fieldId;
  };
  
  const getOperatorLabel = (operator: string): string => {
    switch (operator) {
      case 'equals': return 'Equals';
      case 'not_equals': return 'Not Equals';
      case 'greater_than': return 'Greater Than';
      case 'greater_than_or_equals': return 'Greater Than or Equals';
      case 'less_than': return 'Less Than';
      case 'less_than_or_equals': return 'Less Than or Equals';
      case 'contains': return 'Contains';
      case 'starts_with': return 'Starts With';
      case 'ends_with': return 'Ends With';
      case 'in': return 'In';
      default: return operator;
    }
  };
  
  const getAvailableOperators = (fieldType: string): string[] => {
    switch (fieldType) {
      case 'string':
        return ['equals', 'not_equals', 'contains', 'starts_with', 'ends_with', 'in'];
      case 'number':
      case 'integer':
      case 'float':
        return ['equals', 'not_equals', 'greater_than', 'greater_than_or_equals', 'less_than', 'less_than_or_equals', 'in'];
      case 'date':
      case 'datetime':
        return ['equals', 'not_equals', 'greater_than', 'greater_than_or_equals', 'less_than', 'less_than_or_equals'];
      case 'boolean':
        return ['equals', 'not_equals'];
      default:
        return ['equals', 'not_equals'];
    }
  };
  
  const handleFieldChange = (event: SelectChangeEvent) => {
    const fieldId = event.target.value;
    setSelectedField(fieldId);
    
    // Reset operator and value when field changes
    const fieldType = getFieldType(fieldId);
    const operators = getAvailableOperators(fieldType);
    setSelectedOperator(operators[0]);
    setFilterValue('');
  };
  
  const handleOperatorChange = (event: SelectChangeEvent) => {
    setSelectedOperator(event.target.value);
  };
  
  const handleValueChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFilterValue(event.target.value);
  };
  
  const handleAddFilter = () => {
    if (!selectedField || !selectedOperator) return;
    
    const newFilter: Filter = {
      id: `${selectedField}_${selectedOperator}_${Date.now()}`,
      field: selectedField,
      operator: selectedOperator,
      value: filterValue
    };
    
    setFiltersList([...filtersList, newFilter]);
    
    // Reset form
    setFilterValue('');
  };
  
  const handleRemoveFilter = (filterId: string) => {
    setFiltersList(filtersList.filter(filter => filter.id !== filterId));
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Filters
      </Typography>
      
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Field</InputLabel>
            <Select
              value={selectedField}
              onChange={handleFieldChange}
              label="Field"
            >
              <MenuItem value="" disabled>Select a field</MenuItem>
              
              {dimensions.length > 0 && [
                <MenuItem key="dimensions-header" disabled>
                  <Typography variant="caption">Dimensions</Typography>
                </MenuItem>,
                ...dimensions.map(dimension => (
                  <MenuItem key={dimension.id} value={dimension.id}>
                    {dimension.name}
                  </MenuItem>
                ))
              ]}
              
              {metrics.length > 0 && [
                <MenuItem key="metrics-header" disabled>
                  <Typography variant="caption">Metrics</Typography>
                </MenuItem>,
                ...metrics.map(metric => (
                  <MenuItem key={metric.id} value={metric.id}>
                    {metric.name}
                  </MenuItem>
                ))
              ]}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={3}>
          <FormControl fullWidth size="small" disabled={!selectedField}>
            <InputLabel>Operator</InputLabel>
            <Select
              value={selectedOperator}
              onChange={handleOperatorChange}
              label="Operator"
            >
              {getAvailableOperators(getFieldType(selectedField)).map(operator => (
                <MenuItem key={operator} value={operator}>
                  {getOperatorLabel(operator)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth size="small">
            <TextField
              label="Value"
              value={filterValue}
              onChange={handleValueChange}
              disabled={!selectedField || !selectedOperator}
              size="small"
            />
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={1}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleAddFilter}
            disabled={!selectedField || !selectedOperator || filterValue === ''}
            sx={{ minWidth: 0 }}
          >
            <AddIcon />
          </Button>
        </Grid>
      </Grid>
      
      {filtersList.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Divider sx={{ mb: 2 }} />
          
          <Typography variant="subtitle2" gutterBottom>
            Applied Filters
          </Typography>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {filtersList.map(filter => (
              <Chip
                key={filter.id}
                label={`${getFieldName(filter.field)} ${getOperatorLabel(filter.operator)} ${filter.value}`}
                onDelete={() => handleRemoveFilter(filter.id)}
                color="primary"
                variant="outlined"
              />
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default FilterBuilder;
