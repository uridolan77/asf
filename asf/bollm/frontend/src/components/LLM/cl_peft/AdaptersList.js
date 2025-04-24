import React, { useState } from 'react';
import {
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemAvatar,
  ListItemSecondaryAction,
  Avatar,
  Typography,
  Chip,
  Box,
  IconButton,
  TextField,
  InputAdornment,
  Tooltip,
  useTheme,
  Divider
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Search as SearchIcon,
  FilterList as FilterListIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as HourglassEmptyIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';

const AdaptersList = ({ adapters, selectedAdapter, onSelect }) => {
  const theme = useTheme();
  const [searchTerm, setSearchTerm] = useState('');

  const getStatusIcon = (status) => {
    switch (status) {
      case 'ready':
        return <CheckCircleIcon fontSize="small" sx={{ color: theme.palette.success.main }} />;
      case 'error':
        return <ErrorIcon fontSize="small" sx={{ color: theme.palette.error.main }} />;
      case 'initializing':
      case 'training':
        return <HourglassEmptyIcon fontSize="small" sx={{ color: theme.palette.warning.main }} />;
      default:
        return null;
    }
  };

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

  const filteredAdapters = adapters.filter(adapter =>
    adapter.adapter_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    adapter.base_model_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    adapter.cl_strategy.toLowerCase().includes(searchTerm.toLowerCase()) ||
    adapter.peft_method.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (adapter.description && adapter.description.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <TextField
          fullWidth
          size="small"
          placeholder="Search adapters..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon fontSize="small" />
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <Tooltip title="Filter">
                  <IconButton size="small">
                    <FilterListIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </InputAdornment>
            )
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: theme.shape.borderRadius,
            }
          }}
        />
      </Box>

      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        <List sx={{ p: 0 }}>
          {filteredAdapters.length === 0 ? (
            <Box sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="body2" color="textSecondary">
                No adapters found
              </Typography>
            </Box>
          ) : (
            filteredAdapters.map((adapter) => (
              <ListItem
                key={adapter.adapter_id}
                disablePadding
                divider
                secondaryAction={
                  <Box sx={{ pr: 1 }}>
                    <Tooltip title={adapter.status}>
                      <Box>{getStatusIcon(adapter.status)}</Box>
                    </Tooltip>
                  </Box>
                }
                sx={{
                  borderLeft: selectedAdapter && selectedAdapter.adapter_id === adapter.adapter_id
                    ? `4px solid ${theme.palette.primary.main}`
                    : '4px solid transparent',
                  backgroundColor: selectedAdapter && selectedAdapter.adapter_id === adapter.adapter_id
                    ? theme.palette.action.selected
                    : 'transparent',
                }}
              >
                <ListItemButton
                  onClick={() => onSelect(adapter)}
                  sx={{ py: 1.5 }}
                >
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: theme.palette.primary.main }}>
                      <MemoryIcon />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Typography variant="subtitle2" noWrap fontWeight="medium">
                        {adapter.adapter_name}
                      </Typography>
                    }
                    secondary={
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 0.5 }}>
                        <Typography variant="caption" color="textSecondary" noWrap>
                          {adapter.base_model_name}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                          <Chip
                            label={adapter.cl_strategy}
                            size="small"
                            variant="outlined"
                            sx={{ height: 20, fontSize: '0.7rem' }}
                          />
                          <Chip
                            label={adapter.peft_method}
                            size="small"
                            variant="outlined"
                            sx={{ height: 20, fontSize: '0.7rem' }}
                          />
                        </Box>
                      </Box>
                    }
                  />
                </ListItemButton>
              </ListItem>
            ))
          )}
        </List>
      </Box>
    </Box>
  );
};

export default AdaptersList;
