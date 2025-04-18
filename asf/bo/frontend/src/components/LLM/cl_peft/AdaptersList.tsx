import React, { useState } from 'react';
import {
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Typography,
  Chip,
  Box,
  IconButton,
  TextField,
  InputAdornment,
  Tooltip,
  useTheme,
  Divider,
  CircularProgress,
  Button
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Search as SearchIcon,
  FilterList as FilterListIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as HourglassEmptyIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import { useCLPEFT, Adapter } from '../../../hooks/useCLPEFT';
import { useFeatureFlags } from '../../../context/FeatureFlagContext';

interface AdaptersListProps {
  selectedAdapter?: Adapter;
  onSelect: (adapter: Adapter) => void;
}

const AdaptersList: React.FC<AdaptersListProps> = ({ selectedAdapter, onSelect }) => {
  const theme = useTheme();
  const [searchTerm, setSearchTerm] = useState<string>('');
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Use the CL-PEFT hook
  const {
    adapters,
    isLoadingAdapters,
    isErrorAdapters,
    errorAdapters,
    refetchAdapters
  } = useCLPEFT();

  const getStatusIcon = (status: string) => {
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

  const filteredAdapters = adapters.filter(adapter =>
    adapter.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    adapter.base_model.toLowerCase().includes(searchTerm.toLowerCase()) ||
    adapter.cl_strategy.toLowerCase().includes(searchTerm.toLowerCase()) ||
    adapter.adapter_type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}`, display: 'flex', alignItems: 'center' }}>
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
        <Tooltip title="Refresh adapters">
          <IconButton
            size="small"
            onClick={() => refetchAdapters()}
            sx={{ ml: 1 }}
            disabled={isLoadingAdapters}
          >
            {isLoadingAdapters ? <CircularProgress size={20} /> : <RefreshIcon />}
          </IconButton>
        </Tooltip>
      </Box>

      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        {isLoadingAdapters && (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 100 }}>
            <CircularProgress size={24} />
            <Typography variant="body2" sx={{ ml: 1 }}>
              Loading adapters...
            </Typography>
          </Box>
        )}

        {isErrorAdapters && (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="error">
              Error loading adapters: {errorAdapters?.message}
            </Typography>
            <Button
              variant="outlined"
              size="small"
              onClick={() => refetchAdapters()}
              sx={{ mt: 1 }}
            >
              Retry
            </Button>
          </Box>
        )}

        {!isLoadingAdapters && !isErrorAdapters && (
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
                  key={adapter.id}
                  disablePadding
                  divider
                  sx={{
                    borderLeft: selectedAdapter && selectedAdapter.id === adapter.id
                      ? `4px solid ${theme.palette.primary.main}`
                      : '4px solid transparent',
                    backgroundColor: selectedAdapter && selectedAdapter.id === adapter.id
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
                          {adapter.name}
                        </Typography>
                      }
                      secondary={
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 0.5 }}>
                          <Typography variant="caption" color="textSecondary" noWrap>
                            {adapter.base_model}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 0.5 }}>
                            <Chip
                              label={adapter.cl_strategy}
                              size="small"
                              variant="outlined"
                              sx={{ height: 20, fontSize: '0.7rem' }}
                            />
                            <Chip
                              label={adapter.adapter_type}
                              size="small"
                              variant="outlined"
                              sx={{ height: 20, fontSize: '0.7rem' }}
                            />
                          </Box>
                        </Box>
                      }
                    />
                    <Box sx={{ ml: 1 }}>
                      <Tooltip title={adapter.status}>
                        {getStatusIcon(adapter.status)}
                      </Tooltip>
                    </Box>
                  </ListItemButton>
                </ListItem>
              ))
            )}
          </List>
        )}
      </Box>
    </Box>
  );
};

export default AdaptersList;
