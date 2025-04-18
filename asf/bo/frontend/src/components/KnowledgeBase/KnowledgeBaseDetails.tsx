import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Button,
  IconButton,
  Tooltip,
  CircularProgress,
  Alert,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Tabs,
  Tab
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Search as SearchIcon,
  Add as AddIcon,
  ArrowBack as ArrowBackIcon
} from '@mui/icons-material';

import { useKnowledgeBase } from '../../hooks/useKnowledgeBase';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import { ButtonLoader } from '../UI/LoadingIndicators';

interface KnowledgeBaseDetailsProps {
  knowledgeBaseId: string;
  onBack?: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`kb-tabpanel-${index}`}
      aria-labelledby={`kb-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

/**
 * KnowledgeBaseDetails Component
 * 
 * Displays detailed information about a knowledge base, including articles and concepts.
 */
const KnowledgeBaseDetails: React.FC<KnowledgeBaseDetailsProps> = ({ 
  knowledgeBaseId,
  onBack
}) => {
  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // State for tabs
  const [tabValue, setTabValue] = useState<number>(0);

  // State for dialogs
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<boolean>(false);
  const [editDialogOpen, setEditDialogOpen] = useState<boolean>(false);
  
  // Form state
  const [formData, setFormData] = useState<{
    name: string;
    query: string;
    update_schedule: string;
  }>({
    name: '',
    query: '',
    update_schedule: 'daily'
  });

  // Use the knowledge base hook
  const {
    getKnowledgeBaseDetails,
    updateKnowledgeBase,
    deleteKnowledgeBase,
    exportKnowledgeBase
  } = useKnowledgeBase();

  // Get knowledge base details
  const {
    data: knowledgeBase,
    isLoading,
    isError,
    error,
    refetch
  } = getKnowledgeBaseDetails(knowledgeBaseId);

  // Get the update mutation function
  const { 
    mutate: updateMutate, 
    isPending: isUpdating 
  } = updateKnowledgeBase(knowledgeBaseId);

  // Get the delete mutation function
  const { 
    mutate: deleteMutate, 
    isPending: isDeleting 
  } = deleteKnowledgeBase(knowledgeBaseId);

  // Get the export mutation function
  const { 
    mutate: exportMutate, 
    isPending: isExporting 
  } = exportKnowledgeBase(knowledgeBaseId, 'json');

  // Handle tab change
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Handle form input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // Handle select changes
  const handleSelectChange = (e: SelectChangeEvent<string>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  // Handle edit dialog open
  const handleOpenEditDialog = () => {
    if (knowledgeBase) {
      setFormData({
        name: knowledgeBase.name,
        query: knowledgeBase.query,
        update_schedule: knowledgeBase.update_schedule
      });
      setEditDialogOpen(true);
    }
  };

  // Handle edit dialog close
  const handleCloseEditDialog = () => {
    setEditDialogOpen(false);
  };

  // Handle update knowledge base
  const handleUpdateKnowledgeBase = () => {
    updateMutate(formData);
    setEditDialogOpen(false);
  };

  // Handle delete dialog open
  const handleOpenDeleteDialog = () => {
    setDeleteDialogOpen(true);
  };

  // Handle delete dialog close
  const handleCloseDeleteDialog = () => {
    setDeleteDialogOpen(false);
  };

  // Handle delete knowledge base
  const handleDeleteKnowledgeBase = () => {
    deleteMutate({});
    setDeleteDialogOpen(false);
    if (onBack) {
      onBack();
    }
  };

  // Handle export knowledge base
  const handleExportKnowledgeBase = (format: string) => {
    exportMutate({ format });
  };

  // Format date
  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch (error) {
      return dateString;
    }
  };

  if (useMockData) {
    return (
      <Alert severity="info">
        Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
      </Alert>
    );
  }

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (isError) {
    return (
      <Alert 
        severity="error" 
        action={
          <Button color="inherit" size="small" onClick={() => refetch()}>
            Retry
          </Button>
        }
      >
        Error loading knowledge base details: {error?.message || 'Unknown error'}
      </Alert>
    );
  }

  if (!knowledgeBase) {
    return (
      <Alert severity="warning">
        Knowledge base not found.
      </Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {onBack && (
            <IconButton onClick={onBack} sx={{ mr: 1 }}>
              <ArrowBackIcon />
            </IconButton>
          )}
          <Typography variant="h6">{knowledgeBase.name}</Typography>
          <Chip 
            label={`${knowledgeBase.article_count} articles`} 
            size="small" 
            color="primary" 
            sx={{ ml: 1 }} 
          />
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetch()}
            size="small"
          >
            Refresh
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<SearchIcon />}
            href={`/search?kb=${knowledgeBaseId}`}
            size="small"
          >
            Search
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={() => handleExportKnowledgeBase('json')}
            disabled={isExporting}
            size="small"
          >
            Export
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<EditIcon />}
            onClick={handleOpenEditDialog}
            size="small"
          >
            Edit
          </Button>
          
          <Button
            variant="outlined"
            color="error"
            startIcon={<DeleteIcon />}
            onClick={handleOpenDeleteDialog}
            size="small"
          >
            Delete
          </Button>
        </Box>
      </Box>
      
      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="knowledge base tabs">
            <Tab label="Overview" id="kb-tab-0" aria-controls="kb-tabpanel-0" />
            <Tab label="Articles" id="kb-tab-1" aria-controls="kb-tabpanel-1" />
            <Tab label="Concepts" id="kb-tab-2" aria-controls="kb-tabpanel-2" />
          </Tabs>
        </Box>
        
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader title="Knowledge Base Information" />
                <Divider />
                <CardContent>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Typography variant="subtitle2">Query:</Typography>
                      <Typography variant="body2">{knowledgeBase.query}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Created At:</Typography>
                      <Typography variant="body2">{formatDate(knowledgeBase.created_at)}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Last Updated:</Typography>
                      <Typography variant="body2">{formatDate(knowledgeBase.last_updated)}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Update Schedule:</Typography>
                      <Typography variant="body2">{knowledgeBase.update_schedule}</Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="subtitle2">Article Count:</Typography>
                      <Typography variant="body2">{knowledgeBase.article_count}</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardHeader 
                  title="Top Concepts" 
                  action={
                    <Button
                      variant="text"
                      size="small"
                      endIcon={<ArrowBackIcon sx={{ transform: 'rotate(180deg)' }} />}
                      onClick={() => setTabValue(2)}
                    >
                      View All
                    </Button>
                  }
                />
                <Divider />
                <CardContent>
                  {knowledgeBase.concepts && knowledgeBase.concepts.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {knowledgeBase.concepts.slice(0, 10).map((concept) => (
                        <Chip
                          key={concept.id}
                          label={`${concept.name} (${concept.related_articles})`}
                          color="primary"
                          variant="outlined"
                          size="small"
                          onClick={() => window.open(`/search?query=${encodeURIComponent(concept.name)}`, '_blank')}
                        />
                      ))}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No concepts available.
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Articles ({knowledgeBase.article_count})</Typography>
            
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              size="small"
            >
              Add Articles
            </Button>
          </Box>
          
          {knowledgeBase.articles && knowledgeBase.articles.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Title</TableCell>
                    <TableCell>Journal</TableCell>
                    <TableCell>Year</TableCell>
                    <TableCell align="right">Relevance</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {knowledgeBase.articles.map((article) => (
                    <TableRow key={article.id}>
                      <TableCell>{article.title}</TableCell>
                      <TableCell>{article.journal}</TableCell>
                      <TableCell>{article.year}</TableCell>
                      <TableCell align="right">
                        <Chip
                          label={article.relevance_score.toFixed(2)}
                          size="small"
                          color={
                            article.relevance_score > 0.8 ? 'success' :
                            article.relevance_score > 0.5 ? 'primary' : 'default'
                          }
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title="View Article">
                          <IconButton size="small" href={`/article/${article.id}`}>
                            <SearchIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Remove from Knowledge Base">
                          <IconButton size="small">
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">
              No articles in this knowledge base. Add articles to get started.
            </Alert>
          )}
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Concepts</Typography>
          </Box>
          
          {knowledgeBase.concepts && knowledgeBase.concepts.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Concept</TableCell>
                    <TableCell align="right">Related Articles</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {knowledgeBase.concepts.map((concept) => (
                    <TableRow key={concept.id}>
                      <TableCell>{concept.name}</TableCell>
                      <TableCell align="right">{concept.related_articles}</TableCell>
                      <TableCell align="right">
                        <Tooltip title="Search Articles with this Concept">
                          <IconButton 
                            size="small" 
                            href={`/search?query=${encodeURIComponent(concept.name)}&kb=${knowledgeBaseId}`}
                          >
                            <SearchIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">
              No concepts found in this knowledge base.
            </Alert>
          )}
        </TabPanel>
      </Paper>
      
      {/* Edit Knowledge Base Dialog */}
      <Dialog open={editDialogOpen} onClose={handleCloseEditDialog} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Knowledge Base</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <TextField
              autoFocus
              margin="dense"
              name="name"
              label="Knowledge Base Name"
              type="text"
              fullWidth
              variant="outlined"
              value={formData.name}
              onChange={handleInputChange}
              sx={{ mb: 2 }}
            />
            
            <TextField
              margin="dense"
              name="query"
              label="Search Query"
              type="text"
              fullWidth
              variant="outlined"
              value={formData.query}
              onChange={handleInputChange}
              helperText="Enter a search query to populate this knowledge base"
              sx={{ mb: 2 }}
            />
            
            <FormControl fullWidth margin="dense">
              <InputLabel id="update-schedule-label">Update Schedule</InputLabel>
              <Select
                labelId="update-schedule-label"
                name="update_schedule"
                value={formData.update_schedule}
                label="Update Schedule"
                onChange={handleSelectChange}
              >
                <MenuItem value="hourly">Hourly</MenuItem>
                <MenuItem value="daily">Daily</MenuItem>
                <MenuItem value="weekly">Weekly</MenuItem>
                <MenuItem value="monthly">Monthly</MenuItem>
                <MenuItem value="manual">Manual Only</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseEditDialog}>Cancel</Button>
          <Button 
            onClick={handleUpdateKnowledgeBase} 
            variant="contained" 
            color="primary"
            disabled={!formData.name || !formData.query || isUpdating}
            startIcon={isUpdating ? <ButtonLoader size={20} /> : null}
          >
            {isUpdating ? 'Updating...' : 'Update'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={handleCloseDeleteDialog}>
        <DialogTitle>Delete Knowledge Base</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this knowledge base? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDeleteDialog}>Cancel</Button>
          <Button 
            onClick={handleDeleteKnowledgeBase} 
            color="error"
            disabled={isDeleting}
            startIcon={isDeleting ? <ButtonLoader size={20} /> : null}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default KnowledgeBaseDetails;
