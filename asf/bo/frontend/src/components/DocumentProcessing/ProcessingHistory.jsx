import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  TextField,
  Grid,
  Paper,
  Divider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Pagination
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as HourglassEmptyIcon,
  Visibility as VisibilityIcon,
  ExpandMore as ExpandMoreIcon,
  Science as ScienceIcon,
  LocalHospital as LocalHospitalIcon,
  Biotech as BiotechIcon,
  MedicalInformation as MedicalInformationIcon,
  Share as ShareIcon
} from '@mui/icons-material';
import apiService from '../../services/api';
import KnowledgeGraphViewer from './KnowledgeGraphViewer';

/**
 * Component for viewing document processing history
 */
const ProcessingHistory = () => {
  // Helper function to prepare graph data for visualization
  const prepareGraphData = (results) => {
    if (!results || !results.entities || !results.relations) {
      return { nodes: [], links: [] };
    }

    // Create nodes from entities
    const nodes = results.entities.map(entity => ({
      id: entity.text,
      type: entity.label,
      cui: entity.cui || null
    }));

    // Create links from relations
    const links = results.relations.map(relation => ({
      source: relation.head_entity || relation.head,
      target: relation.tail_entity || relation.tail,
      type: relation.relation_type || relation.relation,
      confidence: relation.confidence || 0.5,
      context: relation.context || ''
    }));

    return { nodes, links };
  };
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('all');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [selectedTask, setSelectedTask] = useState(null);
  const [taskResults, setTaskResults] = useState(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsError, setResultsError] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogTab, setDialogTab] = useState(0);

  const pageSize = 10;

  // Fetch tasks on component mount and when filters change
  useEffect(() => {
    fetchTasks();
  }, [statusFilter, page]);

  // Fetch tasks from API
  const fetchTasks = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.documentProcessing.getTasks(
        statusFilter !== 'all' ? statusFilter : null,
        pageSize,
        (page - 1) * pageSize
      );
      setTasks(response.data);

      // Calculate total pages (this would normally come from the API)
      // For now, we'll just assume there are more pages if we get a full page of results
      setTotalPages(response.data.length < pageSize ? page : page + 1);

      setLoading(false);
    } catch (err) {
      console.error('Error fetching tasks:', err);
      setError('Error fetching processing history. Please try again.');
      setLoading(false);
    }
  };

  // Handle status filter change
  const handleStatusFilterChange = (event) => {
    setStatusFilter(event.target.value);
    setPage(1); // Reset to first page when filter changes
  };

  // Handle page change
  const handlePageChange = (event, value) => {
    setPage(value);
  };

  // View task results
  const handleViewResults = async (taskId) => {
    setSelectedTask(taskId);
    setResultsLoading(true);
    setResultsError(null);
    setDialogOpen(true);

    try {
      const response = await apiService.documentProcessing.getResults(taskId);
      setTaskResults(response.data);
      setResultsLoading(false);
    } catch (err) {
      console.error('Error fetching task results:', err);
      setResultsError('Error fetching task results. The task may not be completed yet.');
      setResultsLoading(false);
    }
  };

  // Close dialog
  const handleCloseDialog = () => {
    setDialogOpen(false);
    setSelectedTask(null);
    setTaskResults(null);
    setResultsError(null);
    setDialogTab(0);
  };

  // Handle dialog tab change
  const handleDialogTabChange = (event, newValue) => {
    setDialogTab(newValue);
  };

  // Get status chip for a task
  const getStatusChip = (status) => {
    switch (status) {
      case 'queued':
        return <Chip icon={<HourglassEmptyIcon />} label="Queued" color="default" size="small" />;
      case 'processing':
        return <Chip icon={<CircularProgress size={16} />} label="Processing" color="primary" size="small" />;
      case 'completed':
        return <Chip icon={<CheckCircleIcon />} label="Completed" color="success" size="small" />;
      case 'failed':
        return <Chip icon={<ErrorIcon />} label="Failed" color="error" size="small" />;
      default:
        return <Chip label={status} size="small" />;
    }
  };

  // Render entity list
  const renderEntities = () => {
    if (!taskResults || !taskResults.results || !taskResults.results.entities) {
      return <Typography>No entities found</Typography>;
    }

    const entities = taskResults.results.entities;
    const entityTypes = [...new Set(entities.map(entity => entity.label))];

    return (
      <Box>
        <Box sx={{ mb: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {entityTypes.map(type => (
            <Chip
              key={type}
              label={`${type} (${entities.filter(e => e.label === type).length})`}
              color="primary"
              variant="outlined"
            />
          ))}
        </Box>

        <List dense>
          {entities.slice(0, 10).map((entity, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                {entity.label === 'DISEASE' ? <LocalHospitalIcon color="error" /> :
                 entity.label === 'DRUG' ? <MedicalInformationIcon color="primary" /> :
                 entity.label === 'GENE' ? <BiotechIcon color="success" /> :
                 <ScienceIcon color="secondary" />}
              </ListItemIcon>
              <ListItemText
                primary={entity.text}
                secondary={entity.label}
              />
            </ListItem>
          ))}
          {entities.length > 10 && (
            <ListItem>
              <ListItemText
                primary={`... and ${entities.length - 10} more entities`}
                secondary="Showing first 10 entities only"
              />
            </ListItem>
          )}
        </List>
      </Box>
    );
  };

  // Render relations list
  const renderRelations = () => {
    if (!taskResults || !taskResults.results || !taskResults.results.relations) {
      return <Typography>No relations found</Typography>;
    }

    const relations = taskResults.results.relations;

    return (
      <List dense>
        {relations.slice(0, 10).map((relation, index) => (
          <ListItem key={index}>
            <ListItemText
              primary={`${relation.head} → ${relation.relation} → ${relation.tail}`}
              secondary={relation.confidence ? `Confidence: ${(relation.confidence * 100).toFixed(1)}%` : ''}
            />
          </ListItem>
        ))}
        {relations.length > 10 && (
          <ListItem>
            <ListItemText
              primary={`... and ${relations.length - 10} more relations`}
              secondary="Showing first 10 relations only"
            />
          </ListItem>
        )}
      </List>
    );
  };

  // Render summary
  const renderSummary = () => {
    if (!taskResults || !taskResults.results || !taskResults.results.summary) {
      return <Typography>No summary available</Typography>;
    }

    const summary = taskResults.results.summary;

    return (
      <Box>
        {summary.abstract && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>Abstract</Typography>
            <Typography variant="body2">{summary.abstract}</Typography>
          </Box>
        )}

        {summary.key_findings && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom>Key Findings</Typography>
            <Typography variant="body2">{summary.key_findings}</Typography>
          </Box>
        )}

        {summary.conclusion && (
          <Box>
            <Typography variant="subtitle1" gutterBottom>Conclusion</Typography>
            <Typography variant="body2">{summary.conclusion}</Typography>
          </Box>
        )}
      </Box>
    );
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <FormControl sx={{ minWidth: 150 }}>
          <InputLabel id="status-filter-label">Status</InputLabel>
          <Select
            labelId="status-filter-label"
            id="status-filter"
            value={statusFilter}
            label="Status"
            onChange={handleStatusFilterChange}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="queued">Queued</MenuItem>
            <MenuItem value="processing">Processing</MenuItem>
            <MenuItem value="completed">Completed</MenuItem>
            <MenuItem value="failed">Failed</MenuItem>
          </Select>
        </FormControl>

        <Button
          variant="outlined"
          color="primary"
          startIcon={<RefreshIcon />}
          onClick={fetchTasks}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : tasks.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body1" color="textSecondary">
              No processing tasks found
            </Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>File Name</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Completed</TableCell>
                  <TableCell>Processing Time</TableCell>
                  <TableCell>Entities</TableCell>
                  <TableCell>Relations</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {tasks.map((task) => (
                  <TableRow key={task.task_id}>
                    <TableCell>{task.file_name}</TableCell>
                    <TableCell>{getStatusChip(task.status)}</TableCell>
                    <TableCell>
                      {task.created_at ? new Date(task.created_at).toLocaleString() : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {task.completed_at ? new Date(task.completed_at).toLocaleString() : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {task.processing_time ? `${task.processing_time.toFixed(2)}s` : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {task.entity_count !== undefined ? task.entity_count : 'N/A'}
                    </TableCell>
                    <TableCell>
                      {task.relation_count !== undefined ? task.relation_count : 'N/A'}
                    </TableCell>
                    <TableCell align="right">
                      {task.status === 'completed' && (
                        <Tooltip title="View Results">
                          <IconButton
                            color="primary"
                            onClick={() => handleViewResults(task.task_id)}
                          >
                            <VisibilityIcon />
                          </IconButton>
                        </Tooltip>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <Pagination
            count={totalPages}
            page={page}
            onChange={handlePageChange}
            color="primary"
            disabled={loading}
          />
        </Box>
      </Paper>

      {/* Results Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Document Processing Results
          {taskResults && (
            <Typography variant="subtitle2" color="textSecondary">
              {taskResults.file_name}
            </Typography>
          )}
        </DialogTitle>

        <DialogContent dividers>
          {resultsLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : resultsError ? (
            <Alert severity="error">
              {resultsError}
            </Alert>
          ) : taskResults ? (
            <Box>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="textSecondary">
                        Processing Time
                      </Typography>
                      <Typography variant="h5">
                        {taskResults.processing_time ? `${taskResults.processing_time.toFixed(2)}s` : 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="textSecondary">
                        Entities Extracted
                      </Typography>
                      <Typography variant="h5">
                        {taskResults.entity_count || 0}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="textSecondary">
                        Relations Extracted
                      </Typography>
                      <Typography variant="h5">
                        {taskResults.relation_count || 0}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="textSecondary">
                        Document Title
                      </Typography>
                      <Typography variant="body1" noWrap>
                        {taskResults.results?.title || 'Untitled Document'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>

              <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                <Tabs value={dialogTab} onChange={handleDialogTabChange} aria-label="results tabs">
                  <Tab label="Summary" id="tab-0" aria-controls="tabpanel-0" />
                  <Tab label="Entities" id="tab-1" aria-controls="tabpanel-1" />
                  <Tab label="Relations" id="tab-2" aria-controls="tabpanel-2" />
                  <Tab label="Knowledge Graph" id="tab-3" aria-controls="tabpanel-3" icon={<ShareIcon fontSize="small" />} iconPosition="end" />
                  <Tab label="Raw JSON" id="tab-4" aria-controls="tabpanel-4" />
                </Tabs>
              </Box>

              <Box role="tabpanel" hidden={dialogTab !== 0} id="tabpanel-0" aria-labelledby="tab-0">
                {dialogTab === 0 && renderSummary()}
              </Box>

              <Box role="tabpanel" hidden={dialogTab !== 1} id="tabpanel-1" aria-labelledby="tab-1">
                {dialogTab === 1 && renderEntities()}
              </Box>

              <Box role="tabpanel" hidden={dialogTab !== 2} id="tabpanel-2" aria-labelledby="tab-2">
                {dialogTab === 2 && renderRelations()}
              </Box>

              <Box role="tabpanel" hidden={dialogTab !== 3} id="tabpanel-3" aria-labelledby="tab-3">
                {dialogTab === 3 && (
                  <Box sx={{ height: '600px' }}>
                    <KnowledgeGraphViewer
                      graphData={prepareGraphData(taskResults.results)}
                      isLoading={false}
                      error={null}
                    />
                  </Box>
                )}
              </Box>

              <Box role="tabpanel" hidden={dialogTab !== 4} id="tabpanel-4" aria-labelledby="tab-4">
                {dialogTab === 4 && (
                  <Box sx={{ maxHeight: '400px', overflow: 'auto' }}>
                    <pre>{JSON.stringify(taskResults.results, null, 2)}</pre>
                  </Box>
                )}
              </Box>
            </Box>
          ) : (
            <Typography>No results available</Typography>
          )}
        </DialogContent>

        <DialogActions>
          <Button onClick={handleCloseDialog}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ProcessingHistory;
