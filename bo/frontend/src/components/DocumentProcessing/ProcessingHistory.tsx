import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  Grid,
  Paper,
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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Pagination,
  SelectChangeEvent
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassEmpty as HourglassEmptyIcon,
  Visibility as VisibilityIcon,
  Science as ScienceIcon,
  LocalHospital as LocalHospitalIcon,
  Biotech as BiotechIcon,
  MedicalInformation as MedicalInformationIcon,
  Share as ShareIcon
} from '@mui/icons-material';

import { useMedicalResearchSynthesis, SynthesisResult } from '../../hooks/useMedicalResearchSynthesis';
import { useFeatureFlags } from '../../context/FeatureFlagContext';
import KnowledgeGraphViewer from './KnowledgeGraphViewer';

interface Task {
  task_id: string;
  file_name: string;
  status: string;
  created_at: string;
  completed_at?: string;
  processing_time?: number;
  entity_count?: number;
  relation_count?: number;
}

interface TaskResults {
  task_id: string;
  file_name: string;
  processing_time: number;
  entity_count: number;
  relation_count: number;
  results: SynthesisResult;
}

/**
 * Component for viewing document processing history
 */
const ProcessingHistory: React.FC = () => {
  // Feature flags
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Helper function to prepare graph data for visualization
  const prepareGraphData = (results?: SynthesisResult) => {
    if (!results || !results.entities || !results.relations) {
      return { nodes: [], links: [] };
    }

    // Create nodes from entities
    const nodes = results.entities.map(entity => ({
      id: entity.text,
      type: entity.label,
      label: entity.text,
      cui: entity.umls_id || null
    }));

    // Create links from relations
    const links = results.relations.map(relation => ({
      source: relation.head,
      target: relation.tail,
      type: relation.relation,
      confidence: relation.confidence || 0.5
    }));

    return { nodes, links };
  };

  // State
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [page, setPage] = useState<number>(1);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState<boolean>(false);
  const [dialogTab, setDialogTab] = useState<number>(0);

  const pageSize = 10;

  // Use the medical research synthesis hook
  const {
    getSynthesisHistory,
    getSynthesisResult
  } = useMedicalResearchSynthesis();

  // Get synthesis history
  const {
    data: historyData,
    isLoading: isLoadingHistory,
    isError: isErrorHistory,
    error: historyError,
    refetch: refetchHistory
  } = getSynthesisHistory();

  // Get synthesis result for selected task
  const {
    data: resultData,
    isLoading: isLoadingResult,
    isError: isErrorResult,
    error: resultError
  } = getSynthesisResult(selectedTask || '');

  // Filter and paginate tasks
  const filteredTasks = historyData?.documents
    ? historyData.documents
      .filter((task: any) => statusFilter === 'all' || task.status === statusFilter)
      .slice((page - 1) * pageSize, page * pageSize)
    : [];

  // Calculate total pages
  const totalPages = historyData?.documents
    ? Math.ceil(historyData.documents.length / pageSize)
    : 1;

  // Handle status filter change
  const handleStatusFilterChange = (event: SelectChangeEvent<string>) => {
    setStatusFilter(event.target.value);
    setPage(1); // Reset to first page when filter changes
  };

  // Handle page change
  const handlePageChange = (_event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
  };

  // View task results
  const handleViewResults = (taskId: string) => {
    setSelectedTask(taskId);
    setDialogOpen(true);
  };

  // Close dialog
  const handleCloseDialog = () => {
    setDialogOpen(false);
    setSelectedTask(null);
    setDialogTab(0);
  };

  // Handle dialog tab change
  const handleDialogTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setDialogTab(newValue);
  };

  // Get status chip for a task
  const getStatusChip = (status: string) => {
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
    if (!resultData || !resultData.entities) {
      return <Typography>No entities found</Typography>;
    }

    const entities = resultData.entities;
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
    if (!resultData || !resultData.relations) {
      return <Typography>No relations found</Typography>;
    }

    const relations = resultData.relations;

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
    if (!resultData || !resultData.summary) {
      return <Typography>No summary available</Typography>;
    }

    const summary = resultData.summary;

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

  if (useMockData) {
    return (
      <Alert severity="info">
        Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
      </Alert>
    );
  }

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
          onClick={() => refetchHistory()}
          disabled={isLoadingHistory}
        >
          Refresh
        </Button>
      </Box>

      {isErrorHistory && (
        <Alert 
          severity="error" 
          sx={{ mb: 3 }}
          action={
            <Button color="inherit" size="small" onClick={() => refetchHistory()}>
              Retry
            </Button>
          }
        >
          Error loading processing history: {historyError?.message || 'Unknown error'}
        </Alert>
      )}

      <Paper>
        {isLoadingHistory ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : filteredTasks.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body1" color="text.secondary">
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
                {filteredTasks.map((task: any) => (
                  <TableRow key={task.id}>
                    <TableCell>{task.title}</TableCell>
                    <TableCell>{getStatusChip(task.status || 'completed')}</TableCell>
                    <TableCell>
                      {task.processed_at ? new Date(task.processed_at).toLocaleString() : 'N/A'}
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
                      <Tooltip title="View Results">
                        <IconButton
                          color="primary"
                          onClick={() => handleViewResults(task.id)}
                        >
                          <VisibilityIcon />
                        </IconButton>
                      </Tooltip>
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
            disabled={isLoadingHistory}
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
          {resultData && (
            <Typography variant="subtitle2" color="text.secondary">
              {resultData.title}
            </Typography>
          )}
        </DialogTitle>

        <DialogContent dividers>
          {isLoadingResult ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : isErrorResult ? (
            <Alert severity="error">
              Error loading results: {resultError?.message || 'Unknown error'}
            </Alert>
          ) : resultData ? (
            <Box>
              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary">
                        Processing Time
                      </Typography>
                      <Typography variant="h5">
                        {resultData.processing_time ? `${resultData.processing_time.toFixed(2)}s` : 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary">
                        Entities Extracted
                      </Typography>
                      <Typography variant="h5">
                        {resultData.entity_count || 0}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary">
                        Relations Extracted
                      </Typography>
                      <Typography variant="h5">
                        {resultData.relation_count || 0}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="subtitle2" color="text.secondary">
                        Document Title
                      </Typography>
                      <Typography variant="body1" noWrap>
                        {resultData.title || 'Untitled Document'}
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
                </Tabs>
              </Box>

              <Box hidden={dialogTab !== 0} id="tabpanel-0" aria-labelledby="tab-0">
                {dialogTab === 0 && renderSummary()}
              </Box>

              <Box hidden={dialogTab !== 1} id="tabpanel-1" aria-labelledby="tab-1">
                {dialogTab === 1 && renderEntities()}
              </Box>

              <Box hidden={dialogTab !== 2} id="tabpanel-2" aria-labelledby="tab-2">
                {dialogTab === 2 && renderRelations()}
              </Box>

              <Box hidden={dialogTab !== 3} id="tabpanel-3" aria-labelledby="tab-3">
                {dialogTab === 3 && (
                  <Box sx={{ height: 500 }}>
                    <KnowledgeGraphViewer graphData={prepareGraphData(resultData)} />
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
