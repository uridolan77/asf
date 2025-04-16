import React, { useState, useEffect } from 'react';
import {
  Box, Button, Card, CardContent, CircularProgress, Container,
  Dialog, DialogActions, DialogContent, DialogTitle,
  Divider, FormControl, Grid, IconButton, InputLabel,
  List, ListItem, ListItemIcon, ListItemText, MenuItem,
  Paper, Select, Tab, Tabs, TextField, Typography, Alert,
  Chip, Tooltip, LinearProgress
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Upload as UploadIcon,
  FileCopy as FileIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import { useNotification } from '../context/NotificationContext';
import PageLayout from '../components/Layout/PageLayout';

const KnowledgeBasePage = () => {
  const { user, api } = useAuth();
  const { showSuccess, showError } = useNotification();

  // State for knowledge base list and selected kb
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [selectedKnowledgeBase, setSelectedKnowledgeBase] = useState(null);
  const [loading, setLoading] = useState(true);
  const [documents, setDocuments] = useState([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [tabValue, setTabValue] = useState(0);

  // State for modals
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [searchDialogOpen, setSearchDialogOpen] = useState(false);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);

  // Form states
  const [newKbName, setNewKbName] = useState('');
  const [newKbDescription, setNewKbDescription] = useState('');
  const [newKbType, setNewKbType] = useState('document');
  const [editKbName, setEditKbName] = useState('');
  const [editKbDescription, setEditKbDescription] = useState('');
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  
  // Config states
  const [embeddingModel, setEmbeddingModel] = useState('text-embedding-ada-002');
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.75);

  // Fetch knowledge bases on component mount
  useEffect(() => {
    fetchKnowledgeBases();
  }, []);

  // Fetch knowledge bases from API
  const fetchKnowledgeBases = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/knowledge-base/list');
      setKnowledgeBases(response.data);
      if (response.data.length > 0 && !selectedKnowledgeBase) {
        setSelectedKnowledgeBase(response.data[0]);
        fetchDocuments(response.data[0].id);
      }
    } catch (error) {
      showError('Failed to fetch knowledge bases: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  // Fetch documents for a knowledge base
  const fetchDocuments = async (kbId) => {
    setDocumentsLoading(true);
    try {
      const response = await api.get(`/api/knowledge-base/${kbId}/documents`);
      setDocuments(response.data);
    } catch (error) {
      showError('Failed to fetch documents: ' + (error.response?.data?.detail || error.message));
    } finally {
      setDocumentsLoading(false);
    }
  };

  // Handle knowledge base selection
  const handleKnowledgeBaseSelect = (kb) => {
    setSelectedKnowledgeBase(kb);
    fetchDocuments(kb.id);
    setTabValue(0);
  };

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Create a new knowledge base
  const handleCreateKnowledgeBase = async () => {
    try {
      const response = await api.post('/api/knowledge-base/create', {
        name: newKbName,
        description: newKbDescription,
        type: newKbType
      });
      
      setKnowledgeBases([...knowledgeBases, response.data]);
      setSelectedKnowledgeBase(response.data);
      setCreateDialogOpen(false);
      setNewKbName('');
      setNewKbDescription('');
      setNewKbType('document');
      showSuccess('Knowledge base created successfully');
      fetchDocuments(response.data.id);
    } catch (error) {
      showError('Failed to create knowledge base: ' + (error.response?.data?.detail || error.message));
    }
  };

  // Open edit dialog with current KB data
  const openEditDialog = () => {
    if (selectedKnowledgeBase) {
      setEditKbName(selectedKnowledgeBase.name);
      setEditKbDescription(selectedKnowledgeBase.description || '');
      setEditDialogOpen(true);
    }
  };

  // Update knowledge base
  const handleUpdateKnowledgeBase = async () => {
    try {
      const response = await api.put(`/api/knowledge-base/${selectedKnowledgeBase.id}`, {
        name: editKbName,
        description: editKbDescription
      });
      
      const updatedKbs = knowledgeBases.map(kb => 
        kb.id === selectedKnowledgeBase.id ? response.data : kb
      );
      
      setKnowledgeBases(updatedKbs);
      setSelectedKnowledgeBase(response.data);
      setEditDialogOpen(false);
      showSuccess('Knowledge base updated successfully');
    } catch (error) {
      showError('Failed to update knowledge base: ' + (error.response?.data?.detail || error.message));
    }
  };

  // Delete knowledge base
  const handleDeleteKnowledgeBase = async () => {
    try {
      await api.delete(`/api/knowledge-base/${selectedKnowledgeBase.id}`);
      
      const updatedKbs = knowledgeBases.filter(kb => kb.id !== selectedKnowledgeBase.id);
      setKnowledgeBases(updatedKbs);
      setSelectedKnowledgeBase(updatedKbs.length > 0 ? updatedKbs[0] : null);
      setDeleteDialogOpen(false);
      showSuccess('Knowledge base deleted successfully');
      
      if (updatedKbs.length > 0) {
        fetchDocuments(updatedKbs[0].id);
      } else {
        setDocuments([]);
      }
    } catch (error) {
      showError('Failed to delete knowledge base: ' + (error.response?.data?.detail || error.message));
    }
  };

  // Handle file selection for upload
  const handleFileChange = (event) => {
    setUploadFile(event.target.files[0]);
  };

  // Upload document
  const handleUploadDocument = async () => {
    if (!uploadFile || !selectedKnowledgeBase) return;
    
    setUploading(true);
    setUploadProgress(0);
    
    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', uploadFile);
      
      // Upload with progress tracking
      const response = await api.post(
        `/api/knowledge-base/${selectedKnowledgeBase.id}/upload`, 
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
        }
      );
      
      // Refresh documents list
      fetchDocuments(selectedKnowledgeBase.id);
      setUploadDialogOpen(false);
      setUploadFile(null);
      showSuccess('Document uploaded successfully');
    } catch (error) {
      showError('Failed to upload document: ' + (error.response?.data?.detail || error.message));
    } finally {
      setUploading(false);
    }
  };

  // Delete document
  const handleDeleteDocument = async (documentId) => {
    try {
      await api.delete(`/api/knowledge-base/${selectedKnowledgeBase.id}/documents/${documentId}`);
      
      // Refresh documents list
      fetchDocuments(selectedKnowledgeBase.id);
      showSuccess('Document deleted successfully');
    } catch (error) {
      showError('Failed to delete document: ' + (error.response?.data?.detail || error.message));
    }
  };

  // Search within knowledge base
  const handleSearch = async () => {
    if (!searchQuery.trim() || !selectedKnowledgeBase) return;
    
    setSearching(true);
    try {
      const response = await api.post(`/api/knowledge-base/${selectedKnowledgeBase.id}/search`, {
        query: searchQuery,
        limit: 10
      });
      
      setSearchResults(response.data.results || []);
    } catch (error) {
      showError('Search failed: ' + (error.response?.data?.detail || error.message));
    } finally {
      setSearching(false);
    }
  };

  // Update knowledge base config
  const handleUpdateConfig = async () => {
    try {
      await api.post(`/api/knowledge-base/${selectedKnowledgeBase.id}/config`, {
        embedding_model: embeddingModel,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        similarity_threshold: similarityThreshold
      });
      
      setConfigDialogOpen(false);
      showSuccess('Configuration updated successfully');
    } catch (error) {
      showError('Failed to update configuration: ' + (error.response?.data?.detail || error.message));
    }
  };

  // Fetch knowledge base configuration
  const fetchKnowledgeBaseConfig = async () => {
    try {
      const response = await api.get(`/api/knowledge-base/${selectedKnowledgeBase.id}/config`);
      setEmbeddingModel(response.data.embedding_model || 'text-embedding-ada-002');
      setChunkSize(response.data.chunk_size || 1000);
      setChunkOverlap(response.data.chunk_overlap || 200);
      setSimilarityThreshold(response.data.similarity_threshold || 0.75);
      setConfigDialogOpen(true);
    } catch (error) {
      showError('Failed to fetch configuration: ' + (error.response?.data?.detail || error.message));
    }
  };

  // Get file size in human-readable format
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get document status chip
  const getStatusChip = (status) => {
    if (status === 'processed') {
      return <Chip label="Processed" color="success" size="small" />;
    } else if (status === 'processing') {
      return <Chip label="Processing" color="warning" size="small" />;
    } else if (status === 'failed') {
      return <Chip label="Failed" color="error" size="small" />;
    }
    return <Chip label="Pending" color="default" size="small" />;
  };

  return (
    <PageLayout
      title="Knowledge Base Management"
      breadcrumbs={[{ label: 'Knowledge Bases', path: '/knowledge-base' }]}
      loading={false}
      user={user}
    >
      <Grid container spacing={3}>
        {/* Knowledge Base Sidebar */}
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Knowledge Bases</Typography>
              <Button
                variant="contained"
                color="primary"
                startIcon={<AddIcon />}
                onClick={() => setCreateDialogOpen(true)}
                size="small"
              >
                New
              </Button>
            </Box>
            
            <Divider sx={{ mb: 2 }} />
            
            {loading ? (
              <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />
            ) : knowledgeBases.length === 0 ? (
              <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                No knowledge bases found. Create one to get started.
              </Typography>
            ) : (
              <List>
                {knowledgeBases.map((kb) => (
                  <ListItem
                    key={kb.id}
                    button
                    selected={selectedKnowledgeBase?.id === kb.id}
                    onClick={() => handleKnowledgeBaseSelect(kb)}
                    sx={{ borderRadius: 1, mb: 0.5 }}
                  >
                    <ListItemIcon>
                      <FileIcon />
                    </ListItemIcon>
                    <ListItemText 
                      primary={kb.name}
                      secondary={kb.type.charAt(0).toUpperCase() + kb.type.slice(1)}
                      primaryTypographyProps={{ noWrap: true }}
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Paper>
        </Grid>
        
        {/* Main Content */}
        <Grid item xs={12} md={9}>
          {selectedKnowledgeBase ? (
            <Paper sx={{ p: 2 }}>
              {/* Knowledge Base Header */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Box>
                  <Typography variant="h5">{selectedKnowledgeBase.name}</Typography>
                  {selectedKnowledgeBase.description && (
                    <Typography variant="body2" color="text.secondary">
                      {selectedKnowledgeBase.description}
                    </Typography>
                  )}
                </Box>
                
                <Box>
                  <Tooltip title="Upload Document">
                    <IconButton
                      color="primary"
                      onClick={() => setUploadDialogOpen(true)}
                      sx={{ mr: 1 }}
                    >
                      <UploadIcon />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Search Knowledge Base">
                    <IconButton
                      color="primary"
                      onClick={() => setSearchDialogOpen(true)}
                      sx={{ mr: 1 }}
                    >
                      <SearchIcon />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Configure">
                    <IconButton
                      color="primary"
                      onClick={fetchKnowledgeBaseConfig}
                      sx={{ mr: 1 }}
                    >
                      <SettingsIcon />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Edit">
                    <IconButton
                      color="primary"
                      onClick={openEditDialog}
                      sx={{ mr: 1 }}
                    >
                      <EditIcon />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Delete">
                    <IconButton
                      color="error"
                      onClick={() => setDeleteDialogOpen(true)}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              
              <Divider sx={{ mb: 2 }} />
              
              {/* Tabs */}
              <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                <Tabs value={tabValue} onChange={handleTabChange}>
                  <Tab label="Documents" />
                  <Tab label="Statistics" />
                </Tabs>
              </Box>
              
              {/* Documents Tab */}
              {tabValue === 0 && (
                <>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1">
                      {documents.length} document(s) in this knowledge base
                    </Typography>
                    
                    <Button
                      startIcon={<RefreshIcon />}
                      variant="outlined"
                      size="small"
                      onClick={() => fetchDocuments(selectedKnowledgeBase.id)}
                    >
                      Refresh
                    </Button>
                  </Box>
                  
                  {documentsLoading ? (
                    <CircularProgress sx={{ display: 'block', mx: 'auto', my: 4 }} />
                  ) : documents.length === 0 ? (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      No documents have been added to this knowledge base yet. Click the upload button to add your first document.
                    </Alert>
                  ) : (
                    <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                      <List>
                        {documents.map((doc) => (
                          <ListItem
                            key={doc.id}
                            secondaryAction={
                              <IconButton
                                edge="end"
                                color="error"
                                onClick={() => handleDeleteDocument(doc.id)}
                              >
                                <DeleteIcon />
                              </IconButton>
                            }
                            sx={{ borderBottom: '1px solid', borderColor: 'divider' }}
                          >
                            <ListItemIcon>
                              <FileIcon />
                            </ListItemIcon>
                            <ListItemText
                              primary={doc.filename}
                              secondary={
                                <React.Fragment>
                                  <Typography variant="body2" component="span">
                                    {formatFileSize(doc.size)} • {new Date(doc.created_at).toLocaleString()}
                                  </Typography>
                                  <Box sx={{ mt: 0.5 }}>
                                    {getStatusChip(doc.status)}
                                    {doc.chunk_count > 0 && (
                                      <Chip
                                        label={`${doc.chunk_count} chunks`}
                                        size="small"
                                        variant="outlined"
                                        sx={{ ml: 1 }}
                                      />
                                    )}
                                  </Box>
                                </React.Fragment>
                              }
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                </>
              )}
              
              {/* Statistics Tab */}
              {tabValue === 1 && (
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                          Total Documents
                        </Typography>
                        <Typography variant="h5" component="div">
                          {documents.length}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                          Total Chunks
                        </Typography>
                        <Typography variant="h5" component="div">
                          {documents.reduce((acc, doc) => acc + (doc.chunk_count || 0), 0)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  <Grid item xs={12} sm={6} md={4}>
                    <Card>
                      <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                          Last Updated
                        </Typography>
                        <Typography variant="h5" component="div">
                          {documents.length > 0
                            ? new Date(Math.max(...documents.map(d => new Date(d.created_at).getTime()))).toLocaleDateString()
                            : 'N/A'
                          }
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  <Grid item xs={12}>
                    <Card>
                      <CardContent>
                        <Typography color="textSecondary" gutterBottom>
                          Document Processing Status
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'space-around', mt: 2 }}>
                          <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h6">
                              {documents.filter(d => d.status === 'processed').length}
                            </Typography>
                            <Typography variant="body2">Processed</Typography>
                          </Box>
                          <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h6">
                              {documents.filter(d => d.status === 'processing').length}
                            </Typography>
                            <Typography variant="body2">Processing</Typography>
                          </Box>
                          <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h6">
                              {documents.filter(d => d.status === 'failed').length}
                            </Typography>
                            <Typography variant="body2">Failed</Typography>
                          </Box>
                          <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="h6">
                              {documents.filter(d => d.status === 'pending').length}
                            </Typography>
                            <Typography variant="body2">Pending</Typography>
                          </Box>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              )}
            </Paper>
          ) : (
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <Typography variant="h6" gutterBottom>
                No Knowledge Base Selected
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Select a knowledge base from the sidebar or create a new one to get started.
              </Typography>
              <Button
                variant="contained"
                color="primary"
                startIcon={<AddIcon />}
                onClick={() => setCreateDialogOpen(true)}
              >
                Create New Knowledge Base
              </Button>
            </Paper>
          )}
        </Grid>
      </Grid>
      
      {/* Create Knowledge Base Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Knowledge Base</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            variant="outlined"
            value={newKbName}
            onChange={(e) => setNewKbName(e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Description (Optional)"
            fullWidth
            variant="outlined"
            value={newKbDescription}
            onChange={(e) => setNewKbDescription(e.target.value)}
            multiline
            rows={2}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth>
            <InputLabel>Type</InputLabel>
            <Select
              value={newKbType}
              label="Type"
              onChange={(e) => setNewKbType(e.target.value)}
            >
              <MenuItem value="document">Document</MenuItem>
              <MenuItem value="graph">Graph</MenuItem>
              <MenuItem value="structured">Structured Data</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreateKnowledgeBase}
            color="primary"
            variant="contained"
            disabled={!newKbName.trim()}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Edit Knowledge Base Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Knowledge Base</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            variant="outlined"
            value={editKbName}
            onChange={(e) => setEditKbName(e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Description (Optional)"
            fullWidth
            variant="outlined"
            value={editKbDescription}
            onChange={(e) => setEditKbDescription(e.target.value)}
            multiline
            rows={2}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleUpdateKnowledgeBase}
            color="primary"
            variant="contained"
            disabled={!editKbName.trim()}
          >
            Update
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Delete Knowledge Base Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Knowledge Base</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the knowledge base "{selectedKnowledgeBase?.name}"?
            This action cannot be undone and all associated documents will be permanently deleted.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteKnowledgeBase} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Upload Document Dialog */}
      <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Upload Document</DialogTitle>
        <DialogContent>
          <Typography variant="body2" gutterBottom>
            Upload a document to add to the knowledge base "{selectedKnowledgeBase?.name}".
            Supported formats: PDF, TXT, DOCX, CSV, MD.
          </Typography>
          
          <Box sx={{ mt: 2 }}>
            <input
              type="file"
              accept=".pdf,.txt,.docx,.csv,.md"
              onChange={handleFileChange}
              style={{ display: 'none' }}
              id="document-upload"
            />
            <label htmlFor="document-upload">
              <Button
                variant="outlined"
                component="span"
                fullWidth
                startIcon={<UploadIcon />}
                sx={{ mb: 2 }}
              >
                Select File
              </Button>
            </label>
            
            {uploadFile && (
              <Box sx={{ mt: 1, mb: 2 }}>
                <Typography variant="subtitle2">Selected File:</Typography>
                <Typography variant="body2">
                  {uploadFile.name} ({formatFileSize(uploadFile.size)})
                </Typography>
              </Box>
            )}
            
            {uploading && (
              <Box sx={{ mt: 2, mb: 1 }}>
                <Typography variant="body2" gutterBottom>
                  Uploading: {uploadProgress}%
                </Typography>
                <LinearProgress variant="determinate" value={uploadProgress} />
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleUploadDocument}
            color="primary"
            variant="contained"
            disabled={!uploadFile || uploading}
            startIcon={uploading ? <CircularProgress size={20} /> : null}
          >
            Upload
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Search Dialog */}
      <Dialog open={searchDialogOpen} onClose={() => setSearchDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Search Knowledge Base</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', mb: 2 }}>
            <TextField
              autoFocus
              fullWidth
              label="Search Query"
              variant="outlined"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <Button
              variant="contained"
              color="primary"
              onClick={handleSearch}
              disabled={searching || !searchQuery.trim()}
              sx={{ ml: 1, px: 3 }}
              startIcon={searching ? <CircularProgress size={20} /> : <SearchIcon />}
            >
              Search
            </Button>
          </Box>
          
          {searchResults.length > 0 ? (
            <List sx={{ maxHeight: 400, overflow: 'auto' }}>
              {searchResults.map((result, index) => (
                <ListItem key={index} sx={{ flexDirection: 'column', alignItems: 'flex-start', py: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    {result.document_name || 'Document'}
                    {result.score && (
                      <Chip 
                        label={`Score: ${result.score.toFixed(2)}`}
                        size="small"
                        color="primary"
                        sx={{ ml: 1 }}
                      />
                    )}
                  </Typography>
                  <Paper variant="outlined" sx={{ p: 2, width: '100%', bgcolor: 'background.paper' }}>
                    <Typography variant="body2">{result.content}</Typography>
                  </Paper>
                </ListItem>
              ))}
            </List>
          ) : searching ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : (
            <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
              {searchQuery.trim() ? 'No results found' : 'Enter a search query'}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSearchDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
      
      {/* Configuration Dialog */}
      <Dialog open={configDialogOpen} onClose={() => setConfigDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Knowledge Base Configuration</DialogTitle>
        <DialogContent>
          <FormControl fullWidth margin="normal">
            <InputLabel>Embedding Model</InputLabel>
            <Select
              value={embeddingModel}
              label="Embedding Model"
              onChange={(e) => setEmbeddingModel(e.target.value)}
            >
              <MenuItem value="text-embedding-ada-002">OpenAI Ada 002</MenuItem>
              <MenuItem value="text-embedding-3-small">OpenAI Embedding 3 Small</MenuItem>
              <MenuItem value="text-embedding-3-large">OpenAI Embedding 3 Large</MenuItem>
              <MenuItem value="all-mpnet-base-v2">MPNET Base v2</MenuItem>
              <MenuItem value="instructor-xl">Instructor XL</MenuItem>
            </Select>
          </FormControl>
          
          <TextField
            margin="normal"
            label="Chunk Size"
            type="number"
            fullWidth
            variant="outlined"
            value={chunkSize}
            onChange={(e) => setChunkSize(parseInt(e.target.value) || 1000)}
            InputProps={{ inputProps: { min: 100, max: 8000 } }}
          />
          
          <TextField
            margin="normal"
            label="Chunk Overlap"
            type="number"
            fullWidth
            variant="outlined"
            value={chunkOverlap}
            onChange={(e) => setChunkOverlap(parseInt(e.target.value) || 200)}
            InputProps={{ inputProps: { min: 0, max: 1000 } }}
          />
          
          <TextField
            margin="normal"
            label="Similarity Threshold"
            type="number"
            fullWidth
            variant="outlined"
            value={similarityThreshold}
            onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value) || 0.75)}
            InputProps={{ inputProps: { min: 0, max: 1, step: 0.01 } }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfigDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleUpdateConfig}
            color="primary"
            variant="contained"
          >
            Update Configuration
          </Button>
        </DialogActions>
      </Dialog>
    </PageLayout>
  );
};

export default KnowledgeBasePage;