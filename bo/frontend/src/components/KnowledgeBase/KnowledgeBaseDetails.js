import React from 'react';
import {
  Box, Typography, Button, Paper, Chip, Divider,
  Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, CircularProgress, Grid
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Add as AddIcon
} from '@mui/icons-material';

/**
 * Component to display knowledge base details
 */
const KnowledgeBaseDetails = ({
  selectedKB,
  loading,
  actionInProgress,
  onUpdate,
  onDelete,
  onCreateNew
}) => {
  // Helper function to get color based on relevance score
  const getRelevanceColor = (score) => {
    if (score >= 0.9) return 'success'; // High relevance - green
    if (score >= 0.7) return 'primary'; // Medium relevance - blue
    return 'warning'; // Low relevance - orange
  };

  if (loading) {
    return (
      <Box sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100%'
      }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!selectedKB) {
    return (
      <Box sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100%',
        p: 3
      }}>
        <Typography variant="h6" color="text.secondary" gutterBottom>
          Select a knowledge base from the list to view details
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          - or -
        </Typography>
        <Button
          variant="contained"
          color="secondary"
          startIcon={<AddIcon />}
          onClick={onCreateNew}
          sx={{ mt: 2 }}
        >
          Create New Knowledge Base
        </Button>
      </Box>
    );
  }

  // Safely access nested properties
  const safelyGetData = (obj, path, defaultValue = '') => {
    try {
      return path.split('.').reduce((o, key) => (o || {})[key], obj) || defaultValue;
    } catch (e) {
      return defaultValue;
    }
  };

  // Safely format date
  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch (e) {
      return 'N/A';
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mb: 2
      }}>
        <Typography variant="h5">{safelyGetData(selectedKB, 'name', 'Knowledge Base')}</Typography>
        <Box>
          <Button
            variant="outlined"
            color="primary"
            startIcon={<RefreshIcon />}
            onClick={() => onUpdate(selectedKB.id)}
            disabled={actionInProgress}
            sx={{ mr: 1 }}
          >
            {actionInProgress ? 'Updating...' : 'Update Now'}
          </Button>
          <Button
            variant="outlined"
            color="error"
            startIcon={<DeleteIcon />}
            onClick={() => onDelete(selectedKB.id)}
            disabled={actionInProgress}
          >
            Delete
          </Button>
        </Box>
      </Box>

      <Paper elevation={0} sx={{ p: 2, mb: 3, bgcolor: 'background.paper', borderRadius: 2 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography variant="body2">
              <strong>Query:</strong> {safelyGetData(selectedKB, 'query')}
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2">
              <strong>Update Schedule:</strong> {safelyGetData(selectedKB, 'update_schedule')}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">
              <strong>Created:</strong> {formatDate(safelyGetData(selectedKB, 'created_at'))}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">
              <strong>Last Updated:</strong> {formatDate(safelyGetData(selectedKB, 'last_updated'))}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">
              <strong>Articles:</strong> {safelyGetData(selectedKB, 'article_count', 0)}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Articles Section */}
      <Typography variant="h6" gutterBottom>
        Articles ({safelyGetData(selectedKB, 'articles.length', 0)})
      </Typography>
      <TableContainer
        component={Paper}
        elevation={0}
        sx={{
          mb: 3,
          maxHeight: 300,
          overflow: 'auto',
          flex: '0 0 auto'
        }}
      >
        {selectedKB?.articles?.length > 0 ? (
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Title</TableCell>
                <TableCell>Journal</TableCell>
                <TableCell>Year</TableCell>
                <TableCell>Relevance</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {selectedKB.articles.slice(0, 50).map((article, index) => (
                <TableRow key={article.id || index} hover>
                  <TableCell>{safelyGetData(article, 'title')}</TableCell>
                  <TableCell>{safelyGetData(article, 'journal')}</TableCell>
                  <TableCell>{safelyGetData(article, 'year')}</TableCell>
                  <TableCell>
                    <Chip
                      label={`${(safelyGetData(article, 'relevance_score', 0) * 100).toFixed(0)}%`}
                      color={getRelevanceColor(safelyGetData(article, 'relevance_score', 0))}
                      size="small"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Box sx={{ p: 2 }}>
            <Typography variant="body2" color="text.secondary">
              No articles found in this knowledge base.
            </Typography>
          </Box>
        )}
      </TableContainer>

      {/* Concepts Section */}
      <Typography variant="h6" gutterBottom>
        Key Concepts ({safelyGetData(selectedKB, 'concepts.length', 0)})
      </Typography>
      <Box sx={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 1,
        mb: 2,
        flex: '1 1 auto',
        overflow: 'auto'
      }}>
        {selectedKB?.concepts?.length > 0 ? (
          selectedKB.concepts.slice(0, 50).map((concept, index) => (
            <Chip
              key={concept.id || index}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <span>{safelyGetData(concept, 'name')}</span>
                  <Chip
                    label={safelyGetData(concept, 'related_articles', 0)}
                    size="small"
                    color="primary"
                    sx={{ ml: 0.5, height: 20, '& .MuiChip-label': { px: 1, py: 0 } }}
                  />
                </Box>
              }
              variant="outlined"
              sx={{ px: 1 }}
            />
          ))
        ) : (
          <Typography variant="body2" color="text.secondary">
            No concepts extracted yet.
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default KnowledgeBaseDetails;
