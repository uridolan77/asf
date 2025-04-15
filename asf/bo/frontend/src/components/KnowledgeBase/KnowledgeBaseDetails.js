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

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mb: 2 
      }}>
        <Typography variant="h5">{selectedKB.name}</Typography>
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
              <strong>Query:</strong> {selectedKB.query}
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="body2">
              <strong>Update Schedule:</strong> {selectedKB.update_schedule}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">
              <strong>Created:</strong> {new Date(selectedKB.created_at).toLocaleString()}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">
              <strong>Last Updated:</strong> {new Date(selectedKB.last_updated).toLocaleString()}
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="body2">
              <strong>Articles:</strong> {selectedKB.article_count}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Articles Section */}
      <Typography variant="h6" gutterBottom>
        Articles ({selectedKB.articles?.length || 0})
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
        {selectedKB.articles?.length > 0 ? (
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
              {selectedKB.articles.map(article => (
                <TableRow key={article.id} hover>
                  <TableCell>{article.title}</TableCell>
                  <TableCell>{article.journal}</TableCell>
                  <TableCell>{article.year}</TableCell>
                  <TableCell>
                    <Chip
                      label={`${(article.relevance_score * 100).toFixed(0)}%`}
                      color={getRelevanceColor(article.relevance_score)}
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
        Key Concepts ({selectedKB.concepts?.length || 0})
      </Typography>
      <Box sx={{ 
        display: 'flex', 
        flexWrap: 'wrap', 
        gap: 1,
        mb: 2,
        flex: '1 1 auto',
        overflow: 'auto'
      }}>
        {selectedKB.concepts?.length > 0 ? (
          selectedKB.concepts.map(concept => (
            <Chip
              key={concept.id}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <span>{concept.name}</span>
                  <Chip
                    label={concept.related_articles}
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
