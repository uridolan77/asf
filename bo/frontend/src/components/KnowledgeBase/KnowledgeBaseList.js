import React from 'react';
import { 
  Box, Typography, List, ListItem, ListItemText, 
  ListItemButton, Divider, Button, Chip
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';

/**
 * Component to display a list of knowledge bases
 */
const KnowledgeBaseList = ({ 
  knowledgeBases, 
  selectedKB, 
  onSelectKB, 
  onAddNew 
}) => {
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        mb: 2,
        px: 1
      }}>
        <Typography variant="h6">Knowledge Bases</Typography>
        <Button
          variant="contained"
          color="secondary"
          size="small"
          startIcon={<AddIcon />}
          onClick={onAddNew}
        >
          New
        </Button>
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {knowledgeBases.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
          No knowledge bases found. Create one to get started.
        </Typography>
      ) : (
        <List sx={{ 
          overflow: 'auto', 
          flex: 1,
          '& .MuiListItemButton-root.Mui-selected': {
            bgcolor: 'primary.light',
            borderLeft: '4px solid',
            borderColor: 'primary.main',
            '&:hover': {
              bgcolor: 'primary.light',
            }
          }
        }}>
          {knowledgeBases.map(kb => (
            <ListItemButton
              key={kb.id}
              selected={selectedKB && selectedKB.id === kb.id}
              onClick={() => onSelectKB(kb.id)}
              sx={{ 
                borderBottom: '1px solid',
                borderColor: 'divider',
                pl: selectedKB && selectedKB.id === kb.id ? 1 : 2
              }}
            >
              <ListItemText
                primary={kb.name}
                secondary={
                  <Box>
                    <Typography variant="body2" component="span">
                      {kb.article_count} articles
                    </Typography>
                    <Typography variant="caption" display="block" color="text.secondary">
                      Last updated: {new Date(kb.last_updated).toLocaleDateString()}
                    </Typography>
                  </Box>
                }
              />
            </ListItemButton>
          ))}
        </List>
      )}
    </Box>
  );
};

export default KnowledgeBaseList;
