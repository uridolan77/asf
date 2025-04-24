import React from 'react';
import { 
  Dialog, DialogTitle, DialogContent, DialogActions, 
  Button, TextField, FormControl, InputLabel, Select, 
  MenuItem, DialogContentText, Grid, CircularProgress
} from '@mui/material';

/**
 * Dialog for creating a new knowledge base
 */
const CreateKnowledgeBaseDialog = ({ 
  open, 
  onClose, 
  formData, 
  onChange, 
  onSubmit, 
  actionInProgress 
}) => {
  const handleChange = (e) => {
    const { name, value } = e.target;
    onChange({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit();
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      fullWidth
      maxWidth="sm"
    >
      <DialogTitle>Create New Knowledge Base</DialogTitle>
      <form onSubmit={handleSubmit}>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Create a new knowledge base to automatically collect and organize research on a specific topic.
          </DialogContentText>
          
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                autoFocus
                margin="dense"
                id="name"
                name="name"
                label="Knowledge Base Name"
                type="text"
                fullWidth
                variant="outlined"
                value={formData.name}
                onChange={handleChange}
                placeholder="e.g., Community Acquired Pneumonia Research"
                required
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                margin="dense"
                id="query"
                name="query"
                label="Search Query"
                type="text"
                fullWidth
                variant="outlined"
                value={formData.query}
                onChange={handleChange}
                placeholder="e.g., community acquired pneumonia treatment outcomes"
                required
                multiline
                rows={4}
                helperText="Enter keywords or phrases to search for relevant medical literature"
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormControl fullWidth margin="dense">
                <InputLabel id="update-schedule-label">Update Schedule</InputLabel>
                <Select
                  labelId="update-schedule-label"
                  id="updateSchedule"
                  name="updateSchedule"
                  value={formData.updateSchedule}
                  label="Update Schedule"
                  onChange={handleChange}
                >
                  <MenuItem value="daily">Daily</MenuItem>
                  <MenuItem value="weekly">Weekly</MenuItem>
                  <MenuItem value="monthly">Monthly</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={onClose} disabled={actionInProgress}>
            Cancel
          </Button>
          <Button 
            type="submit" 
            variant="contained" 
            color="primary"
            disabled={actionInProgress}
            startIcon={actionInProgress ? <CircularProgress size={20} /> : null}
          >
            {actionInProgress ? 'Creating...' : 'Create Knowledge Base'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default CreateKnowledgeBaseDialog;
