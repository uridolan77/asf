import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  Grid,
  Typography,
  IconButton,
  Chip,
  Box,
  CircularProgress,
  useTheme
} from '@mui/material';
import {
  Close as CloseIcon,
  Add as AddIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';

/**
 * CreateAdapterDialog component for creating new CL-PEFT adapters
 */
const CreateAdapterDialog = ({ 
  open, 
  onClose, 
  onSubmit, 
  clStrategies = [],
  peftMethods = [],
  baseModels = []
}) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  
  const [formData, setFormData] = useState({
    adapter_name: '',
    base_model_name: '',
    cl_strategy: '',
    peft_method: '',
    description: '',
    tags: []
  });
  
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [newTag, setNewTag] = useState('');
  
  // Handle input change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    
    // Clear error for this field
    if (errors[name]) {
      setErrors({
        ...errors,
        [name]: null
      });
    }
  };
  
  // Handle tag input change
  const handleTagInputChange = (e) => {
    setNewTag(e.target.value);
  };
  
  // Handle add tag
  const handleAddTag = () => {
    if (newTag.trim() && !formData.tags.includes(newTag.trim())) {
      setFormData({
        ...formData,
        tags: [...formData.tags, newTag.trim()]
      });
      setNewTag('');
    }
  };
  
  // Handle remove tag
  const handleRemoveTag = (tagToRemove) => {
    setFormData({
      ...formData,
      tags: formData.tags.filter(tag => tag !== tagToRemove)
    });
  };
  
  // Handle key press for tag input
  const handleTagKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddTag();
    }
  };
  
  // Validate form
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.adapter_name.trim()) {
      newErrors.adapter_name = 'Adapter name is required';
    }
    
    if (!formData.base_model_name) {
      newErrors.base_model_name = 'Base model is required';
    }
    
    if (!formData.cl_strategy) {
      newErrors.cl_strategy = 'CL strategy is required';
    }
    
    if (!formData.peft_method) {
      newErrors.peft_method = 'PEFT method is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  // Handle submit
  const handleSubmit = async () => {
    if (!validateForm()) {
      enqueueSnackbar('Please fill all required fields', { variant: 'error' });
      return;
    }
    
    setLoading(true);
    
    try {
      await onSubmit(formData);
      handleReset();
    } catch (error) {
      console.error('Error submitting form:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle reset
  const handleReset = () => {
    setFormData({
      adapter_name: '',
      base_model_name: '',
      cl_strategy: '',
      peft_method: '',
      description: '',
      tags: []
    });
    setErrors({});
    setNewTag('');
  };
  
  // Handle close
  const handleClose = () => {
    handleReset();
    onClose();
  };
  
  return (
    <Dialog 
      open={open} 
      onClose={handleClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>
        Create New Adapter
        <IconButton
          aria-label="close"
          onClick={handleClose}
          sx={{ position: 'absolute', right: 8, top: 8 }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      
      <DialogContent dividers>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Adapter Name"
              name="adapter_name"
              value={formData.adapter_name}
              onChange={handleInputChange}
              error={!!errors.adapter_name}
              helperText={errors.adapter_name}
              required
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControl fullWidth error={!!errors.base_model_name} required>
              <InputLabel id="base-model-label">Base Model</InputLabel>
              <Select
                labelId="base-model-label"
                id="base-model"
                name="base_model_name"
                value={formData.base_model_name}
                onChange={handleInputChange}
                label="Base Model"
              >
                {baseModels.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    {model.name}
                  </MenuItem>
                ))}
              </Select>
              {errors.base_model_name && (
                <FormHelperText>{errors.base_model_name}</FormHelperText>
              )}
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControl fullWidth error={!!errors.cl_strategy} required>
              <InputLabel id="cl-strategy-label">CL Strategy</InputLabel>
              <Select
                labelId="cl-strategy-label"
                id="cl-strategy"
                name="cl_strategy"
                value={formData.cl_strategy}
                onChange={handleInputChange}
                label="CL Strategy"
              >
                {clStrategies.map((strategy) => (
                  <MenuItem key={strategy.id} value={strategy.id}>
                    {strategy.name}
                  </MenuItem>
                ))}
              </Select>
              {errors.cl_strategy && (
                <FormHelperText>{errors.cl_strategy}</FormHelperText>
              )}
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControl fullWidth error={!!errors.peft_method} required>
              <InputLabel id="peft-method-label">PEFT Method</InputLabel>
              <Select
                labelId="peft-method-label"
                id="peft-method"
                name="peft_method"
                value={formData.peft_method}
                onChange={handleInputChange}
                label="PEFT Method"
              >
                {peftMethods.map((method) => (
                  <MenuItem key={method.id} value={method.id}>
                    {method.name}
                  </MenuItem>
                ))}
              </Select>
              {errors.peft_method && (
                <FormHelperText>{errors.peft_method}</FormHelperText>
              )}
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
              <TextField
                fullWidth
                label="Add Tags"
                value={newTag}
                onChange={handleTagInputChange}
                onKeyPress={handleTagKeyPress}
                placeholder="Enter tag and press Enter"
                size="medium"
              />
              <Button
                variant="outlined"
                onClick={handleAddTag}
                sx={{ mt: 1 }}
              >
                <AddIcon />
              </Button>
            </Box>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
              {formData.tags.map((tag, index) => (
                <Chip
                  key={index}
                  label={tag}
                  onDelete={() => handleRemoveTag(tag)}
                  size="small"
                />
              ))}
            </Box>
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Description"
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              multiline
              rows={3}
              placeholder="Enter a description for this adapter"
            />
          </Grid>
        </Grid>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          color="primary"
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : null}
        >
          {loading ? 'Creating...' : 'Create Adapter'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CreateAdapterDialog;
