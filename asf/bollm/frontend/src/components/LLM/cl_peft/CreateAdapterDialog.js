import React, { useState } from 'react';
import { 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions, 
  Button, 
  TextField, 
  Grid, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  FormHelperText,
  Typography,
  Divider,
  Box,
  Chip,
  Autocomplete,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  useTheme
} from '@mui/material';
import { useSnackbar } from 'notistack';

import { createAdapter } from '../../../services/cl_peft_service';

const CL_STRATEGIES = [
  { value: 'naive', label: 'Naive (Sequential Fine-Tuning)', description: 'Simple sequential fine-tuning without any CL mechanisms.' },
  { value: 'ewc', label: 'EWC (Elastic Weight Consolidation)', description: 'Adds a regularization term to the loss to prevent forgetting.' },
  { value: 'replay', label: 'Experience Replay', description: 'Stores and replays examples from previous tasks.' },
  { value: 'generative_replay', label: 'Generative Replay', description: 'Uses the model to generate synthetic examples from previous tasks.' },
  { value: 'orthogonal_lora', label: 'Orthogonal LoRA (O-LoRA)', description: 'Enforces orthogonality between LoRA updates for different tasks.' },
  { value: 'adaptive_svd', label: 'Adaptive SVD', description: 'Projects gradient updates onto orthogonal subspaces.' },
  { value: 'mask_based', label: 'Mask-Based', description: 'Uses binary masks to activate specific parameters for each task.' }
];

const PEFT_METHODS = [
  { value: 'lora', label: 'LoRA', description: 'Low-Rank Adaptation for efficient fine-tuning.' },
  { value: 'qlora', label: 'QLoRA', description: 'Quantized Low-Rank Adaptation for efficient fine-tuning.' }
];

const QUANTIZATION_MODES = [
  { value: 'none', label: 'None', description: 'No quantization.' },
  { value: 'int8', label: 'INT8', description: '8-bit quantization.' },
  { value: 'int4', label: 'INT4', description: '4-bit quantization (QLoRA).' }
];

const TASK_TYPES = [
  { value: 'causal_lm', label: 'Causal Language Modeling', description: 'For generative language models like GPT.' },
  { value: 'seq_cls', label: 'Sequence Classification', description: 'For classification tasks.' },
  { value: 'seq2seq', label: 'Sequence-to-Sequence', description: 'For encoder-decoder models like T5.' }
];

const POPULAR_BASE_MODELS = [
  'meta-llama/Llama-2-7b-hf',
  'meta-llama/Llama-2-13b-hf',
  'meta-llama/Llama-2-70b-hf',
  'mistralai/Mistral-7B-v0.1',
  'mistralai/Mixtral-8x7B-v0.1',
  'google/gemma-2b',
  'google/gemma-7b',
  'tiiuae/falcon-7b',
  'tiiuae/falcon-40b',
  'EleutherAI/gpt-neox-20b',
  'bigscience/bloom',
  'mosaicml/mpt-7b',
  'facebook/opt-6.7b',
  'facebook/opt-13b',
  'facebook/opt-30b',
  'google/flan-t5-base',
  'google/flan-t5-large',
  'google/flan-t5-xl',
  'google/flan-t5-xxl'
];

const CreateAdapterDialog = ({ open, onClose, onAdapterCreated }) => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  
  const [formData, setFormData] = useState({
    adapter_name: '',
    base_model_name: '',
    description: '',
    cl_strategy: 'naive',
    peft_method: 'lora',
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    target_modules: null,
    quantization_mode: 'none',
    task_type: 'causal_lm',
    tags: []
  });
  
  const [errors, setErrors] = useState({});
  
  const handleChange = (e) => {
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
  
  const handleTagsChange = (event, newValue) => {
    setFormData({
      ...formData,
      tags: newValue
    });
  };
  
  const validateStep = (step) => {
    const newErrors = {};
    
    if (step === 0) {
      if (!formData.adapter_name.trim()) {
        newErrors.adapter_name = 'Adapter name is required';
      }
      
      if (!formData.base_model_name.trim()) {
        newErrors.base_model_name = 'Base model name is required';
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleNext = () => {
    if (validateStep(activeStep)) {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };
  
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };
  
  const handleSubmit = async () => {
    if (!validateStep(activeStep)) {
      return;
    }
    
    setLoading(true);
    try {
      await createAdapter(formData);
      onAdapterCreated();
      resetForm();
    } catch (error) {
      console.error('Error creating adapter:', error);
      enqueueSnackbar('Failed to create adapter', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };
  
  const resetForm = () => {
    setFormData({
      adapter_name: '',
      base_model_name: '',
      description: '',
      cl_strategy: 'naive',
      peft_method: 'lora',
      lora_r: 16,
      lora_alpha: 32,
      lora_dropout: 0.05,
      target_modules: null,
      quantization_mode: 'none',
      task_type: 'causal_lm',
      tags: []
    });
    setErrors({});
    setActiveStep(0);
  };
  
  const handleClose = () => {
    resetForm();
    onClose();
  };
  
  const steps = ['Basic Information', 'CL & PEFT Configuration', 'Advanced Settings'];
  
  return (
    <Dialog 
      open={open} 
      onClose={handleClose}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle>Create New CL-PEFT Adapter</DialogTitle>
      
      <DialogContent>
        <Stepper activeStep={activeStep} sx={{ mb: 4, mt: 2 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        {activeStep === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                name="adapter_name"
                label="Adapter Name"
                fullWidth
                required
                value={formData.adapter_name}
                onChange={handleChange}
                error={!!errors.adapter_name}
                helperText={errors.adapter_name}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Autocomplete
                freeSolo
                options={POPULAR_BASE_MODELS}
                value={formData.base_model_name}
                onChange={(event, newValue) => {
                  setFormData({
                    ...formData,
                    base_model_name: newValue || ''
                  });
                  if (errors.base_model_name) {
                    setErrors({
                      ...errors,
                      base_model_name: null
                    });
                  }
                }}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    name="base_model_name"
                    label="Base Model Name"
                    required
                    error={!!errors.base_model_name}
                    helperText={errors.base_model_name || 'Enter the Hugging Face model ID'}
                  />
                )}
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                name="description"
                label="Description"
                fullWidth
                multiline
                rows={3}
                value={formData.description}
                onChange={handleChange}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Autocomplete
                multiple
                freeSolo
                options={[]}
                value={formData.tags}
                onChange={handleTagsChange}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip
                      label={option}
                      {...getTagProps({ index })}
                      size="small"
                    />
                  ))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Tags"
                    placeholder="Add tags..."
                    helperText="Press Enter to add a tag"
                  />
                )}
              />
            </Grid>
          </Grid>
        )}
        
        {activeStep === 1 && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>CL Strategy</InputLabel>
                <Select
                  name="cl_strategy"
                  value={formData.cl_strategy}
                  onChange={handleChange}
                  label="CL Strategy"
                >
                  {CL_STRATEGIES.map((strategy) => (
                    <MenuItem key={strategy.value} value={strategy.value}>
                      {strategy.label}
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText>
                  {CL_STRATEGIES.find(s => s.value === formData.cl_strategy)?.description}
                </FormHelperText>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>PEFT Method</InputLabel>
                <Select
                  name="peft_method"
                  value={formData.peft_method}
                  onChange={handleChange}
                  label="PEFT Method"
                >
                  {PEFT_METHODS.map((method) => (
                    <MenuItem key={method.value} value={method.value}>
                      {method.label}
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText>
                  {PEFT_METHODS.find(m => m.value === formData.peft_method)?.description}
                </FormHelperText>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Divider sx={{ my: 1 }} />
              <Typography variant="subtitle2" gutterBottom>
                LoRA Configuration
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <TextField
                name="lora_r"
                label="LoRA Rank (r)"
                type="number"
                fullWidth
                value={formData.lora_r}
                onChange={handleChange}
                InputProps={{ inputProps: { min: 1, max: 128 } }}
                helperText="Rank of the low-rank matrices (8-64 typical)"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <TextField
                name="lora_alpha"
                label="LoRA Alpha"
                type="number"
                fullWidth
                value={formData.lora_alpha}
                onChange={handleChange}
                InputProps={{ inputProps: { min: 1, max: 128 } }}
                helperText="Scaling factor (typically 2x rank)"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <TextField
                name="lora_dropout"
                label="LoRA Dropout"
                type="number"
                fullWidth
                value={formData.lora_dropout}
                onChange={handleChange}
                InputProps={{ inputProps: { min: 0, max: 0.5, step: 0.01 } }}
                helperText="Dropout probability (0.0-0.1 typical)"
              />
            </Grid>
          </Grid>
        )}
        
        {activeStep === 2 && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Quantization Mode</InputLabel>
                <Select
                  name="quantization_mode"
                  value={formData.quantization_mode}
                  onChange={handleChange}
                  label="Quantization Mode"
                >
                  {QUANTIZATION_MODES.map((mode) => (
                    <MenuItem key={mode.value} value={mode.value}>
                      {mode.label}
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText>
                  {QUANTIZATION_MODES.find(m => m.value === formData.quantization_mode)?.description}
                </FormHelperText>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Task Type</InputLabel>
                <Select
                  name="task_type"
                  value={formData.task_type}
                  onChange={handleChange}
                  label="Task Type"
                >
                  {TASK_TYPES.map((type) => (
                    <MenuItem key={type.value} value={type.value}>
                      {type.label}
                    </MenuItem>
                  ))}
                </Select>
                <FormHelperText>
                  {TASK_TYPES.find(t => t.value === formData.task_type)?.description}
                </FormHelperText>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="textSecondary">
                  Target modules will be automatically determined based on the base model.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        {activeStep > 0 && (
          <Button onClick={handleBack}>
            Back
          </Button>
        )}
        {activeStep < steps.length - 1 ? (
          <Button onClick={handleNext} variant="contained">
            Next
          </Button>
        ) : (
          <Button 
            onClick={handleSubmit} 
            variant="contained" 
            disabled={loading}
            startIcon={loading && <CircularProgress size={20} />}
          >
            {loading ? 'Creating...' : 'Create Adapter'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default CreateAdapterDialog;
