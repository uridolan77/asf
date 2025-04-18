import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Chip,
  FormControlLabel,
  Switch,
  Tooltip
} from '@mui/material';
import {
  Upload as UploadIcon,
  Settings as SettingsIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

interface DocumentUploaderProps {
  file: File | null;
  setFile: (file: File | null) => void;
  useEnhanced: boolean;
  setUseEnhanced: (value: boolean) => void;
  useStreaming: boolean;
  setUseStreaming: (value: boolean) => void;
  useParallel: boolean;
  setUseParallel: (value: boolean) => void;
}

/**
 * Component for uploading documents with processing options
 */
const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  file,
  setFile,
  useEnhanced,
  setUseEnhanced,
  useStreaming,
  setUseStreaming,
  useParallel,
  setUseParallel
}) => {
  // Dropzone configuration
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/json': ['.json']
    },
    maxFiles: 1,
    onDrop: acceptedFiles => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
      }
    }
  });

  return (
    <>
      <Paper
        {...getRootProps()}
        sx={{
          p: 3,
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          backgroundColor: isDragActive ? 'rgba(0, 0, 0, 0.05)' : 'background.paper',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            borderColor: 'primary.main',
            backgroundColor: 'rgba(0, 0, 0, 0.05)'
          }
        }}
      >
        <input {...getInputProps()} />
        <UploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive ? 'Drop the file here' : 'Drag & drop a file here, or click to select'}
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Supported formats: PDF, TXT, MD, JSON
        </Typography>
        {file && (
          <Box sx={{ mt: 2 }}>
            <Chip
              label={file.name}
              onDelete={() => setFile(null)}
              color="primary"
            />
          </Box>
        )}
      </Paper>

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={useEnhanced}
              onChange={(e) => setUseEnhanced(e.target.checked)}
              color="primary"
            />
          }
          label={
            <Tooltip title="Use the enhanced document processor with improved performance and features">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <SettingsIcon fontSize="small" sx={{ mr: 0.5 }} />
                <span>Use enhanced processor</span>
              </Box>
            </Tooltip>
          }
        />

        <FormControlLabel
          control={
            <Switch
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
              disabled={!useEnhanced}
              color="primary"
            />
          }
          label={
            <Tooltip title="Stream results as they become available">
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <SpeedIcon fontSize="small" sx={{ mr: 0.5 }} />
                <span>Use streaming</span>
              </Box>
            </Tooltip>
          }
        />

        <FormControlLabel
          control={
            <Switch
              checked={useParallel}
              onChange={(e) => setUseParallel(e.target.checked)}
              color="primary"
            />
          }
          label="Use parallel processing"
        />
      </Box>
    </>
  );
};

export default DocumentUploader;
