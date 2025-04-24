// frontend/src/components/UI/AccessibleComponents.js
import React from 'react';
import { 
  Button, 
  TextField, 
  IconButton, 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions,
  Tooltip,
  Typography,
  Link,
  Box,
  Alert,
  FormControlLabel,
  Switch,
  Tabs,
  Tab,
  CircularProgress
} from '@mui/material';

/**
 * Accessible Button with improved ARIA support
 */
export const AccessibleButton = ({ 
  children, 
  loading = false, 
  disabled = false, 
  tooltip = '', 
  onClick, 
  startIcon = null,
  endIcon = null,
  ...props 
}) => {
  const buttonContent = loading ? (
    <CircularProgress size={24} color="inherit" />
  ) : children;

  const button = (
    <Button
      onClick={onClick}
      disabled={disabled || loading}
      startIcon={loading ? null : startIcon}
      endIcon={loading ? null : endIcon}
      aria-busy={loading}
      aria-disabled={disabled}
      {...props}
    >
      {buttonContent}
    </Button>
  );

  // Wrap with tooltip if provided
  if (tooltip) {
    return (
      <Tooltip title={tooltip}>
        <span>{button}</span>
      </Tooltip>
    );
  }

  return button;
};

/**
 * Accessible Form Field with label association and error messaging
 */
export const AccessibleTextField = ({ 
  id, 
  label, 
  error = false, 
  helperText = '', 
  required = false,
  type = 'text',
  ...props 
}) => {
  const fieldId = id || `field-${Math.random().toString(36).substring(2, 9)}`;
  const helperTextId = `${fieldId}-helper-text`;
  
  return (
    <TextField
      id={fieldId}
      label={label}
      error={error}
      helperText={helperText}
      required={required}
      type={type}
      aria-describedby={helperText ? helperTextId : undefined}
      aria-required={required}
      aria-invalid={error}
      InputLabelProps={{
        id: `${fieldId}-label`,
        htmlFor: fieldId,
      }}
      FormHelperTextProps={{
        id: helperTextId
      }}
      {...props}
    />
  );
};

/**
 * Accessible Modal Dialog
 */
export const AccessibleDialog = ({ 
  open, 
  onClose, 
  title, 
  description = '', 
  children, 
  actions = null,
  maxWidth = 'sm',
  fullWidth = true,
  ...props 
}) => {
  const titleId = `dialog-${Math.random().toString(36).substring(2, 9)}-title`;
  const descriptionId = `dialog-${Math.random().toString(36).substring(2, 9)}-description`;
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      aria-labelledby={titleId}
      aria-describedby={description ? descriptionId : undefined}
      maxWidth={maxWidth}
      fullWidth={fullWidth}
      {...props}
    >
      <DialogTitle id={titleId}>{title}</DialogTitle>
      <DialogContent>
        {description && (
          <Typography id={descriptionId} variant="body2" gutterBottom>
            {description}
          </Typography>
        )}
        {children}
      </DialogContent>
      {actions && <DialogActions>{actions}</DialogActions>}
    </Dialog>
  );
};

/**
 * Accessible Tab Panel
 */
export const AccessibleTabPanel = ({ children, value, index, id, ...props }) => {
  const panelId = `tabpanel-${id || index}`;
  const tabId = `tab-${id || index}`;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={panelId}
      aria-labelledby={tabId}
      {...props}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

/**
 * Accessible Tabs
 */
export const AccessibleTabs = ({ 
  value, 
  onChange, 
  tabs = [], 
  id,
  orientation = 'horizontal',
  ...props 
}) => {
  const baseId = id || Math.random().toString(36).substring(2, 9);
  
  return (
    <Tabs
      value={value}
      onChange={onChange}
      aria-label={`${baseId}-tabs`}
      orientation={orientation}
      {...props}
    >
      {tabs.map((tab, index) => (
        <Tab
          key={index}
          id={`tab-${baseId}-${index}`}
          aria-controls={`tabpanel-${baseId}-${index}`}
          label={tab.label}
          disabled={tab.disabled}
          icon={tab.icon}
          iconPosition={tab.iconPosition}
        />
      ))}
    </Tabs>
  );
};

/**
 * Accessible Alert with proper ARIA roles
 */
export const AccessibleAlert = ({ 
  severity = 'info', 
  children, 
  onClose,
  ...props 
}) => {
  return (
    <Alert
      severity={severity}
      onClose={onClose}
      role={severity === 'error' ? 'alert' : 'status'}
      aria-live={severity === 'error' ? 'assertive' : 'polite'}
      {...props}
    >
      {children}
    </Alert>
  );
};

/**
 * Accessible Toggle Switch with proper ARIA roles
 */
export const AccessibleSwitch = ({ 
  label, 
  checked, 
  onChange, 
  disabled = false,
  id,
  ...props 
}) => {
  const switchId = id || `switch-${Math.random().toString(36).substring(2, 9)}`;
  
  return (
    <FormControlLabel
      control={
        <Switch
          id={switchId}
          checked={checked}
          onChange={onChange}
          disabled={disabled}
          inputProps={{
            'aria-checked': checked,
            'aria-label': label,
            role: 'switch'
          }}
          {...props}
        />
      }
      label={label}
      htmlFor={switchId}
    />
  );
};

/**
 * Loading container with accessibility improvements
 */
export const AccessibleLoadingContainer = ({ 
  loading, 
  children, 
  loadingText = 'Loading...',
  size = 'medium', 
  ...props 
}) => {
  return (
    <Box
      {...props}
      aria-busy={loading}
      role={loading ? 'status' : undefined}
    >
      {loading ? (
        <Box 
          sx={{ 
            display: 'flex', 
            flexDirection: 'column',
            alignItems: 'center', 
            justifyContent: 'center', 
            p: 4 
          }}
        >
          <CircularProgress 
            size={size === 'small' ? 24 : size === 'large' ? 48 : 36} 
            aria-label={loadingText} 
          />
          <Typography aria-live="polite" sx={{ mt: 2 }}>
            {loadingText}
          </Typography>
        </Box>
      ) : (
        children
      )}
    </Box>
  );
};