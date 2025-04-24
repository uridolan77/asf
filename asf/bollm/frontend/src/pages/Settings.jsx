import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Button, Card, CardContent, CardHeader, Divider, FormControlLabel,
  Grid, IconButton, InputAdornment, MenuItem, Paper, Select, Switch,
  TextField, Typography, Alert, FormControl, InputLabel
} from '@mui/material';
import {
  Save as SaveIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Person as PersonIcon,
  Email as EmailIcon,
  Lock as LockIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';
import { useAuth } from '../context/AuthContext.jsx';
import { useNotification } from '../context/NotificationContext.jsx';

const Settings = () => {
  const { user, api, updateProfile } = useAuth();
  const { showSuccess, showError } = useNotification();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [profileData, setProfileData] = useState({
    username: '',
    email: '',
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [systemSettings, setSystemSettings] = useState({
    sessionTimeout: 60,
    enableNotifications: true,
    darkMode: false,
    language: 'en'
  });
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserData = async () => {
      if (!user) {
        navigate('/');
        return;
      }

      try {
        setProfileData({
          ...profileData,
          username: user.username,
          email: user.email
        });

        // If user is admin, fetch system settings
        if (user.role_id === 2) {
          try {
            const settingsResponse = await api.get('/api/settings');
            if (settingsResponse.data) {
              setSystemSettings(settingsResponse.data);
            }
          } catch (settingsErr) {
            console.error('Failed to fetch settings:', settingsErr);
            // Don't show error - settings may not exist yet
          }
        }
        setError('');
      } catch (err) {
        console.error('Failed to fetch user data:', err);
        showError('Failed to load user data. You may need to log in again.');
        if (err.response && (err.response.status === 401 || err.response.status === 403)) {
          navigate('/');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, [api, navigate, profileData, user, showError]);

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    setProfileData({
      ...profileData,
      [name]: value
    });
  };

  const handleSystemSettingChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSystemSettings({
      ...systemSettings,
      [name]: type === 'checkbox' ? checked : value
    });
  };

  const validateProfileForm = () => {
    // Reset errors and success
    setError('');
    setSuccess('');

    // Password validation
    if (profileData.newPassword) {
      if (!profileData.currentPassword) {
        setError('Current password is required to set a new password');
        return false;
      }
      if (profileData.newPassword.length < 6) {
        setError('New password must be at least 6 characters');
        return false;
      }
      if (profileData.newPassword !== profileData.confirmPassword) {
        setError('New passwords do not match');
        return false;
      }
    }

    return true;
  };

  const handleProfileSubmit = async (e) => {
    e.preventDefault();

    if (!validateProfileForm()) return;

    try {
      const updateData = {
        username: profileData.username,
        email: profileData.email
      };

      // Only include password fields if user is changing password
      if (profileData.newPassword) {
        updateData.current_password = profileData.currentPassword;
        updateData.new_password = profileData.newPassword;
      }

      await api.put('/api/me', updateData);

      setSuccess('Profile updated successfully');
      showSuccess('Profile updated successfully');
      
      // Clear password fields after successful update
      setProfileData({
        ...profileData,
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      });
    } catch (err) {
      console.error('Failed to update profile:', err);
      const errorMessage = err.response?.data?.detail || 'Failed to update profile';
      setError(errorMessage);
      showError(errorMessage);
    }
  };

  const handleSystemSettingsSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    try {
      await api.put('/api/settings', systemSettings);
      setSuccess('System settings updated successfully');
      showSuccess('System settings updated successfully');
    } catch (err) {
      console.error('Failed to update system settings:', err);
      setError('Failed to update system settings');
      showError('Failed to update system settings');
    }
  };

  // State for password visibility
  const [showPassword, setShowPassword] = useState({
    current: false,
    new: false,
    confirm: false
  });

  const togglePasswordVisibility = (field) => {
    setShowPassword({
      ...showPassword,
      [field]: !showPassword[field]
    });
  };

  return (
    <PageLayout
      title="Settings"
      breadcrumbs={[{ label: 'Settings', path: '/settings' }]}
      loading={loading}
      user={user}
    >
      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Profile Settings */}
        <Grid item xs={12} md={6}>
          <Card sx={{ mb: { xs: 3, md: 0 } }}>
            <CardHeader
              title="Profile Settings"
              avatar={<PersonIcon color="primary" />}
            />
            <Divider />
            <CardContent>
              <form onSubmit={handleProfileSubmit}>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Username"
                      name="username"
                      value={profileData.username}
                      onChange={handleProfileChange}
                      required
                      variant="outlined"
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <PersonIcon />
                          </InputAdornment>
                        ),
                      }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Email"
                      name="email"
                      type="email"
                      value={profileData.email}
                      onChange={handleProfileChange}
                      required
                      variant="outlined"
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <EmailIcon />
                          </InputAdornment>
                        ),
                      }}
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                      Change Password
                    </Typography>
                    <Divider sx={{ mb: 2 }} />
                  </Grid>

                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Current Password"
                      name="currentPassword"
                      type={showPassword.current ? 'text' : 'password'}
                      value={profileData.currentPassword}
                      onChange={handleProfileChange}
                      variant="outlined"
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <LockIcon />
                          </InputAdornment>
                        ),
                        endAdornment: (
                          <InputAdornment position="end">
                            <IconButton
                              onClick={() => togglePasswordVisibility('current')}
                              edge="end"
                            >
                              {showPassword.current ? <VisibilityOffIcon /> : <VisibilityIcon />}
                            </IconButton>
                          </InputAdornment>
                        ),
                      }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="New Password"
                      name="newPassword"
                      type={showPassword.new ? 'text' : 'password'}
                      value={profileData.newPassword}
                      onChange={handleProfileChange}
                      variant="outlined"
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <LockIcon />
                          </InputAdornment>
                        ),
                        endAdornment: (
                          <InputAdornment position="end">
                            <IconButton
                              onClick={() => togglePasswordVisibility('new')}
                              edge="end"
                            >
                              {showPassword.new ? <VisibilityOffIcon /> : <VisibilityIcon />}
                            </IconButton>
                          </InputAdornment>
                        ),
                      }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Confirm New Password"
                      name="confirmPassword"
                      type={showPassword.confirm ? 'text' : 'password'}
                      value={profileData.confirmPassword}
                      onChange={handleProfileChange}
                      variant="outlined"
                      InputProps={{
                        startAdornment: (
                          <InputAdornment position="start">
                            <LockIcon />
                          </InputAdornment>
                        ),
                        endAdornment: (
                          <InputAdornment position="end">
                            <IconButton
                              onClick={() => togglePasswordVisibility('confirm')}
                              edge="end"
                            >
                              {showPassword.confirm ? <VisibilityOffIcon /> : <VisibilityIcon />}
                            </IconButton>
                          </InputAdornment>
                        ),
                      }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <Button
                      type="submit"
                      variant="contained"
                      color="primary"
                      startIcon={<SaveIcon />}
                    >
                      Update Profile
                    </Button>
                  </Grid>
                </Grid>
              </form>
            </CardContent>
          </Card>
        </Grid>

        {/* System Settings (Admin Only) */}
        <Grid item xs={12} md={6}>
          {user && user.role_id === 2 && (
            <Card>
              <CardHeader
                title="System Settings"
                avatar={<SettingsIcon color="primary" />}
              />
              <Divider />
              <CardContent>
                <form onSubmit={handleSystemSettingsSubmit}>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Session Timeout (minutes)"
                        name="sessionTimeout"
                        type="number"
                        value={systemSettings.sessionTimeout}
                        onChange={handleSystemSettingChange}
                        variant="outlined"
                        InputProps={{ inputProps: { min: 15, max: 480 } }}
                        required
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControl fullWidth>
                        <InputLabel id="language-label">Default Language</InputLabel>
                        <Select
                          labelId="language-label"
                          id="language"
                          name="language"
                          value={systemSettings.language}
                          label="Default Language"
                          onChange={handleSystemSettingChange}
                        >
                          <MenuItem value="en">English</MenuItem>
                          <MenuItem value="es">Spanish</MenuItem>
                          <MenuItem value="fr">French</MenuItem>
                          <MenuItem value="de">German</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={systemSettings.enableNotifications}
                            onChange={handleSystemSettingChange}
                            name="enableNotifications"
                            color="primary"
                          />
                        }
                        label="Enable Email Notifications"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={systemSettings.darkMode}
                            onChange={handleSystemSettingChange}
                            name="darkMode"
                            color="primary"
                          />
                        }
                        label="Enable Dark Mode"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <Button
                        type="submit"
                        variant="contained"
                        color="primary"
                        startIcon={<SaveIcon />}
                      >
                        Save System Settings
                      </Button>
                    </Grid>
                  </Grid>
                </form>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </PageLayout>
  );
};

export default Settings;