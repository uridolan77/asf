import React, { useState, useEffect } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Avatar, Box, Button, Card, CardContent, Container, CssBaseline,
  FormControl, Grid, IconButton, InputAdornment, InputLabel,
  Link, MenuItem, Select, TextField, Typography, Alert
} from '@mui/material';
import {
  PersonAddOutlined as PersonAddOutlinedIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Email as EmailIcon,
  Person as PersonIcon,
  Lock as LockIcon,
  Badge as BadgeIcon
} from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { useAuth } from '../context/AuthContext.jsx';
import { useNotification } from '../context/NotificationContext.jsx';

const theme = createTheme({
  palette: {
    primary: {
      main: '#3498db',
    },
    secondary: {
      main: '#2ecc71',
    },
    error: {
      main: '#e74c3c',
    },
    background: {
      default: '#f5f7fa',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 10px rgba(0,0,0,0.08)',
          borderRadius: '8px',
        },
      },
    },
  },
});

const Register = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [roleId, setRoleId] = useState('');
  const [roles, setRoles] = useState([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();
  const { register, api } = useAuth();
  const { showSuccess, showError } = useNotification();

  useEffect(() => {
    // Set default roles since backend may not have this endpoint ready
    setRoles([
      { id: 1, name: 'User' },
      { id: 2, name: 'Admin' }
    ]);

    // Try to fetch roles from backend if available
    async function fetchRoles() {
      try {
        const response = await api.get('/api/roles');
        if (response.data) {
          setRoles(response.data);
        }
      } catch (err) {
        // Silently fail and use default roles
        console.warn('Could not fetch roles:', err);
      }
    }
    
    fetchRoles();
  }, [api]);

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    
    // Validation
    if (!username) {
      setError('Username is required');
      return;
    }
    if (!email) {
      setError('Email is required');
      return;
    }
    if (!email.includes('@')) {
      setError('Invalid email format');
      return;
    }
    if (!password) {
      setError('Password is required');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }
    if (!roleId) {
      setError('Please select a role');
      return;
    }

    try {
      const success = await register({
        username,
        email,
        password,
        role_id: parseInt(roleId)
      });
      
      if (success) {
        setSuccess('Registration successful! Redirecting to login...');
        showSuccess('Registration successful!');
        setTimeout(() => navigate('/login'), 1500);
      }
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'Registration failed';
      setError(errorMessage);
      showError(errorMessage);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <ThemeProvider theme={theme}>
      <Container component="main" maxWidth="xs">
        <CssBaseline />
        <Box
          sx={{
            marginTop: 8,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
            <PersonAddOutlinedIcon />
          </Avatar>
          <Typography component="h1" variant="h5">
            Sign up
          </Typography>

          <Card sx={{ mt: 3, width: '100%' }}>
            <CardContent>
              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}

              {success && (
                <Alert severity="success" sx={{ mb: 2 }}>
                  {success}
                </Alert>
              )}

              <Box component="form" onSubmit={handleRegister} noValidate sx={{ mt: 1 }}>
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="username"
                  label="Username"
                  name="username"
                  autoComplete="username"
                  autoFocus
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <PersonIcon />
                      </InputAdornment>
                    ),
                  }}
                />
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="email"
                  label="Email Address"
                  name="email"
                  autoComplete="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <EmailIcon />
                      </InputAdornment>
                    ),
                  }}
                />
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  name="password"
                  label="Password"
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <LockIcon />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton
                          aria-label="toggle password visibility"
                          onClick={togglePasswordVisibility}
                          edge="end"
                        >
                          {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />
                <FormControl fullWidth margin="normal" required>
                  <InputLabel id="role-label">Role</InputLabel>
                  <Select
                    labelId="role-label"
                    id="role"
                    value={roleId}
                    label="Role"
                    onChange={e => setRoleId(e.target.value)}
                    startAdornment={
                      <InputAdornment position="start">
                        <BadgeIcon />
                      </InputAdornment>
                    }
                  >
                    <MenuItem value=""><em>Select a role</em></MenuItem>
                    {roles.map(role => (
                      <MenuItem key={role.id} value={role.id}>{role.name}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  color="secondary"
                  sx={{ mt: 3, mb: 2 }}
                >
                  Sign Up
                </Button>
                <Grid container justifyContent="center">
                  <Grid item>
                    <Link component={RouterLink} to="/login" variant="body2">
                      {"Already have an account? Sign in"}
                    </Link>
                  </Grid>
                </Grid>
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Container>
    </ThemeProvider>
  );
};

export default Register;