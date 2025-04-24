// frontend/src/pages/Login.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Avatar, Box, Button, Card, CardContent, Container, CssBaseline,
  Grid, IconButton, InputAdornment, Link, TextField, Typography,
  Alert, CircularProgress
} from '@mui/material';
// Import all login logos
import loginLogo1 from '../assets/images/loginlogo1.jpg';
import loginLogo2 from '../assets/images/loginlogo2.jpg';
import loginLogo3 from '../assets/images/loginlogo3.jpg';
import loginLogo4 from '../assets/images/loginlogo4.jpg';
import loginLogo5 from '../assets/images/loginlogo5.jpg';
import loginLogo6 from '../assets/images/loginlogo6.jpg';
import defaultLogo from '../assets/images/asfmrslogo.jpg';

import {
  LockOutlined as LockOutlinedIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Email as EmailIcon,
  Lock as LockIcon
} from '@mui/icons-material';
import { useForm, Controller } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import { useAuth } from '../context/AuthContext.jsx';

// Form validation schema
const schema = yup.object({
  email: yup
    .string()
    .email('Please enter a valid email address')
    .required('Email is required'),
  password: yup
    .string()
    .required('Password is required')
    .min(6, 'Password must be at least 6 characters')
}).required();

const Login = () => {
  const [showPassword, setShowPassword] = React.useState(false);
  const [logoImage, setLogoImage] = useState(defaultLogo);
  const { login, loading } = useAuth();
  const navigate = useNavigate();

  // Select a random login logo on component mount
  useEffect(() => {
    const logos = [
      loginLogo1,
      loginLogo2,
      loginLogo3,
      loginLogo4,
      loginLogo5,
      loginLogo6
    ];
    
    // Select a random logo
    const randomIndex = Math.floor(Math.random() * logos.length);
    setLogoImage(logos[randomIndex]);
  }, []);

  // Initialize form with react-hook-form
  const { control, handleSubmit, formState: { errors } } = useForm({
    resolver: yupResolver(schema),
    defaultValues: {
      email: '',
      password: ''
    }
  });

  // Form submission handler
  const onSubmit = async (data) => {
    try {
      console.log('Submitting login with:', { username: data.email, password: data.password });
      await login({
        username: data.email, // Backend expects username field for email
        password: data.password
      });
    } catch (error) {
      console.error('Login submission error:', error);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
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
        <Box sx={{ mb: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <img 
            src={logoImage} 
            alt="Medical Research Synthesizer Logo" 
            style={{ width: 555, marginBottom: 16 }}
            onError={() => setLogoImage(defaultLogo)}
          />
        </Box>
        <Avatar sx={{ m: 1, bgcolor: 'primary.main' }}>
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5">
          Sign in
        </Typography>

        <Card sx={{ mt: 3, width: '100%' }}>
          <CardContent>
            <Box component="form" onSubmit={handleSubmit(onSubmit)} noValidate sx={{ mt: 1 }}>
              <Controller
                name="email"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    margin="normal"
                    required
                    fullWidth
                    id="email"
                    label="Email Address"
                    autoComplete="email"
                    autoFocus
                    error={!!errors.email}
                    helperText={errors.email?.message}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <EmailIcon />
                        </InputAdornment>
                      ),
                    }}
                  />
                )}
              />

              <Controller
                name="password"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    margin="normal"
                    required
                    fullWidth
                    label="Password"
                    type={showPassword ? 'text' : 'password'}
                    id="password"
                    autoComplete="current-password"
                    error={!!errors.password}
                    helperText={errors.password?.message}
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
                )}
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
                disabled={loading}
              >
                {loading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : (
                  'Sign In'
                )}
              </Button>

              <Grid container justifyContent="center">
                <Grid item>
                  <Link component={RouterLink} to="/register" variant="body2">
                    {"Don't have an account? Sign Up"}
                  </Link>
                </Grid>
              </Grid>
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default Login;
