// ClinicalData.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

// Material UI imports
import {
  Box, Container, Grid, Paper, Typography, Button, Divider,
  Card, CardContent, CardHeader, List, ListItem, ListItemIcon,
  ListItemText, CircularProgress, Chip, AppBar, Toolbar,
  IconButton, Drawer, Avatar, Tabs, Tab, Alert
} from '@mui/material';

// Material Icons
import {
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  Settings as SettingsIcon,
  Search as SearchIcon,
  Book as BookIcon,
  Biotech as BiotechIcon,
  Science as ScienceIcon,
  MenuOpen as MenuOpenIcon,
  ExitToApp as LogoutIcon,
  Notifications as NotificationsIcon,
  LocalHospital as LocalHospitalIcon,
  MedicalServices as MedicalServicesIcon
} from '@mui/icons-material';

// Custom theme
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
    h1: {
      fontWeight: 500,
    },
    h2: {
      fontWeight: 500,
    },
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

// Import components
import TermSearch from '../components/ClinicalData/TermSearch';
import ConceptExplorer from '../components/ClinicalData/ConceptExplorer';
import TrialMapping from '../components/ClinicalData/TrialMapping';
import SemanticSearch from '../components/ClinicalData/SemanticSearch';

const drawerWidth = 260;

/**
 * Clinical Data page component
 * Provides access to clinical data functionality including:
 * - Term search with SNOMED CT concepts and clinical trials
 * - Concept explorer for finding trials by medical concept
 * - Trial mapping for mapping conditions to SNOMED CT
 * - Semantic search with term expansion
 */
const ClinicalData = () => {
  // State
  const { user, logout } = useAuth();
  const { showError } = useNotification();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [tabValue, setTabValue] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const navigate = useNavigate();

  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Handle logout
  const handleLogout = () => {
    logout();
  };

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex', minHeight: '100vh' }}>
        {/* App Bar */}
        <AppBar
          position="fixed"
          sx={{
            zIndex: (theme) => theme.zIndex.drawer + 1,
            boxShadow: '0 2px 10px rgba(0,0,0,0.08)'
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={toggleDrawer}
              sx={{ mr: 2 }}
            >
              <MenuOpenIcon />
            </IconButton>
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              Medical Research Synthesizer
            </Typography>
            <IconButton color="inherit">
              <NotificationsIcon />
            </IconButton>
            {user && (
              <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
                <Typography variant="subtitle1" sx={{ mr: 1 }}>
                  {user.username}
                </Typography>
                <Avatar
                  sx={{
                    bgcolor: user.role_id === 2 ? 'secondary.main' : 'primary.main',
                    width: 32,
                    height: 32
                  }}
                >
                  {user.username.charAt(0).toUpperCase()}
                </Avatar>
              </Box>
            )}
          </Toolbar>
        </AppBar>

        {/* Sidebar */}
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: {
              width: drawerWidth,
              boxSizing: 'border-box',
              borderRight: '1px solid rgba(0, 0, 0, 0.12)',
              boxShadow: 'none',
              ...(drawerOpen ? {} : { width: theme.spacing(7) })
            },
            ...(drawerOpen ? {} : { width: theme.spacing(7) })
          }}
          open={drawerOpen}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto', mt: 2 }}>
            <List>
              <ListItem button onClick={() => navigate('/dashboard')}>
                <ListItemIcon>
                  <DashboardIcon />
                </ListItemIcon>
                {drawerOpen && <ListItemText primary="Dashboard" />}
              </ListItem>

              <ListItem button onClick={() => navigate('/pico-search')}>
                <ListItemIcon>
                  <SearchIcon />
                </ListItemIcon>
                {drawerOpen && <ListItemText primary="PICO Search" />}
              </ListItem>

              <ListItem button onClick={() => navigate('/knowledge-base')}>
                <ListItemIcon>
                  <BookIcon />
                </ListItemIcon>
                {drawerOpen && <ListItemText primary="Knowledge Base" />}
              </ListItem>

              <ListItem button selected onClick={() => navigate('/clinical-data')}>
                <ListItemIcon>
                  <MedicalServicesIcon color="primary" />
                </ListItemIcon>
                {drawerOpen && <ListItemText primary="Clinical Data" />}
              </ListItem>

              {user && user.role_id === 2 && (
                <ListItem button onClick={() => navigate('/users')}>
                  <ListItemIcon>
                    <PeopleIcon />
                  </ListItemIcon>
                  {drawerOpen && <ListItemText primary="Users" />}
                </ListItem>
              )}

              <ListItem button onClick={() => navigate('/settings')}>
                <ListItemIcon>
                  <SettingsIcon />
                </ListItemIcon>
                {drawerOpen && <ListItemText primary="Settings" />}
              </ListItem>
            </List>

            <Divider sx={{ my: 2 }} />

            <List>
              <ListItem button onClick={handleLogout}>
                <ListItemIcon>
                  <LogoutIcon color="error" />
                </ListItemIcon>
                {drawerOpen && <ListItemText primary="Logout" sx={{ color: theme.palette.error.main }} />}
              </ListItem>
            </List>
          </Box>
        </Drawer>

        {/* Main content */}
        <Box component="main" sx={{ flexGrow: 1, p: 3, pt: 10 }}>
          <Container maxWidth="xl">
            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            <Card sx={{ mb: 4 }}>
              <CardHeader
                title="Clinical Data Integration"
                subheader="Connecting medical terminology with clinical trials data"
              />
              <Divider />
              <CardContent>
                <Typography variant="body1" paragraph>
                  These tools combine medical terminology (SNOMED CT) with clinical trials data to provide
                  powerful insights and connections between standardized medical concepts and real-world clinical research.
                </Typography>

                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                  <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    aria-label="clinical data tabs"
                  >
                    <Tab label="Term Search" icon={<SearchIcon />} iconPosition="start" />
                    <Tab label="Concept Explorer" icon={<ScienceIcon />} iconPosition="start" />
                    <Tab label="Trial Mapping" icon={<LocalHospitalIcon />} iconPosition="start" />
                    <Tab label="Semantic Search" icon={<BiotechIcon />} iconPosition="start" />
                  </Tabs>
                </Box>

                {/* Tab content */}
                {tabValue === 0 && <TermSearch />}
                {tabValue === 1 && <ConceptExplorer />}
                {tabValue === 2 && <TrialMapping />}
                {tabValue === 3 && <SemanticSearch />}
              </CardContent>
            </Card>
          </Container>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default ClinicalData;