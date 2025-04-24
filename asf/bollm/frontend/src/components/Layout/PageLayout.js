import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Container, Typography, Breadcrumbs, Link, AppBar, Toolbar,
  IconButton, Drawer, Avatar, List, ListItem, ListItemIcon,
  ListItemText, Divider, Badge, Tooltip, Menu, MenuItem, useTheme
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { useNotification } from '../../context/NotificationContext';
import { ContentLoader } from '../UI/LoadingIndicators.js';
import { FadeIn, SlideIn } from '../UI/Animations.js';
import LLMSideMenu from '../Navigation/LLMSideMenu';

// Material Icons
import {
  MenuOpen as MenuOpenIcon,
  ExitToApp as LogoutIcon,
  Notifications as NotificationsIcon,
  NavigateNext as NavigateNextIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';

// Drawer width
const drawerWidth = 280;

/**
 * A consistent page layout component with app bar, sidebar, and content area
 */
const PageLayout = ({
  title,
  breadcrumbs = [],
  actions,
  loading = false,
  user,
  children
}) => {
  const [drawerOpen, setDrawerOpen] = useState(true);
  const navigate = useNavigate();
  const theme = useTheme();

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  // Notification context
  const { showSuccess } = useNotification();

  // Profile menu state
  const [anchorEl, setAnchorEl] = useState(null);
  const openMenu = Boolean(anchorEl);

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  // Handle logout with notification
  const handleLogoutWithNotification = () => {
    handleMenuClose();
    showSuccess('Successfully logged out');
    handleLogout();
  };

  // Loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <ContentLoader height="100vh" message="Loading page..." />
      </Box>
    );
  }

  return (
    <FadeIn>
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
              LLM Management Console
            </Typography>
            <Tooltip title="Notifications">
              <IconButton color="inherit">
                <Badge badgeContent={3} color="error">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>
            {user && (
              <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
                <Typography variant="subtitle1" sx={{ mr: 1 }}>
                  {user.username}
                </Typography>
                <Tooltip title="Account settings">
                  <IconButton
                    onClick={handleMenuOpen}
                    size="small"
                    sx={{ ml: 1 }}
                    aria-controls={openMenu ? 'account-menu' : undefined}
                    aria-haspopup="true"
                    aria-expanded={openMenu ? 'true' : undefined}
                  >
                    <Avatar
                      sx={{
                        bgcolor: user.role_id === 2 ? 'secondary.main' : 'primary.main',
                        width: 32,
                        height: 32
                      }}
                    >
                      {user?.username?.charAt(0).toUpperCase() || 'U'}
                    </Avatar>
                  </IconButton>
                </Tooltip>
                <Menu
                  anchorEl={anchorEl}
                  id="account-menu"
                  open={openMenu}
                  onClose={handleMenuClose}
                  onClick={handleMenuClose}
                  transformOrigin={{ horizontal: 'right', vertical: 'top' }}
                  anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
                >
                  <MenuItem onClick={() => navigate('/llm/settings/gateway')}>
                    <ListItemIcon>
                      <SettingsIcon fontSize="small" />
                    </ListItemIcon>
                    Settings
                  </MenuItem>
                  <Divider />
                  <MenuItem onClick={handleLogoutWithNotification}>
                    <ListItemIcon>
                      <LogoutIcon fontSize="small" color="error" />
                    </ListItemIcon>
                    Logout
                  </MenuItem>
                </Menu>
              </Box>
            )}
          </Toolbar>
        </AppBar>

        {/* LLM Side Menu */}
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
            {drawerOpen ? (
              <LLMSideMenu />
            ) : (
              <List>
                <ListItem button onClick={() => navigate('/llm/dashboard')}>
                  <ListItemIcon>
                    <img src="/logo.png" alt="LLM" style={{ width: 24 }} />
                  </ListItemIcon>
                </ListItem>
              </List>
            )}

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
            {/* Page header */}
            <Box sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: { xs: 'flex-start', sm: 'center' },
              flexDirection: { xs: 'column', sm: 'row' },
              gap: { xs: 2, sm: 0 },
              mb: 3
            }}>
              <Box>
                <Typography variant="h4" component="h1" gutterBottom={breadcrumbs.length > 0}>
                  {title}
                </Typography>

                {breadcrumbs.length > 0 && (
                  <Breadcrumbs
                    separator={<NavigateNextIcon fontSize="small" />}
                    aria-label="breadcrumb"
                  >
                    <Link
                      component={RouterLink}
                      to="/llm/dashboard"
                      underline="hover"
                      color="inherit"
                    >
                      Dashboard
                    </Link>

                    {breadcrumbs.map((crumb, index) => {
                      const isLast = index === breadcrumbs.length - 1;

                      return isLast ? (
                        <Typography key={index} color="text.primary">
                          {crumb.label}
                        </Typography>
                      ) : (
                        <Link
                          key={index}
                          component={RouterLink}
                          to={crumb.path}
                          underline="hover"
                          color="inherit"
                        >
                          {crumb.label}
                        </Link>
                      );
                    })}
                  </Breadcrumbs>
                )}
              </Box>

              {actions && (
                <Box sx={{ display: 'flex', gap: 1, mt: { xs: 1, sm: 0 } }}>
                  {actions}
                </Box>
              )}
            </Box>

            {/* Page content */}
            <SlideIn direction="up">
              {children}
            </SlideIn>
          </Container>
        </Box>
      </Box>
    </FadeIn>
  );
};

export default PageLayout;
