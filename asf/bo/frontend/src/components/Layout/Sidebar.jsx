import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Collapse,
  Typography,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  People as PeopleIcon,
  Settings as SettingsIcon,
  Search as SearchIcon,
  Book as BookIcon,
  LocalHospital as LocalHospitalIcon,
  Analytics as AnalyticsIcon,
  Psychology as PsychologyIcon,
  Cloud as CloudIcon,
  Biotech as BiotechIcon,
  Article as ArticleIcon,
  ExpandLess,
  ExpandMore,
  SmartToy as SmartToyIcon,
  Storage as StorageIcon,
  Dns as DnsIcon,
  BarChart as BarChartIcon,
  Devices as DevicesIcon,
  Timelapse as TimelapseIcon
} from '@mui/icons-material';
import { useAuth } from '../../context/AuthContext.jsx';

// Sidebar width
const drawerWidth = 240;

// Sidebar component
const Sidebar = ({ open, onClose }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  
  // State for expanded menu items
  const [expandedItems, setExpandedItems] = React.useState({
    llm: true,
    medical: false,
    ml: false
  });
  
  // Toggle expanded state
  const toggleExpand = (item) => {
    setExpandedItems(prev => ({
      ...prev,
      [item]: !prev[item]
    }));
  };
  
  // Check if a route is active
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  // Navigate to a route
  const navigateTo = (path) => {
    navigate(path);
    if (onClose) {
      onClose();
    }
  };
  
  // Menu items
  const menuItems = [
    {
      text: 'Dashboard',
      icon: <DashboardIcon />,
      path: '/dashboard'
    },
    {
      text: 'Users',
      icon: <PeopleIcon />,
      path: '/users',
      adminOnly: true
    },
    {
      text: 'LLM Services',
      icon: <SmartToyIcon />,
      expandKey: 'llm',
      subItems: [
        {
          text: 'LLM Management',
          icon: <CloudIcon />,
          path: '/llm-management'
        },
        {
          text: 'MCP Dashboard',
          icon: <DnsIcon />,
          path: '/mcp-dashboard'
        },
        {
          text: 'Progress Tracking',
          icon: <TimelapseIcon />,
          path: '/llm-progress'
        },
        {
          text: 'Document Processing',
          icon: <ArticleIcon />,
          path: '/document-processing'
        }
      ]
    },
    {
      text: 'Medical Data',
      icon: <LocalHospitalIcon />,
      expandKey: 'medical',
      subItems: [
        {
          text: 'Search',
          icon: <SearchIcon />,
          path: '/search'
        },
        {
          text: 'Knowledge Base',
          icon: <BookIcon />,
          path: '/knowledge-base'
        },
        {
          text: 'Clinical Data',
          icon: <BiotechIcon />,
          path: '/clinical-data'
        },
        {
          text: 'Clients Management',
          icon: <StorageIcon />,
          path: '/clients-management'
        }
      ]
    },
    {
      text: 'ML Services',
      icon: <PsychologyIcon />,
      expandKey: 'ml',
      subItems: [
        {
          text: 'ML Dashboard',
          icon: <BarChartIcon />,
          path: '/ml-services'
        },
        {
          text: 'Analysis',
          icon: <AnalyticsIcon />,
          path: '/analysis'
        }
      ]
    },
    {
      text: 'Settings',
      icon: <SettingsIcon />,
      path: '/settings'
    }
  ];
  
  // Render sidebar content
  const sidebarContent = (
    <Box sx={{ overflow: 'auto' }}>
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Medical BO
        </Typography>
      </Box>
      
      <Divider />
      
      <List>
        {menuItems.map((item) => {
          // Skip admin-only items for non-admin users
          if (item.adminOnly && (!user || !user.is_admin)) {
            return null;
          }
          
          // If item has subitems, render as expandable
          if (item.subItems) {
            return (
              <React.Fragment key={item.text}>
                <ListItem button onClick={() => toggleExpand(item.expandKey)}>
                  <ListItemIcon>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                  {expandedItems[item.expandKey] ? <ExpandLess /> : <ExpandMore />}
                </ListItem>
                
                <Collapse in={expandedItems[item.expandKey]} timeout="auto" unmountOnExit>
                  <List component="div" disablePadding>
                    {item.subItems.map((subItem) => (
                      <ListItem 
                        button 
                        key={subItem.text}
                        onClick={() => navigateTo(subItem.path)}
                        selected={isActive(subItem.path)}
                        sx={{ pl: 4 }}
                      >
                        <ListItemIcon>{subItem.icon}</ListItemIcon>
                        <ListItemText primary={subItem.text} />
                      </ListItem>
                    ))}
                  </List>
                </Collapse>
              </React.Fragment>
            );
          }
          
          // Regular menu item
          return (
            <ListItem 
              button 
              key={item.text}
              onClick={() => navigateTo(item.path)}
              selected={isActive(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItem>
          );
        })}
      </List>
    </Box>
  );
  
  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        display: { xs: 'none', md: 'block' },
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
      open
    >
      {sidebarContent}
    </Drawer>
  );
};

export default Sidebar;
