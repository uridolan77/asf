import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext.jsx';
import {
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Collapse,
  Typography,
  Box
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Cloud as CloudIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  CellTower as CellTowerIcon,
  Code as CodeIcon,
  SmartToy as SmartToyIcon,
  Speed as SpeedIcon,
  PsychologyAlt as PsychologyAltIcon,
  AutoFixHigh as AutoFixHighIcon,
  BiotechOutlined as BiotechOutlinedIcon,
  Settings as SettingsIcon,
  ExpandLess,
  ExpandMore,
  AccountTree as AccountTreeIcon,
  MonitorHeart as MonitorHeartIcon,
  VpnKey as VpnKeyIcon,
  Key as KeyIcon,
  Analytics as AnalyticsIcon,
  QueryStats as QueryStatsIcon,
  BarChart as BarChartIcon,
  Dns as DnsIcon,
  DataObject as DataObjectIcon,
  ViewTimeline as ViewTimelineIcon,
  Notifications as NotificationsIcon,
  Extension as ExtensionIcon,
  EditNote as EditNoteIcon,
  Api as ApiIcon,
  Terminal as TerminalIcon,
  Chat as ChatIcon,
  Timeline as TimelineIcon,
  Logout as LogoutIcon
} from '@mui/icons-material';

const LLMSideMenu = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { logout } = useAuth();
  
  // State for each expandable section
  const [providerOpen, setProviderOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [requestOpen, setRequestOpen] = useState(false);
  const [playgroundOpen, setPlaygroundOpen] = useState(false);
  const [cacheOpen, setCacheOpen] = useState(false);
  const [observabilityOpen, setObservabilityOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [serviceConfigOpen, setServiceConfigOpen] = useState(false);

  // Toggle handlers
  const toggleProvider = () => setProviderOpen(!providerOpen);
  const toggleModel = () => setModelOpen(!modelOpen);
  const toggleRequest = () => setRequestOpen(!requestOpen);
  const togglePlayground = () => setPlaygroundOpen(!playgroundOpen);
  const toggleCache = () => setCacheOpen(!cacheOpen);
  const toggleObservability = () => setObservabilityOpen(!observabilityOpen);
  const toggleAdvanced = () => setAdvancedOpen(!advancedOpen);
  const toggleSettings = () => setSettingsOpen(!settingsOpen);
  const toggleServiceConfig = () => setServiceConfigOpen(!serviceConfigOpen);

  // Check if a menu item is active
  const isActive = (path) => location.pathname === path;
  const isGroupActive = (paths) => paths.some(path => location.pathname.startsWith(path));

  // Menu item styles
  const menuItemStyles = (active) => ({
    pl: 2,
    ...(active && {
      bgcolor: 'action.selected',
      borderLeft: '4px solid',
      borderColor: 'primary.main',
      '&:hover': {
        bgcolor: 'action.hover',
      }
    })
  });

  // Submenu item styles
  const submenuItemStyles = (active) => ({
    pl: 4,
    ...(active && {
      bgcolor: 'action.selected',
      borderLeft: '4px solid',
      borderColor: 'primary.main',
      '&:hover': {
        bgcolor: 'action.hover',
      }
    })
  });

  return (
    <List component="nav" sx={{ width: '100%', p: 0 }}>
      {/* Dashboard */}
      <ListItem disablePadding>
        <ListItemButton
          onClick={() => navigate('/llm/dashboard')}
          sx={menuItemStyles(isActive('/llm/dashboard'))}
        >
          <ListItemIcon>
            <DashboardIcon color={isActive('/llm/dashboard') ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Dashboard" 
            primaryTypographyProps={{ 
              fontWeight: isActive('/llm/dashboard') ? 'bold' : 'normal' 
            }}
          />
        </ListItemButton>
      </ListItem>

      {/* Provider Management */}
      <ListItem disablePadding>
        <ListItemButton 
          onClick={toggleProvider}
          sx={menuItemStyles(isGroupActive(['/llm/providers']))}
        >
          <ListItemIcon>
            <CloudIcon color={isGroupActive(['/llm/providers']) ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Provider Management" 
            primaryTypographyProps={{ 
              fontWeight: isGroupActive(['/llm/providers']) ? 'bold' : 'normal' 
            }}
          />
          {providerOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
      </ListItem>
      <Collapse in={providerOpen} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/providers'))}
            onClick={() => navigate('/llm/providers')}
          >
            <ListItemIcon>
              <AccountTreeIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Providers List" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/providers/config'))}
            onClick={() => navigate('/llm/providers/config')}
          >
            <ListItemIcon>
              <SettingsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Configuration" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/providers/keys'))}
            onClick={() => navigate('/llm/providers/keys')}
          >
            <ListItemIcon>
              <VpnKeyIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="API Keys" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/providers/health'))}
            onClick={() => navigate('/llm/providers/health')}
          >
            <ListItemIcon>
              <MonitorHeartIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Health Status" />
          </ListItemButton>
        </List>
      </Collapse>

      {/* Model Management */}
      <ListItem disablePadding>
        <ListItemButton 
          onClick={toggleModel}
          sx={menuItemStyles(isGroupActive(['/llm/models']))}
        >
          <ListItemIcon>
            <MemoryIcon color={isGroupActive(['/llm/models']) ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Model Management" 
            primaryTypographyProps={{ 
              fontWeight: isGroupActive(['/llm/models']) ? 'bold' : 'normal' 
            }}
          />
          {modelOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
      </ListItem>
      <Collapse in={modelOpen} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/models'))}
            onClick={() => navigate('/llm/models')}
          >
            <ListItemIcon>
              <DataObjectIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Models List" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/models/parameters'))}
            onClick={() => navigate('/llm/models/parameters')}
          >
            <ListItemIcon>
              <SettingsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Parameters" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/models/performance'))}
            onClick={() => navigate('/llm/models/performance')}
          >
            <ListItemIcon>
              <SpeedIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Performance" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/models/usage'))}
            onClick={() => navigate('/llm/models/usage')}
          >
            <ListItemIcon>
              <BarChartIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Usage" />
          </ListItemButton>
        </List>
      </Collapse>

      {/* Request Management */}
      <ListItem disablePadding>
        <ListItemButton 
          onClick={toggleRequest}
          sx={menuItemStyles(isGroupActive(['/llm/requests']))}
        >
          <ListItemIcon>
            <CodeIcon color={isGroupActive(['/llm/requests']) ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Request Management" 
            primaryTypographyProps={{ 
              fontWeight: isGroupActive(['/llm/requests']) ? 'bold' : 'normal' 
            }}
          />
          {requestOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
      </ListItem>
      <Collapse in={requestOpen} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/requests/logs'))}
            onClick={() => navigate('/llm/requests/logs')}
          >
            <ListItemIcon>
              <ViewTimelineIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Logs" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/requests/analysis'))}
            onClick={() => navigate('/llm/requests/analysis')}
          >
            <ListItemIcon>
              <QueryStatsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Analysis" />
          </ListItemButton>
        </List>
      </Collapse>

      {/* Playground */}
      <ListItem disablePadding>
        <ListItemButton 
          onClick={togglePlayground}
          sx={menuItemStyles(isGroupActive(['/llm/playground']))}
        >
          <ListItemIcon>
            <SmartToyIcon color={isGroupActive(['/llm/playground']) ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Playground" 
            primaryTypographyProps={{ 
              fontWeight: isGroupActive(['/llm/playground']) ? 'bold' : 'normal' 
            }}
          />
          {playgroundOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
      </ListItem>
      <Collapse in={playgroundOpen} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/playground/text'))}
            onClick={() => navigate('/llm/playground/text')}
          >
            <ListItemIcon>
              <EditNoteIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Text Completion" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/playground/chat'))}
            onClick={() => navigate('/llm/playground/chat')}
          >
            <ListItemIcon>
              <ChatIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Chat" />
          </ListItemButton>
        </List>
      </Collapse>

      {/* Cache Management */}
      <ListItem disablePadding>
        <ListItemButton 
          onClick={toggleCache}
          sx={menuItemStyles(isGroupActive(['/llm/cache']))}
        >
          <ListItemIcon>
            <StorageIcon color={isGroupActive(['/llm/cache']) ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Cache Management" 
            primaryTypographyProps={{ 
              fontWeight: isGroupActive(['/llm/cache']) ? 'bold' : 'normal' 
            }}
          />
          {cacheOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
      </ListItem>
      <Collapse in={cacheOpen} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/cache/stats'))}
            onClick={() => navigate('/llm/cache/stats')}
          >
            <ListItemIcon>
              <AnalyticsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Cache Statistics" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/cache/config'))}
            onClick={() => navigate('/llm/cache/config')}
          >
            <ListItemIcon>
              <SettingsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Configuration" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/cache/semantic'))}
            onClick={() => navigate('/llm/cache/semantic')}
          >
            <ListItemIcon>
              <PsychologyAltIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Semantic Cache" />
          </ListItemButton>
        </List>
      </Collapse>

      {/* Observability */}
      <ListItem disablePadding>
        <ListItemButton 
          onClick={toggleObservability}
          sx={menuItemStyles(isGroupActive(['/llm/observability']))}
        >
          <ListItemIcon>
            <CellTowerIcon color={isGroupActive(['/llm/observability']) ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Observability" 
            primaryTypographyProps={{ 
              fontWeight: isGroupActive(['/llm/observability']) ? 'bold' : 'normal' 
            }}
          />
          {observabilityOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
      </ListItem>
      <Collapse in={observabilityOpen} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/observability/metrics'))}
            onClick={() => navigate('/llm/observability/metrics')}
          >
            <ListItemIcon>
              <BarChartIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Metrics" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/observability/traces'))}
            onClick={() => navigate('/llm/observability/traces')}
          >
            <ListItemIcon>
              <ViewTimelineIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Traces" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/observability/alerts'))}
            onClick={() => navigate('/llm/observability/alerts')}
          >
            <ListItemIcon>
              <NotificationsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Alerts" />
          </ListItemButton>
        </List>
      </Collapse>

      {/* Advanced Features */}
      <ListItem disablePadding>
        <ListItemButton 
          onClick={toggleAdvanced}
          sx={menuItemStyles(isGroupActive(['/llm/advanced']))}
        >
          <ListItemIcon>
            <AutoFixHighIcon color={isGroupActive(['/llm/advanced']) ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Advanced Features" 
            primaryTypographyProps={{ 
              fontWeight: isGroupActive(['/llm/advanced']) ? 'bold' : 'normal' 
            }}
          />
          {advancedOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
      </ListItem>
      <Collapse in={advancedOpen} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/advanced/cl-peft'))}
            onClick={() => navigate('/llm/advanced/cl-peft')}
          >
            <ListItemIcon>
              <BiotechOutlinedIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="CL-PEFT" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/advanced/dspy'))}
            onClick={() => navigate('/llm/advanced/dspy')}
          >
            <ListItemIcon>
              <PsychologyAltIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="DSPy" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/advanced/plugins'))}
            onClick={() => navigate('/llm/advanced/plugins')}
          >
            <ListItemIcon>
              <ExtensionIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Plugins" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/advanced/interventions'))}
            onClick={() => navigate('/llm/advanced/interventions')}
          >
            <ListItemIcon>
              <EditNoteIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Interventions" />
          </ListItemButton>
        </List>
      </Collapse>

      {/* Settings */}
      <ListItem disablePadding>
        <ListItemButton 
          onClick={toggleSettings}
          sx={menuItemStyles(isGroupActive(['/llm/settings']))}
        >
          <ListItemIcon>
            <SettingsIcon color={isGroupActive(['/llm/settings']) ? 'primary' : 'inherit'} />
          </ListItemIcon>
          <ListItemText 
            primary="Settings" 
            primaryTypographyProps={{ 
              fontWeight: isGroupActive(['/llm/settings']) ? 'bold' : 'normal' 
            }}
          />
          {settingsOpen ? <ExpandLess /> : <ExpandMore />}
        </ListItemButton>
      </ListItem>
      <Collapse in={settingsOpen} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/settings/gateway'))}
            onClick={() => navigate('/llm/settings/gateway')}
          >
            <ListItemIcon>
              <DnsIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Gateway Config" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/settings/api-keys'))}
            onClick={() => navigate('/llm/settings/api-keys')}
          >
            <ListItemIcon>
              <KeyIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="API Keys" />
          </ListItemButton>
          <ListItemButton 
            sx={submenuItemStyles(isActive('/llm/settings/environment'))}
            onClick={() => navigate('/llm/settings/environment')}
          >
            <ListItemIcon>
              <TerminalIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Environment" />
          </ListItemButton>
          <ListItemButton 
            onClick={toggleServiceConfig}
            sx={submenuItemStyles(isGroupActive(['/llm/settings/service-config']))}
          >
            <ListItemIcon>
              <ApiIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Service Configuration" />
            {serviceConfigOpen ? <ExpandLess /> : <ExpandMore />}
          </ListItemButton>
          <Collapse in={serviceConfigOpen} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              <ListItemButton 
                sx={{ pl: 6 }}
                onClick={() => navigate('/llm/settings/service-config')}
                selected={isActive('/llm/settings/service-config')}
              >
                <ListItemText primary="General" />
              </ListItemButton>
              <ListItemButton 
                sx={{ pl: 6 }}
                onClick={() => navigate('/llm/settings/service-config/caching')}
                selected={isActive('/llm/settings/service-config/caching')}
              >
                <ListItemText primary="Service Caching" />
              </ListItemButton>
              <ListItemButton 
                sx={{ pl: 6 }}
                onClick={() => navigate('/llm/settings/service-config/resilience')}
                selected={isActive('/llm/settings/service-config/resilience')}
              >
                <ListItemText primary="Resilience" />
              </ListItemButton>
              <ListItemButton 
                sx={{ pl: 6 }}
                onClick={() => navigate('/llm/settings/service-config/observability')}
                selected={isActive('/llm/settings/service-config/observability')}
              >
                <ListItemText primary="Service Metrics" />
              </ListItemButton>
              <ListItemButton 
                sx={{ pl: 6 }}
                onClick={() => navigate('/llm/settings/service-config/events')}
                selected={isActive('/llm/settings/service-config/events')}
              >
                <ListItemText primary="Events" />
              </ListItemButton>
              <ListItemButton 
                sx={{ pl: 6 }}
                onClick={() => navigate('/llm/settings/service-config/progress-tracking')}
                selected={isActive('/llm/settings/service-config/progress-tracking')}
              >
                <ListItemText primary="Progress Tracking" />
              </ListItemButton>
              <ListItemButton 
                sx={{ pl: 6 }}
                onClick={() => navigate('/llm/settings/service-config/configurations')}
                selected={isActive('/llm/settings/service-config/configurations')}
              >
                <ListItemText primary="Saved Configurations" />
              </ListItemButton>
            </List>
          </Collapse>
        </List>
      </Collapse>

      {/* No additional logout button needed as it already exists in the UI */}
    </List>
  );
};

export default LLMSideMenu;
