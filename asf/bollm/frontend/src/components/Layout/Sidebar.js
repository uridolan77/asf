import React from 'react';
import { useAppConfigStore } from '../../store/appConfigStore';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Collapse,
  useTheme
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Search as SearchIcon,
  MedicalServices as MedicalServicesIcon,
  Psychology as PsychologyIcon,
  ExpandLess,
  ExpandMore,
  Science as ScienceIcon,
  Storage as StorageIcon,
  Biotech as BiotechIcon,
  Memory as MemoryIcon,
  Tune as TuneIcon,
  Settings as SettingsIcon,
  Help as HelpIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import LLMSideMenu from '../Navigation/LLMSideMenu.jsx';

const DRAWER_WIDTH = 280;

const SidebarRoot = styled('div')(({ theme }) => ({
  [theme.breakpoints.up('md')]: {
    flexShrink: 0,
    width: DRAWER_WIDTH
  }
}));

const SidebarItem = ({ title, icon, href, children, open, onClick }) => {
  const location = useLocation();
  const theme = useTheme();
  const active = href ? location.pathname === href : false;

  if (children) {
    return (
      <>
        <ListItem disablePadding>
          <ListItemButton onClick={onClick} sx={{ pl: 2 }}>
            <ListItemIcon sx={{ color: open ? theme.palette.primary.main : 'inherit' }}>
              {icon}
            </ListItemIcon>
            <ListItemText
              primary={title}
              primaryTypographyProps={{
                fontWeight: open ? 600 : 400,
                color: open ? theme.palette.primary.main : 'inherit'
              }}
            />
            {open ? <ExpandLess /> : <ExpandMore />}
          </ListItemButton>
        </ListItem>
        <Collapse in={open} timeout="auto" unmountOnExit>
          <List component="div" disablePadding>
            {children}
          </List>
        </Collapse>
      </>
    );
  }

  return (
    <ListItem disablePadding>
      <ListItemButton
        component={RouterLink}
        to={href}
        selected={active}
        sx={{
          pl: 2,
          '&.Mui-selected': {
            backgroundColor: theme.palette.action.selected,
            borderLeft: `4px solid ${theme.palette.primary.main}`,
            paddingLeft: '12px',
            '&:hover': {
              backgroundColor: theme.palette.action.hover,
            }
          }
        }}
      >
        <ListItemIcon sx={{ color: active ? theme.palette.primary.main : 'inherit' }}>
          {icon}
        </ListItemIcon>
        <ListItemText
          primary={title}
          primaryTypographyProps={{
            fontWeight: active ? 600 : 400,
            color: active ? theme.palette.primary.main : 'inherit'
          }}
        />
      </ListItemButton>
    </ListItem>
  );
};

const NestedSidebarItem = ({ title, href }) => {
  const location = useLocation();
  const theme = useTheme();
  const active = location.pathname === href;

  return (
    <ListItem disablePadding>
      <ListItemButton
        component={RouterLink}
        to={href}
        selected={active}
        sx={{
          pl: 4,
          '&.Mui-selected': {
            backgroundColor: theme.palette.action.selected,
            borderLeft: `4px solid ${theme.palette.primary.main}`,
            paddingLeft: '28px',
            '&:hover': {
              backgroundColor: theme.palette.action.hover,
            }
          }
        }}
      >
        <ListItemText
          primary={title}
          primaryTypographyProps={{
            fontWeight: active ? 600 : 400,
            color: active ? theme.palette.primary.main : 'inherit',
            fontSize: '0.875rem'
          }}
        />
      </ListItemButton>
    </ListItem>
  );
};

const Sidebar = ({ open, onClose }) => {
  const theme = useTheme();
  const { featureFlags } = useAppConfigStore();
  const reportingEnabled = featureFlags.find(f => f.id === 'reporting-feature')?.enabled ?? true;

  const [medicalOpen, setMedicalOpen] = React.useState(false);
  const [llmOpen, setLlmOpen] = React.useState(false);
  const [searchOpen, setSearchOpen] = React.useState(false);
  const [reportingOpen, setReportingOpen] = React.useState(true);

  const handleMedicalClick = () => {
    setMedicalOpen(!medicalOpen);
  };

  const handleLlmClick = () => {
    setLlmOpen(!llmOpen);
  };

  const handleSearchClick = () => {
    setSearchOpen(!searchOpen);
  };

  const handleReportingClick = () => {
    setReportingOpen(!reportingOpen);
  };

  const content = (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%'
      }}
    >
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <Typography variant="h5" color="primary" fontWeight={600}>
          ASF LLM Manager
        </Typography>
      </Box>
      <Divider />
      <Box sx={{ flexGrow: 1, overflow: 'auto', py: 2 }}>
        <List component="nav" disablePadding>
          <SidebarItem
            title="Dashboard"
            icon={<DashboardIcon />}
            href="/dashboard"
          />

          <Box sx={{ mt: 2 }}>
            <Typography
              variant="overline"
              color="textSecondary"
              sx={{ pl: 2, display: 'block', mb: 1 }}
            >
              Medical
            </Typography>
          </Box>

          <SidebarItem
            title="Medical Clients"
            icon={<MedicalServicesIcon />}
            open={medicalOpen}
            onClick={handleMedicalClick}
          >
            <NestedSidebarItem title="NCBI" href="/medical/clients/ncbi" />
            <NestedSidebarItem title="UMLS" href="/medical/clients/umls" />
            <NestedSidebarItem title="ClinicalTrials" href="/medical/clients/clinical-trials" />
            <NestedSidebarItem title="Cochrane" href="/medical/clients/cochrane" />
            <NestedSidebarItem title="CrossRef" href="/medical/clients/crossref" />
            <NestedSidebarItem title="SNOMED" href="/medical/clients/snomed" />
          </SidebarItem>

          <Box sx={{ mt: 2 }}>
            <Typography
              variant="overline"
              color="textSecondary"
              sx={{ pl: 2, display: 'block', mb: 1 }}
            >
              LLM
            </Typography>
          </Box>

          <LLMSideMenu />

          <Box sx={{ mt: 2 }}>
            <Typography
              variant="overline"
              color="textSecondary"
              sx={{ pl: 2, display: 'block', mb: 1 }}
            >
              Search
            </Typography>
          </Box>

          <SidebarItem
            title="Search Tools"
            icon={<SearchIcon />}
            open={searchOpen}
            onClick={handleSearchClick}
          >
            <NestedSidebarItem title="PICO Search" href="/search/pico" />
            <NestedSidebarItem title="Knowledge Base" href="/search/knowledge-base" />
          </SidebarItem>

          {reportingEnabled && (
            <>
              <Box sx={{ mt: 2 }}>
                <Typography
                  variant="overline"
                  color="textSecondary"
                  sx={{ pl: 2, display: 'block', mb: 1 }}
                >
                  Reporting
                </Typography>
              </Box>

              <SidebarItem
                title="Reporting"
                icon={<AssessmentIcon />}
                open={reportingOpen}
                onClick={handleReportingClick}
              >
                <NestedSidebarItem title="Report Builder" href="/reports/new" />
                <NestedSidebarItem title="Saved Reports" href="/reports/saved" />
                <NestedSidebarItem title="Scheduled Reports" href="/reports/scheduled" />
              </SidebarItem>
            </>
          )}

          <Box sx={{ mt: 2 }}>
            <Typography
              variant="overline"
              color="textSecondary"
              sx={{ pl: 2, display: 'block', mb: 1 }}
            >
              System
            </Typography>
          </Box>

          <SidebarItem
            title="Settings"
            icon={<SettingsIcon />}
            href="/settings"
          />

          <SidebarItem
            title="Help"
            icon={<HelpIcon />}
            href="/help"
          />
        </List>
      </Box>
      <Divider />
      <Box sx={{ p: 2 }}>
        <Typography variant="body2" color="textSecondary" align="center">
          BOLLM v1.0.0
        </Typography>
      </Box>
    </Box>
  );

  return (
    <SidebarRoot>
      <Box
        sx={{
          display: { xs: 'block', md: 'none' }
        }}
      >
        <Drawer
          anchor="left"
          onClose={onClose}
          open={open}
          variant="temporary"
          PaperProps={{
            sx: {
              width: DRAWER_WIDTH
            }
          }}
        >
          {content}
        </Drawer>
      </Box>
      <Box
        sx={{
          display: { xs: 'none', md: 'block' }
        }}
      >
        <Drawer
          anchor="left"
          open
          variant="persistent"
          PaperProps={{
            sx: {
              width: DRAWER_WIDTH,
              border: 'none',
              boxShadow: theme.shadows[3]
            }
          }}
        >
          {content}
        </Drawer>
      </Box>
    </SidebarRoot>
  );
};

export default Sidebar;
