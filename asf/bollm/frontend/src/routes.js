import React from 'react';
import { Navigate } from 'react-router-dom';

// Layout components
import { Outlet } from 'react-router-dom';
import PageLayout from './components/Layout/PageLayout';
import AuthLayout from './components/Layout/AuthLayout';

// Use PageLayout as DashboardLayout since it doesn't exist
const DashboardLayout = () => (
  <PageLayout>
    <Outlet />
  </PageLayout>
);

// Pages
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import NotFound from './pages/NotFound';
import DocumentProcessing from './pages/DocumentProcessing';

// Medical client pages
import NCBIPage from './pages/medical/clients/NCBIPage';
import UMLSPage from './pages/medical/clients/UMLSPage';
import ClinicalTrialsPage from './pages/medical/clients/ClinicalTrialsPage';
import CochranePage from './pages/medical/clients/CochranePage';
import CrossRefPage from './pages/medical/clients/CrossRefPage';
import SNOMEDPage from './pages/medical/clients/SNOMEDPage';

// LLM pages
import LLMDashboard from './pages/LLMManagement/Dashboard';
import GatewayDashboard from './pages/LLMManagement/GatewayDashboard';
import UsageDashboard from './pages/LLMManagement/UsageDashboard';
import CLPEFTDashboard from './pages/LLMManagement/CLPEFTDashboard';
import MCPDashboard from './pages/LLMManagement/MCPDashboard';
import DSPyDashboard from './pages/LLMManagement/DSPyDashboard';

// Search pages
import PICOSearch from './pages/search/PICOSearch';
import KnowledgeBase from './pages/search/KnowledgeBase';

// Reporting pages
import ReportBuilder from './pages/reporting/ReportBuilder';
import SavedReports from './pages/reporting/SavedReports';
import ReportViewer from './pages/reporting/ReportViewer';
import ScheduledReports from './pages/reporting/ScheduledReports';

// Settings page
import Settings from './pages/Settings';

// Placeholder component for pages not yet implemented
const PlaceholderPage = ({ title }) => (
  <div style={{ padding: '2rem' }}>
    <h2>{title || 'Page Under Construction'}</h2>
    <p>This feature is currently under development and will be available soon.</p>
  </div>
);

const routes = [
  {
    path: '/',
    element: <DashboardLayout />,
    children: [
      { path: '', element: <Navigate to="/dashboard" /> },
      { path: 'dashboard', element: <Dashboard /> },

      // Medical client routes
      { path: 'medical/clients/ncbi', element: <NCBIPage /> },
      { path: 'medical/clients/umls', element: <UMLSPage /> },
      { path: 'medical/clients/clinical-trials', element: <ClinicalTrialsPage /> },
      { path: 'medical/clients/cochrane', element: <CochranePage /> },
      { path: 'medical/clients/crossref', element: <CrossRefPage /> },
      { path: 'medical/clients/snomed', element: <SNOMEDPage /> },

      // LLM Dashboard
      { path: 'llm/dashboard', element: <LLMDashboard /> },

      // LLM Provider Management
      { path: 'llm/providers', element: <PlaceholderPage title="LLM Providers List" /> },
      { path: 'llm/providers/config', element: <PlaceholderPage title="Provider Configuration" /> },
      { path: 'llm/providers/keys', element: <PlaceholderPage title="Provider API Keys" /> },
      { path: 'llm/providers/health', element: <PlaceholderPage title="Provider Health Status" /> },

      // LLM Model Management
      { path: 'llm/models', element: <PlaceholderPage title="LLM Models List" /> },
      { path: 'llm/models/performance', element: <PlaceholderPage title="Model Performance" /> },
      { path: 'llm/models/usage', element: <UsageDashboard /> },

      // LLM Request Management
      { path: 'llm/requests/logs', element: <PlaceholderPage title="Request Logs" /> },
      { path: 'llm/requests/analysis', element: <PlaceholderPage title="Request Analysis" /> },

      // LLM Playground
      { path: 'llm/playground/text', element: <GatewayDashboard /> },
      { path: 'llm/playground/chat', element: <PlaceholderPage title="Chat Completion Playground" /> },

      // LLM Cache Management
      { path: 'llm/cache/stats', element: <PlaceholderPage title="Cache Statistics" /> },
      { path: 'llm/cache/config', element: <PlaceholderPage title="Cache Configuration" /> },
      { path: 'llm/cache/semantic', element: <PlaceholderPage title="Semantic Cache" /> },

      // LLM Observability
      { path: 'llm/observability/metrics', element: <PlaceholderPage title="LLM Metrics" /> },
      { path: 'llm/observability/traces', element: <PlaceholderPage title="Request Traces" /> },
      { path: 'llm/observability/alerts', element: <PlaceholderPage title="Alerts" /> },

      // LLM Advanced Features
      { path: 'llm/advanced/cl-peft', element: <CLPEFTDashboard /> },
      { path: 'llm/advanced/dspy', element: <DSPyDashboard /> },
      { path: 'llm/advanced/plugins', element: <PlaceholderPage title="LLM Plugins" /> },
      { path: 'llm/advanced/interventions', element: <PlaceholderPage title="LLM Interventions" /> },

      // LLM Settings
      { path: 'llm/settings/gateway', element: <PlaceholderPage title="Gateway Configuration" /> },
      { path: 'llm/settings/api-keys', element: <PlaceholderPage title="API Keys Management" /> },
      { path: 'llm/settings/environment', element: <PlaceholderPage title="Environment Variables" /> },

      // Legacy LLM routes
      { path: 'llm/cl-peft', element: <Navigate to="/llm/advanced/cl-peft" /> },
      { path: 'mcp-dashboard', element: <MCPDashboard /> },
      { path: 'mcp', element: <MCPDashboard /> },

      // Search routes
      { path: 'search/pico', element: <PICOSearch /> },
      { path: 'search/knowledge-base', element: <KnowledgeBase /> },

      // Document processing route
      { path: 'document-processing', element: <DocumentProcessing /> },

      // Reporting routes
      { path: 'reports/new', element: <ReportBuilder /> },
      { path: 'reports/edit/:id', element: <ReportBuilder /> },
      { path: 'reports/saved', element: <SavedReports /> },
      { path: 'reports/view/:id', element: <ReportViewer /> },
      { path: 'reports/scheduled', element: <ScheduledReports /> },

      // Settings route
      { path: 'settings', element: <Settings /> },

      { path: '404', element: <NotFound /> },
      { path: '*', element: <Navigate to="/404" /> }
    ]
  },
  {
    path: '/auth',
    element: <AuthLayout />,
    children: [
      { path: 'login', element: <Login /> },
      { path: '*', element: <Navigate to="/auth/login" /> }
    ]
  }
];

export default routes;
