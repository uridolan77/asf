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
import LLMDashboard from './pages/llm/LLMDashboard';
import CLPEFTDashboard from './components/llm/cl_peft/CLPEFTDashboard';

// Search pages
import PICOSearch from './pages/search/PICOSearch';
import KnowledgeBase from './pages/search/KnowledgeBase';

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

      // LLM routes
      { path: 'llm/dashboard', element: <LLMDashboard /> },
      { path: 'llm/cl-peft', element: <CLPEFTDashboard /> },

      // Search routes
      { path: 'search/pico', element: <PICOSearch /> },
      { path: 'search/knowledge-base', element: <KnowledgeBase /> },

      // Document processing route
      { path: 'document-processing', element: <DocumentProcessing /> },

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
