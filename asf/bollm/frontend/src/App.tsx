// frontend/src/App.tsx
import React, { useState, useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline, CircularProgress } from '@mui/material';
import { NotificationProvider } from './context/NotificationContext.jsx';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import { ReportProvider } from './context/ReportContext';
import { TopProgressBar } from './components/UI/LoadingIndicators.jsx';
import { PageTransition } from './components/UI/Animations.jsx';

// Theme
import theme from './theme';

// Lazy load page components
const Login = lazy(() => import('./pages/Login.jsx'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Register = lazy(() => import('./pages/Register'));
const Users = lazy(() => import('./pages/Users'));
const Settings = lazy(() => import('./pages/Settings'));
const PICOSearch = lazy(() => import('./pages/PICOSearch'));
const KnowledgeBase = lazy(() => import('./pages/KnowledgeBase'));
const ClinicalData = lazy(() => import('./pages/ClinicalData'));
const Analysis = lazy(() => import('./pages/Analysis'));
const MLServices = lazy(() => import('./pages/MLServices'));
const ClientsManagement = lazy(() => import('./pages/ClientsManagement'));
const NCBIClientPage = lazy(() => import('./pages/NCBIClientPage'));
const LLMManagement = lazy(() => import('./pages/LLMManagement'));
const DSPyDashboard = lazy(() => import('./pages/DSPyDashboard'));
const TextPlayground = lazy(() => import('./pages/LLMManagement/TextPlayground'));

// Reporting pages
const ReportBuilder = lazy(() => import('./pages/reporting/ReportBuilder'));
const SavedReports = lazy(() => import('./pages/reporting/SavedReports'));
const ReportViewer = lazy(() => import('./pages/reporting/ReportViewer'));
const ScheduledReports = lazy(() => import('./pages/reporting/ScheduledReports'));

// LLM Models pages
const ModelsListPage = lazy(() => import('./pages/LLMManagement/Models/ModelsListPage'));
const ModelParametersPage = lazy(() => import('./pages/LLMManagement/Models/ModelParametersPage'));
const ModelConfigPage = lazy(() => import('./pages/LLMManagement/Models/ModelConfigPage'));
const ModelTestPage = lazy(() => import('./pages/LLMManagement/Models/ModelTestPage'));

// Loading fallback component
const LoadingFallback: React.FC = () => (
  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
    <CircularProgress />
  </div>
);

interface ProtectedRouteProps {
  children: React.ReactNode;
}

// Protected route component
const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <LoadingFallback />;
  }

  return isAuthenticated ? <>{children}</> : <Navigate to="/" />;
};

// Animated route change component
const AnimatedRoutes: React.FC = () => {
  const location = useLocation();
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Simulate loading on route change
  useEffect(() => {
    setIsLoading(true);
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 500);

    return () => clearTimeout(timer);
  }, [location]);

  return (
    <>
      <TopProgressBar loading={isLoading} />
      <PageTransition key={location.pathname}>
        <Suspense fallback={<LoadingFallback />}>
          <Routes>
            <Route path="/" element={<Login />} />
            <Route path="/register" element={<Register />} />

            {/* Protected Routes */}
            <Route path="/dashboard" element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } />
            <Route path="/users" element={
              <ProtectedRoute>
                <Users />
              </ProtectedRoute>
            } />
            <Route path="/settings" element={
              <ProtectedRoute>
                <Settings />
              </ProtectedRoute>
            } />
            <Route path="/pico-search" element={
              <ProtectedRoute>
                <PICOSearch />
              </ProtectedRoute>
            } />
            <Route path="/knowledge-base" element={
              <ProtectedRoute>
                <KnowledgeBase />
              </ProtectedRoute>
            } />
            <Route path="/clinical-data" element={
              <ProtectedRoute>
                <ClinicalData />
              </ProtectedRoute>
            } />
            <Route path="/analysis" element={
              <ProtectedRoute>
                <Analysis />
              </ProtectedRoute>
            } />
            <Route path="/ml-services" element={
              <ProtectedRoute>
                <MLServices />
              </ProtectedRoute>
            } />
            <Route path="/clients-management" element={
              <ProtectedRoute>
                <ClientsManagement />
              </ProtectedRoute>
            } />
            <Route path="/clients-management/ncbi" element={
              <ProtectedRoute>
                <NCBIClientPage />
              </ProtectedRoute>
            } />
            <Route path="/llm-management" element={
              <ProtectedRoute>
                <LLMManagement />
              </ProtectedRoute>
            } />
            <Route path="/dspy-dashboard" element={
              <ProtectedRoute>
                <DSPyDashboard />
              </ProtectedRoute>
            } />
            <Route path="/llm/playground/text" element={
              <ProtectedRoute>
                <TextPlayground />
              </ProtectedRoute>
            } />

            {/* LLM Models Routes */}
            <Route path="/llm/models" element={
              <ProtectedRoute>
                <ModelsListPage />
              </ProtectedRoute>
            } />
            <Route path="/llm/models/parameters" element={
              <ProtectedRoute>
                <ModelParametersPage />
              </ProtectedRoute>
            } />
            <Route path="/llm/models/config" element={
              <ProtectedRoute>
                <ModelConfigPage />
              </ProtectedRoute>
            } />
            <Route path="/llm/models/test" element={
              <ProtectedRoute>
                <ModelTestPage />
              </ProtectedRoute>
            } />

            {/* Reporting Routes */}
            <Route path="/reports/new" element={
              <ProtectedRoute>
                <ReportBuilder />
              </ProtectedRoute>
            } />
            <Route path="/reports/edit/:id" element={
              <ProtectedRoute>
                <ReportBuilder />
              </ProtectedRoute>
            } />
            <Route path="/reports/saved" element={
              <ProtectedRoute>
                <SavedReports />
              </ProtectedRoute>
            } />
            <Route path="/reports/view/:id" element={
              <ProtectedRoute>
                <ReportViewer />
              </ProtectedRoute>
            } />
            <Route path="/reports/scheduled" element={
              <ProtectedRoute>
                <ScheduledReports />
              </ProtectedRoute>
            } />

            {/* Fallback for unknown routes */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Suspense>
      </PageTransition>
    </>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <NotificationProvider>
        <Router>
          <AuthProvider>
            <ReportProvider>
              <AnimatedRoutes />
            </ReportProvider>
          </AuthProvider>
        </Router>
      </NotificationProvider>
    </ThemeProvider>
  );
}

export default App;