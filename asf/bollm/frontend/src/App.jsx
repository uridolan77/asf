2// frontend/src/App.jsx
import React, { useState, useEffect, Suspense, lazy, useRef } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline, CircularProgress } from '@mui/material';
import { NotificationProvider } from './context/NotificationContext.jsx';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import QueryProvider from './context/QueryProvider.jsx';
import { TopProgressBar } from './components/UI/LoadingIndicators.js';
import { PageTransition } from './components/UI/Animations.js';

// Theme
import theme from './theme';

// Lazy load page components
const Login = lazy(() => import('./pages/Login.jsx'));
const Register = lazy(() => import('./pages/Register'));

// LLM Management pages - Only import those that actually exist
const LLMDashboard = lazy(() => import('./pages/LLMManagement/Dashboard'));
const GatewayDashboard = lazy(() => import('./pages/LLMManagement/GatewayDashboard'));
const UsageDashboard = lazy(() => import('./pages/LLMManagement/UsageDashboard'));
const CLPEFTDashboard = lazy(() => import('./pages/LLMManagement/CLPEFTDashboard'));
const DSPyDashboard = lazy(() => import('./pages/LLMManagement/DSPyDashboard'));
const TextPlayground = lazy(() => import('./pages/LLMManagement/TextPlayground'));
const MCPDashboard = lazy(() => import('./pages/LLMManagement/MCPDashboard'));
const PlaceholderPage = lazy(() => import('./pages/LLMManagement/PlaceholderPage'));
const Settings = lazy(() => import('./pages/Settings'));

// Import Provider Management pages
const ProvidersListPage = lazy(() => import('./pages/LLMManagement/Providers/ProvidersListPage'));
const ProviderConfigPage = lazy(() => import('./pages/LLMManagement/Providers/ProviderConfigPage'));
const ProviderKeysPage = lazy(() => import('./pages/LLMManagement/Providers/ProviderKeysPage'));
const ProviderHealthPage = lazy(() => import('./pages/LLMManagement/Providers/ProviderHealthPage'));

// Import Model Management pages
const ModelsListPage = lazy(() => import('./pages/LLMManagement/Models/ModelsListPage'));
const ModelParametersPage = lazy(() => import('./pages/LLMManagement/Models/ModelParametersPage'));
const ModelConfigPage = lazy(() => import('./pages/LLMManagement/Models/ModelConfigPage'));
const ModelTestPage = lazy(() => import('./pages/LLMManagement/Models/ModelTestPage'));

// Import Service Configuration pages
const ServiceConfigPage = lazy(() => import('./pages/LLMManagement/ServiceConfig/ServiceConfigPage'));
const ConfigurationsListPage = lazy(() => import('./pages/LLMManagement/ServiceConfig/ConfigurationsListPage'));
const ConfigurationEditPage = lazy(() => import('./pages/LLMManagement/ServiceConfig/ConfigurationEditPage'));

// Loading fallback component
const LoadingFallback = () => {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      flexDirection: 'column',
      backgroundColor: '#f5f5f5'
    }}>
      <CircularProgress size={40} />
      <div style={{ marginTop: 16 }}>Loading page...</div>
    </div>
  );
};

// Protected route component
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  const [showLoader, setShowLoader] = useState(true);

  // Use a delayed loader to prevent flickering
  useEffect(() => {
    // Only show loader if still loading after 300ms
    const timer = setTimeout(() => {
      setShowLoader(loading);
    }, 300);

    // If not loading anymore, immediately hide loader
    if (!loading) {
      setShowLoader(false);
    }

    return () => clearTimeout(timer);
  }, [loading]);

  // Keep showing children if already authenticated, even while revalidating
  if (isAuthenticated) {
    return children;
  }

  // Only show loading state if still loading after delay
  if (loading && showLoader) {
    return <LoadingFallback />;
  }

  // If not authenticated and not loading, redirect to login
  return <Navigate to="/login" replace />;
};

// Animated routes component with route transition effects
const AnimatedRoutes = () => {
  const location = useLocation();
  const progressBarRef = useRef(null);

  // Trigger progress bar on route change
  useEffect(() => {
    if (progressBarRef.current) {
      progressBarRef.current.start();

      const timer = setTimeout(() => {
        progressBarRef.current.complete();
      }, 500);

      return () => clearTimeout(timer);
    }
  }, [location]);

  return (
    <>
      <TopProgressBar ref={progressBarRef} />
      <PageTransition location={location}>
        <Suspense fallback={<LoadingFallback />}>
          <Routes>
            {/* Public Routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />

            {/* Redirect root to LLM dashboard */}
            <Route path="/" element={<Navigate to="/llm/dashboard" replace />} />
            
            {/* LLM Management Routes */}
            <Route path="/llm/dashboard" element={
              <ProtectedRoute>
                <LLMDashboard />
              </ProtectedRoute>
            } />
            
            {/* Provider Management Routes - Using proper components */}
            <Route path="/llm/providers" element={
              <ProtectedRoute>
                <ProvidersListPage />
              </ProtectedRoute>
            } />
            <Route path="/llm/providers/config" element={
              <ProtectedRoute>
                <ProviderConfigPage />
              </ProtectedRoute>
            } />
            <Route path="/llm/providers/keys" element={
              <ProtectedRoute>
                <ProviderKeysPage />
              </ProtectedRoute>
            } />
            <Route path="/llm/providers/health" element={
              <ProtectedRoute>
                <ProviderHealthPage />
              </ProtectedRoute>
            } />
            
            {/* Model Management Routes */}
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
            <Route path="/llm/models/performance" element={
              <ProtectedRoute>
                <PlaceholderPage title="Model Performance" />
              </ProtectedRoute>
            } />
            <Route path="/llm/models/usage" element={
              <ProtectedRoute>
                <UsageDashboard />
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
            <Route path="/llm/models/test/:providerId/:modelId" element={
              <ProtectedRoute>
                <ModelTestPage />
              </ProtectedRoute>
            } />
            
            {/* Request Management Routes */}
            <Route path="/llm/requests/logs" element={
              <ProtectedRoute>
                <PlaceholderPage title="Request Logs" />
              </ProtectedRoute>
            } />
            <Route path="/llm/requests/analysis" element={
              <ProtectedRoute>
                <PlaceholderPage title="Request Analysis" />
              </ProtectedRoute>
            } />
            
            {/* Playground Routes */}
            <Route path="/llm/playground/text" element={
              <ProtectedRoute>
                <TextPlayground />
              </ProtectedRoute>
            } />
            <Route path="/llm/playground/chat" element={
              <ProtectedRoute>
                <PlaceholderPage title="Chat Completion Playground" />
              </ProtectedRoute>
            } />
            
            {/* Cache Management Routes */}
            <Route path="/llm/cache/stats" element={
              <ProtectedRoute>
                <PlaceholderPage title="Cache Statistics" />
              </ProtectedRoute>
            } />
            <Route path="/llm/cache/config" element={
              <ProtectedRoute>
                <PlaceholderPage title="Cache Configuration" />
              </ProtectedRoute>
            } />
            <Route path="/llm/cache/semantic" element={
              <ProtectedRoute>
                <PlaceholderPage title="Semantic Cache" />
              </ProtectedRoute>
            } />
            
            {/* Observability Routes */}
            <Route path="/llm/observability/metrics" element={
              <ProtectedRoute>
                <PlaceholderPage title="LLM Metrics" />
              </ProtectedRoute>
            } />
            <Route path="/llm/observability/traces" element={
              <ProtectedRoute>
                <PlaceholderPage title="Request Traces" />
              </ProtectedRoute>
            } />
            <Route path="/llm/observability/alerts" element={
              <ProtectedRoute>
                <PlaceholderPage title="Alerts" />
              </ProtectedRoute>
            } />
            
            {/* Advanced Features Routes */}
            <Route path="/llm/advanced/cl-peft" element={
              <ProtectedRoute>
                <CLPEFTDashboard />
              </ProtectedRoute>
            } />
            <Route path="/llm/advanced/dspy" element={
              <ProtectedRoute>
                <DSPyDashboard />
              </ProtectedRoute>
            } />
            <Route path="/llm/advanced/plugins" element={
              <ProtectedRoute>
                <PlaceholderPage title="LLM Plugins" />
              </ProtectedRoute>
            } />
            <Route path="/llm/advanced/interventions" element={
              <ProtectedRoute>
                <PlaceholderPage title="LLM Interventions" />
              </ProtectedRoute>
            } />
            
            {/* Settings Routes */}
            <Route path="/llm/settings/gateway" element={
              <ProtectedRoute>
                <PlaceholderPage title="Gateway Configuration" />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/api-keys" element={
              <ProtectedRoute>
                <PlaceholderPage title="API Keys Management" />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/environment" element={
              <ProtectedRoute>
                <PlaceholderPage title="Environment Variables" />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/service-config" element={
              <ProtectedRoute>
                <ServiceConfigPage />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/service-config/caching" element={
              <ProtectedRoute>
                <ServiceConfigPage initialTab={1} />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/service-config/resilience" element={
              <ProtectedRoute>
                <ServiceConfigPage initialTab={2} />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/service-config/observability" element={
              <ProtectedRoute>
                <ServiceConfigPage initialTab={3} />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/service-config/events" element={
              <ProtectedRoute>
                <ServiceConfigPage initialTab={4} />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/service-config/progress-tracking" element={
              <ProtectedRoute>
                <ServiceConfigPage initialTab={5} />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/service-config/configurations" element={
              <ProtectedRoute>
                <ConfigurationsListPage />
              </ProtectedRoute>
            } />
            <Route path="/llm/settings/service-config/configurations/:configId" element={
              <ProtectedRoute>
                <ConfigurationEditPage />
              </ProtectedRoute>
            } />
            
            {/* Legacy Routes - Redirect to new structure */}
            <Route path="/dashboard" element={<Navigate to="/llm/dashboard" replace />} />
            <Route path="/llm-management" element={<Navigate to="/llm/dashboard" replace />} />
            <Route path="/mcp-dashboard" element={
              <ProtectedRoute>
                <MCPDashboard />
              </ProtectedRoute>
            } />
            <Route path="/settings" element={
              <ProtectedRoute>
                <Settings />
              </ProtectedRoute>
            } />

            {/* Fallback for unknown routes */}
            <Route path="*" element={<Navigate to="/llm/dashboard" replace />} />
          </Routes>
        </Suspense>
      </PageTransition>
    </>
  );
};

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <NotificationProvider>
        <QueryProvider>
          <Router>
            <AuthProvider>
              <AnimatedRoutes />
            </AuthProvider>
          </Router>
        </QueryProvider>
      </NotificationProvider>
    </ThemeProvider>
  );
}

export default App;
