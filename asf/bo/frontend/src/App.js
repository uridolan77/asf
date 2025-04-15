// frontend/src/App.js
import React, { useState, useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline, CircularProgress } from '@mui/material';
import { NotificationProvider } from './context/NotificationContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import { TopProgressBar } from './components/UI/LoadingIndicators';
import { PageTransition } from './components/UI/Animations';

// Theme
import theme from './theme';

// Lazy load page components
const Login = lazy(() => import('./pages/Login'));
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

// Loading fallback component
const LoadingFallback = () => (
  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
    <CircularProgress />
  </div>
);

// Protected route component
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <LoadingFallback />;
  }

  return isAuthenticated ? children : <Navigate to="/" />;
};

// Animated route change component
const AnimatedRoutes = () => {
  const location = useLocation();
  const [isLoading, setIsLoading] = useState(false);

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

            {/* Fallback for unknown routes */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
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
        <Router>
          <AuthProvider>
            <AnimatedRoutes />
          </AuthProvider>
        </Router>
      </NotificationProvider>
    </ThemeProvider>
  );
}

export default App;