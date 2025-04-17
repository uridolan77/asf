// frontend/src/App.jsx
import React, { useState, useEffect, Suspense, lazy, useRef } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline, CircularProgress } from '@mui/material';
import { NotificationProvider } from './context/NotificationContext.jsx';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import { TopProgressBar } from './components/UI/LoadingIndicators.js';
import { PageTransition } from './components/UI/Animations.js';

// Theme
import theme from './theme';

// Lazy load page components
const Login = lazy(() => import('./pages/Login.jsx'));
const Dashboard = lazy(() => import('./pages/Dashboard.jsx'));
const Register = lazy(() => import('./pages/Register'));
const Users = lazy(() => import('./pages/Users'));
const Settings = lazy(() => import('./pages/Settings'));
const SearchPage = lazy(() => import('./pages/SearchPage'));
const KnowledgeBasePage = lazy(() => import('./pages/KnowledgeBasePage'));
const ClinicalData = lazy(() => import('./pages/ClinicalData'));
const Analysis = lazy(() => import('./pages/Analysis'));
const MLServices = lazy(() => import('./pages/MLServices'));
const ClientsManagementPage = lazy(() => import('./pages/ClientsManagementPage'));
const NCBIClientPage = lazy(() => import('./pages/NCBIClientPage'));
const LLMManagement = lazy(() => import('./pages/LLMManagement'));
const DocumentProcessing = lazy(() => import('./pages/DocumentProcessing'));

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
            <Route path="/search" element={
              <ProtectedRoute>
                <SearchPage />
              </ProtectedRoute>
            } />
            <Route path="/pico-search" element={
              <ProtectedRoute>
                <SearchPage />
              </ProtectedRoute>
            } />
            <Route path="/knowledge-base" element={
              <ProtectedRoute>
                <KnowledgeBasePage />
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
                <ClientsManagementPage />
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
            <Route path="/document-processing" element={
              <ProtectedRoute>
                <DocumentProcessing />
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
