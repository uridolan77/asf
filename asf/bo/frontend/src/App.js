// frontend/src/App.js
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { NotificationProvider } from './context/NotificationContext';
import { TopProgressBar } from './components/UI/LoadingIndicators';
import { PageTransition } from './components/UI/Animations';

// Theme
import theme from './theme';

// Pages
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Register from './pages/Register';
import Users from './pages/Users';
import Settings from './pages/Settings';
import PICOSearch from './pages/PICOSearch';
import KnowledgeBase from './pages/KnowledgeBase';
import ClinicalData from './pages/ClinicalData';
import Analysis from './pages/Analysis';
import MLServices from './pages/MLServices';
import ClientsManagement from './pages/ClientsManagement';

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
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/register" element={<Register />} />
          <Route path="/users" element={<Users />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/pico-search" element={<PICOSearch />} />
          <Route path="/knowledge-base" element={<KnowledgeBase />} />
          <Route path="/clinical-data" element={<ClinicalData />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/ml-services" element={<MLServices />} />
          <Route path="/clients-management" element={<ClientsManagement />} />
        </Routes>
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
          <AnimatedRoutes />
        </Router>
      </NotificationProvider>
    </ThemeProvider>
  );
}

export default App;