import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Grid, Alert, Box } from '@mui/material';

// Import PageLayout component
import PageLayout from '../components/Layout/PageLayout';

// Import Dashboard components
import {
  StatCard,
  FeaturedResearch,
  ResearchTools,
  WelcomeHeader
} from '../components/Dashboard';

/**
 * Dashboard page component
 * Displays overview of the system and quick access to main features
 */
const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [userCount, setUserCount] = useState(0);
  const [activeSessions, setActiveSessions] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserData = async () => {
      const token = localStorage.getItem('token');
      if (!token) {
        navigate('/');
        return;
      }

      try {
        const response = await axios.get('http://localhost:8000/api/me', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        setUser(response.data);
        
        // If user is admin, fetch additional stats
        if (response.data.role_id === 2) {
          try {
            const statsResponse = await axios.get('http://localhost:8000/api/stats', {
              headers: {
                'Authorization': `Bearer ${token}`
              }
            });
            setUserCount(statsResponse.data.user_count || 0);
            setActiveSessions(statsResponse.data.active_sessions || 0);
          } catch (statsErr) {
            console.error('Failed to fetch stats:', statsErr);
          }
        }
      } catch (err) {
        console.error('Failed to fetch user data:', err);
        setError('Failed to load user data. You may need to log in again.');
        if (err.response && (err.response.status === 401 || err.response.status === 403)) {
          handleLogout();
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUserData();
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  // Recent medical research updates
  const recentUpdates = [
    {
      title: "Procalcitonin-guided antibiotic therapy in CAP shows promising results",
      date: "Apr 2025"
    },
    {
      title: "New data on antibiotic resistance patterns in Streptococcus pneumoniae",
      date: "Mar 2025"
    },
    {
      title: "Post-COVID patterns in respiratory infections suggest modified treatment approaches",
      date: "Feb 2025"
    }
  ];

  // Define actions for the page header if needed
  const pageActions = null;

  return (
    <PageLayout
      title="Dashboard"
      breadcrumbs={[]}
      loading={loading}
      user={user}
      actions={pageActions}
    >
      {/* Display error message if any */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Welcome header */}
        <Grid item xs={12}>
          <WelcomeHeader user={user} />
        </Grid>

        {/* Quick stats cards */}
        <Grid item xs={12} md={4}>
          <StatCard 
            title="Total Users" 
            value={userCount} 
            actionText={user?.role_id === 2 ? "Manage Users" : null}
            onAction={() => navigate('/users')}
          />
        </Grid>
        
        <Grid item xs={12} md={4}>
          <StatCard 
            title="Active Sessions" 
            value={activeSessions} 
          />
        </Grid>
        
        <Grid item xs={12} md={4}>
          <StatCard 
            title="System Status" 
            value="Operational"
            icon={
              <Box sx={{ 
                width: 12, 
                height: 12, 
                borderRadius: '50%', 
                bgcolor: 'success.main'
              }} />
            }
          />
        </Grid>

        {/* Featured research */}
        <Grid item xs={12}>
          <FeaturedResearch updates={recentUpdates} />
        </Grid>

        {/* Research tools */}
        <Grid item xs={12}>
          <ResearchTools />
        </Grid>
      </Grid>
    </PageLayout>
  );
};

export default Dashboard;
