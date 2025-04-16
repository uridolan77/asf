import React, { useMemo, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Alert,
  Box,
  Typography,
  Card,
  CardContent
} from '@mui/material';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

// Import PageLayout component
import PageLayout from '../components/Layout/PageLayout.js';

// Import Dashboard components
import {
  StatCard,
  FeaturedResearch,
  ResearchTools,
  WelcomeHeader
} from '../components/Dashboard';

// Import custom hooks
import { useAuth } from '../context/AuthContext.jsx';

// Import skeleton loader
import { DashboardSkeleton } from '../components/UI/SkeletonLoaders.js';

// Chart colors
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

/**
 * Dashboard page component
 * Displays overview of the system and quick access to main features
 */
const Dashboard = () => {
  const { user, hasRole, api } = useAuth();
  const navigate = useNavigate();
  const [statsData, setStatsData] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [updatesData, setUpdatesData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load dashboard data
  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      try {
        // Fetch stats data
        const statsResponse = await api.get('/api/stats');
        setStatsData(statsResponse.data);
        
        // Fetch research metrics
        const metricsResponse = await api.get('/api/research-metrics');
        setMetricsData(metricsResponse.data);
        
        // Fetch recent updates
        const updatesResponse = await api.get('/api/recent-updates');
        setUpdatesData(updatesResponse.data);
        
        setError(null);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [api]);

  // Memoize the aggregated research metrics
  const aggregatedMetrics = useMemo(() => {
    if (!metricsData) return null;

    // Group data by category
    const categories = {};
    metricsData.forEach(item => {
      if (!categories[item.category]) {
        categories[item.category] = 0;
      }
      categories[item.category] += item.count;
    });

    // Convert to array for chart display
    return Object.keys(categories).map(category => ({
      name: category,
      value: categories[category]
    }));
  }, [metricsData]);

  // Memoize the monthly trends data
  const monthlyTrends = useMemo(() => {
    if (!statsData?.monthly_data) return [];
    return statsData.monthly_data;
  }, [statsData]);

  // Recent medical research updates (fallback data)
  const fallbackUpdates = [
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

  // Show skeleton loader while loading
  if (loading) {
    return (
      <PageLayout
        title="Dashboard"
        breadcrumbs={[]}
        loading={loading}
      >
        <DashboardSkeleton />
      </PageLayout>
    );
  }

  return (
    <PageLayout
      title="Dashboard"
      breadcrumbs={[]}
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
            value={statsData?.user_count || 0}
            actionText={hasRole('admin') ? "Manage Users" : null}
            onAction={() => navigate('/users')}
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <StatCard
            title="Active Sessions"
            value={statsData?.active_sessions || 0}
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <StatCard
            title="System Status"
            value={statsData?.system_status || "Operational"}
            icon={
              <Box sx={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                bgcolor: statsData?.system_status === 'Operational' ? 'success.main' : 'error.main'
              }} />
            }
          />
        </Grid>

        {/* Monthly trends chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Monthly Activity
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={monthlyTrends}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="searches" stroke="#8884d8" activeDot={{ r: 8 }} />
                    <Line type="monotone" dataKey="analyses" stroke="#82ca9d" />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Research distribution pie chart */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Research Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={aggregatedMetrics || []}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {(aggregatedMetrics || []).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value} studies`, null]} />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Featured research */}
        <Grid item xs={12}>
          <FeaturedResearch updates={updatesData?.items || fallbackUpdates} />
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
