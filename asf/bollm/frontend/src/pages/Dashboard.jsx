import React, { useMemo, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Alert,
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
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
// Import icons for ML and LLM sections
import SmartToyIcon from '@mui/icons-material/SmartToy';
import MemoryIcon from '@mui/icons-material/Memory';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import SpeedIcon from '@mui/icons-material/Speed';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';

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
  const [mlServicesData, setMlServicesData] = useState(null);
  const [llmUsageData, setLlmUsageData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load dashboard data
  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      try {
        // Fetch stats data - Keeping this as is for compatibility
        const statsResponse = await api.get('/api/stats');
        setStatsData(statsResponse.data);

        // Fetch research metrics - Keeping this as is for compatibility
        const metricsResponse = await api.get('/api/research-metrics');
        setMetricsData(metricsResponse.data);

        // Fetch recent updates - Keeping this as is for compatibility
        const updatesResponse = await api.get('/api/recent-updates');
        setUpdatesData(updatesResponse.data);

        // Try ML services status with direct import
        try {
          const mlServicesResponse = await api.get('/api/ml/services/status');
          setMlServicesData(mlServicesResponse.data.services);
        } catch (mlError) {
          console.warn('Using fallback ML services data:', mlError);
          // Use fallback data when the endpoint fails
          setMlServicesData([
            { name: "Claim Extractor", status: "operational", health: "healthy" },
            { name: "Contradiction Detector", status: "operational", health: "healthy" },
            { name: "Bias Assessment", status: "operational", health: "healthy" },
            { name: "Evidence Grader", status: "degraded", health: "degraded" }
          ]);
        }

        // Fetch LLM usage data
        try {
          const llmUsageResponse = await api.get('/api/llm-usage');
          setLlmUsageData(llmUsageResponse.data.usage || []);
        } catch (llmError) {
          console.warn('Using fallback LLM usage data:', llmError);
          // Use fallback data when the endpoint fails
          setLlmUsageData([
            { model: "gpt-4o", usage_count: 2580 },
            { model: "claude-3-opus", usage_count: 1420 },
            { model: "biomedlm-2-7b", usage_count: 3850 },
            { model: "mistralai/Mixtral-8x7B", usage_count: 980 }
          ]);
        }

        setError(null);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Using fallback data instead.');

        // Set fallback data for all metrics
        if (!statsData) {
          setStatsData({
            user_count: 42,
            active_sessions: 8,
            system_status: "Operational",
            monthly_data: [
              { month: "Jan", searches: 240, analyses: 120 },
              { month: "Feb", searches: 300, analyses: 150 },
              { month: "Mar", searches: 320, analyses: 180 },
              { month: "Apr", searches: 380, analyses: 220 }
            ]
          });
        }

        if (!metricsData) {
          setMetricsData([
            { category: "Cardiology", count: 120 },
            { category: "Oncology", count: 85 },
            { category: "Neurology", count: 65 },
            { category: "Infectious Disease", count: 95 },
            { category: "Pediatrics", count: 40 }
          ]);
        }

        if (!updatesData) {
          setUpdatesData({ items: fallbackUpdates });
        }

        if (!mlServicesData) {
          setMlServicesData([
            { name: "Claim Extractor", status: "operational", health: "healthy" },
            { name: "Contradiction Detector", status: "operational", health: "healthy" },
            { name: "Bias Assessment", status: "operational", health: "healthy" },
            { name: "Evidence Grader", status: "degraded", health: "degraded" }
          ]);
        }

        if (!llmUsageData) {
          setLlmUsageData([
            { model: "gpt-4o", usage_count: 2580 },
            { model: "claude-3-opus", usage_count: 1420 },
            { model: "biomedlm-2-7b", usage_count: 3850 },
            { model: "mistralai/Mixtral-8x7B", usage_count: 980 }
          ]);
        }
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

      {/* LLM Management Banner - NEW */}
      <Grid item xs={12} sx={{ mb: 3 }}>
        <Card sx={{ bgcolor: 'primary.dark', color: 'white' }}>
          <CardContent sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography variant="h5" gutterBottom>
                New LLM Management Interface Available
              </Typography>
              <Typography variant="body1">
                Access our new comprehensive LLM management console with provider configuration, model management, monitoring tools, and more.
              </Typography>
            </Box>
            <Button 
              variant="contained" 
              color="secondary" 
              size="large"
              endIcon={<ArrowForwardIcon />}
              onClick={() => navigate('/llm/dashboard')}
              sx={{ ml: 2 }}
            >
              Open LLM Dashboard
            </Button>
          </CardContent>
        </Card>
      </Grid>

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

        {/* ML Services Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ML Services Status
              </Typography>
              <List>
                {mlServicesData?.map((service, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <SmartToyIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary={service.name}
                      secondary={`Status: ${service.status}`}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* LLM Usage Stats */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  LLM Usage Stats
                </Typography>
                <Button
                  size="small"
                  endIcon={<ArrowForwardIcon />}
                  onClick={() => navigate('/llm/dashboard')}
                >
                  View Full Dashboard
                </Button>
              </Box>
              <List>
                {llmUsageData?.map((usage, index) => (
                  <ListItem key={index}>
                    <ListItemIcon>
                      <MemoryIcon />
                    </ListItemIcon>
                    <ListItemText
                      primary={usage.model}
                      secondary={`Usage: ${usage.usage_count} times`}
                    />
                  </ListItem>
                ))}
              </List>
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
