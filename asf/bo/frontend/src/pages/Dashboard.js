// frontend/src/pages/Dashboard.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

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
        // Consider auto-logout on auth errors
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

  if (loading) {
    return <div style={{ textAlign: 'center', marginTop: '50px' }}>Loading dashboard...</div>;
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar */}
      <div style={{ 
        width: '250px', 
        backgroundColor: '#2c3e50', 
        color: 'white', 
        padding: '20px' 
      }}>
        <h2 style={{ marginBottom: '30px' }}>BO Admin</h2>
        <div style={{ marginBottom: '20px' }}>
          <div style={{ fontWeight: 'bold' }}>Menu</div>
          <ul style={{ listStyleType: 'none', padding: 0 }}>
            <li style={{ padding: '10px 0', borderBottom: '1px solid #34495e' }}>Dashboard</li>
            {user && user.role_id === 2 && (
              <li 
                style={{ padding: '10px 0', borderBottom: '1px solid #34495e', cursor: 'pointer' }}
                onClick={() => navigate('/users')}
              >
                Users
              </li>
            )}
            <li 
              style={{ padding: '10px 0', borderBottom: '1px solid #34495e', cursor: 'pointer' }}
              onClick={() => navigate('/settings')}
            >
              Settings
            </li>
            <li 
              style={{ padding: '10px 0', borderBottom: '1px solid #34495e', cursor: 'pointer' }}
              onClick={() => navigate('/pico-search')}
            >
              PICO Search
            </li>
            <li 
              style={{ padding: '10px 0', borderBottom: '1px solid #34495e', cursor: 'pointer' }}
              onClick={() => navigate('/knowledge-base')}
            >
              Knowledge Base
            </li>
          </ul>
        </div>
        <button 
          onClick={handleLogout}
          style={{
            backgroundColor: '#e74c3c',
            color: 'white',
            border: 'none',
            padding: '8px 15px',
            borderRadius: '4px',
            cursor: 'pointer',
            marginTop: '20px'
          }}
        >
          Logout
        </button>
      </div>

      {/* Main content */}
      <div style={{ flex: 1, padding: '20px' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '20px',
          padding: '10px',
          backgroundColor: '#f8f9fa',
          borderRadius: '5px'
        }}>
          <h1>Dashboard</h1>
          {user && (
            <div>
              <span style={{ marginRight: '10px' }}>Welcome, {user.username}</span>
              <span style={{ backgroundColor: '#3498db', color: 'white', padding: '3px 8px', borderRadius: '10px', fontSize: '0.8em' }}>
                {user.role_id === 1 ? 'User' : 'Admin'}
              </span>
            </div>
          )}
        </div>

        {error && <div style={{ color: 'red', marginBottom: '20px' }}>{error}</div>}

        {/* User info card */}
        {user && (
          <div style={{ 
            border: '1px solid #ddd', 
            borderRadius: '5px',
            padding: '20px',
            marginBottom: '20px',
            backgroundColor: 'white',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h2>User Information</h2>
            <p><strong>ID:</strong> {user.id}</p>
            <p><strong>Username:</strong> {user.username}</p>
            <p><strong>Email:</strong> {user.email}</p>
            <p><strong>Role:</strong> {user.role_id === 1 ? 'User' : 'Admin'}</p>
          </div>
        )}

        {/* Dashboard widgets */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
          <div style={{ 
            border: '1px solid #ddd', 
            borderRadius: '5px',
            padding: '20px',
            backgroundColor: 'white',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>Recent Activity</h3>
            <p>No recent activity to display.</p>
          </div>
          
          <div style={{ 
            border: '1px solid #ddd', 
            borderRadius: '5px',
            padding: '20px',
            backgroundColor: 'white',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>System Status</h3>
            <p style={{ color: 'green' }}>All systems operational</p>
          </div>
          
          <div style={{ 
            border: '1px solid #ddd', 
            borderRadius: '5px',
            padding: '20px',
            backgroundColor: 'white',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h3>Quick Stats</h3>
            <p>Total Users: {userCount || '--'}</p>
            <p>Active Sessions: {activeSessions || '--'}</p>
            {user && user.role_id === 2 && (
              <button
                onClick={() => navigate('/users')}
                style={{
                  backgroundColor: '#3498db',
                  color: 'white',
                  border: 'none',
                  padding: '5px 10px',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  marginTop: '10px'
                }}
              >
                Manage Users
              </button>
            )}
          </div>

          <div style={{ 
            border: '1px solid #ddd', 
            borderRadius: '5px',
            padding: '20px',
            backgroundColor: 'white',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            gridColumn: '1 / -1' // Make this widget span the full width
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
              <h3 style={{ margin: 0 }}>Medical Research Tools</h3>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={() => navigate('/pico-search')}
                  style={{
                    backgroundColor: '#3498db',
                    color: 'white',
                    border: 'none',
                    padding: '5px 10px',
                    borderRadius: '3px',
                    cursor: 'pointer'
                  }}
                >
                  PICO Search
                </button>
                <button
                  onClick={() => navigate('/knowledge-base')}
                  style={{
                    backgroundColor: '#8e44ad',
                    color: 'white',
                    border: 'none',
                    padding: '5px 10px',
                    borderRadius: '3px',
                    cursor: 'pointer'
                  }}
                >
                  Knowledge Base
                </button>
              </div>
            </div>
            
            <div style={{ 
              padding: '15px', 
              backgroundColor: '#f0f7ff', 
              borderRadius: '5px', 
              marginBottom: '15px',
              borderLeft: '4px solid #3498db'
            }}>
              <h4 style={{ margin: '0 0 10px 0', color: '#2980b9' }}>Community Acquired Pneumonia (CAP) Research</h4>
              <p style={{ margin: '0 0 10px 0' }}>
                Access the latest research on Community Acquired Pneumonia treatments, 
                diagnostic criteria, and emerging evidence.
              </p>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={() => navigate('/pico-search')}
                  style={{
                    backgroundColor: '#27ae60',
                    color: 'white',
                    border: 'none',
                    padding: '5px 10px',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '0.9em'
                  }}
                >
                  CAP Treatment Research
                </button>
                <button
                  style={{
                    backgroundColor: '#9b59b6',
                    color: 'white',
                    border: 'none',
                    padding: '5px 10px',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '0.9em'
                  }}
                >
                  View Guidelines
                </button>
              </div>
            </div>
            
            <div style={{ fontSize: '0.9em' }}>
              <p><strong>Recent Medical Research Updates:</strong></p>
              <ul style={{ paddingLeft: '20px' }}>
                <li>Procalcitonin-guided antibiotic therapy in CAP shows promising results (Apr 2025)</li>
                <li>New data on antibiotic resistance patterns in Streptococcus pneumoniae (Mar 2025)</li>
                <li>Post-COVID patterns in respiratory infections suggest modified treatment approaches (Feb 2025)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;