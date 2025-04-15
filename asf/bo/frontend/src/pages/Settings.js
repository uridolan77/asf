import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Settings = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [profileData, setProfileData] = useState({
    username: '',
    email: '',
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  const [systemSettings, setSystemSettings] = useState({
    sessionTimeout: 60,
    enableNotifications: true,
    darkMode: false,
    language: 'en'
  });
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
        setProfileData({
          ...profileData,
          username: response.data.username,
          email: response.data.email
        });

        // If user is admin, fetch system settings
        if (response.data.role_id === 2) {
          try {
            // This endpoint doesn't exist yet, but we're preparing for it
            const settingsResponse = await axios.get('http://localhost:8000/api/settings', {
              headers: {
                'Authorization': `Bearer ${token}`
              }
            });
            if (settingsResponse.data) {
              setSystemSettings(settingsResponse.data);
            }
          } catch (settingsErr) {
            console.error('Failed to fetch settings:', settingsErr);
            // Don't show error - settings may not exist yet
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

  const handleProfileChange = (e) => {
    const { name, value } = e.target;
    setProfileData({
      ...profileData,
      [name]: value
    });
  };

  const handleSystemSettingChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSystemSettings({
      ...systemSettings,
      [name]: type === 'checkbox' ? checked : value
    });
  };

  const validateProfileForm = () => {
    // Reset errors and success
    setError('');
    setSuccess('');

    // Password validation
    if (profileData.newPassword) {
      if (!profileData.currentPassword) {
        setError('Current password is required to set a new password');
        return false;
      }
      if (profileData.newPassword.length < 6) {
        setError('New password must be at least 6 characters');
        return false;
      }
      if (profileData.newPassword !== profileData.confirmPassword) {
        setError('New passwords do not match');
        return false;
      }
    }

    return true;
  };

  const handleProfileSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateProfileForm()) return;

    const token = localStorage.getItem('token');
    try {
      const updateData = {
        username: profileData.username,
        email: profileData.email
      };

      // Only include password fields if user is changing password
      if (profileData.newPassword) {
        updateData.current_password = profileData.currentPassword;
        updateData.new_password = profileData.newPassword;
      }

      await axios.put('http://localhost:8000/api/me', updateData, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      setSuccess('Profile updated successfully');
      // Clear password fields after successful update
      setProfileData({
        ...profileData,
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      });
    } catch (err) {
      console.error('Failed to update profile:', err);
      setError(err.response?.data?.detail || 'Failed to update profile');
    }
  };

  const handleSystemSettingsSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    const token = localStorage.getItem('token');
    try {
      await axios.put('http://localhost:8000/api/settings', systemSettings, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setSuccess('System settings updated successfully');
    } catch (err) {
      console.error('Failed to update system settings:', err);
      setError('Failed to update system settings');
    }
  };

  if (loading) {
    return <div style={{ textAlign: 'center', marginTop: '50px' }}>Loading settings...</div>;
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
            <li 
              style={{ padding: '10px 0', borderBottom: '1px solid #34495e', cursor: 'pointer' }}
              onClick={() => navigate('/dashboard')}
            >
              Dashboard
            </li>
            {user && user.role_id === 2 && (
              <li 
                style={{ padding: '10px 0', borderBottom: '1px solid #34495e', cursor: 'pointer' }}
                onClick={() => navigate('/users')}
              >
                Users
              </li>
            )}
            <li 
              style={{ padding: '10px 0', borderBottom: '1px solid #34495e', fontWeight: 'bold', cursor: 'pointer' }}
            >
              Settings
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
          <h1>Settings</h1>
          {user && (
            <div>
              <span style={{ marginRight: '10px' }}>{user.username}</span>
              <span style={{ backgroundColor: '#3498db', color: 'white', padding: '3px 8px', borderRadius: '10px', fontSize: '0.8em' }}>
                {user.role_id === 1 ? 'User' : 'Admin'}
              </span>
            </div>
          )}
        </div>

        {error && (
          <div style={{ 
            color: 'white', 
            backgroundColor: '#e74c3c', 
            padding: '10px', 
            borderRadius: '5px', 
            marginBottom: '20px' 
          }}>
            {error}
          </div>
        )}

        {success && (
          <div style={{ 
            color: 'white', 
            backgroundColor: '#27ae60', 
            padding: '10px', 
            borderRadius: '5px', 
            marginBottom: '20px' 
          }}>
            {success}
          </div>
        )}

        {/* Profile Settings */}
        <div style={{ 
          border: '1px solid #ddd', 
          borderRadius: '5px',
          padding: '20px',
          marginBottom: '20px',
          backgroundColor: 'white',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <h2>Profile Settings</h2>
          <form onSubmit={handleProfileSubmit}>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="username" style={{ display: 'block', marginBottom: '5px' }}>Username:</label>
              <input
                type="text"
                id="username"
                name="username"
                value={profileData.username}
                onChange={handleProfileChange}
                style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
                required
              />
            </div>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="email" style={{ display: 'block', marginBottom: '5px' }}>Email:</label>
              <input
                type="email"
                id="email"
                name="email"
                value={profileData.email}
                onChange={handleProfileChange}
                style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
                required
              />
            </div>
            <h3>Change Password</h3>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="currentPassword" style={{ display: 'block', marginBottom: '5px' }}>Current Password:</label>
              <input
                type="password"
                id="currentPassword"
                name="currentPassword"
                value={profileData.currentPassword}
                onChange={handleProfileChange}
                style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
              />
            </div>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="newPassword" style={{ display: 'block', marginBottom: '5px' }}>New Password:</label>
              <input
                type="password"
                id="newPassword"
                name="newPassword"
                value={profileData.newPassword}
                onChange={handleProfileChange}
                style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
              />
            </div>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="confirmPassword" style={{ display: 'block', marginBottom: '5px' }}>Confirm New Password:</label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                value={profileData.confirmPassword}
                onChange={handleProfileChange}
                style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
              />
            </div>
            <button 
              type="submit"
              style={{
                backgroundColor: '#3498db',
                color: 'white',
                border: 'none',
                padding: '10px 15px',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Update Profile
            </button>
          </form>
        </div>

        {/* System Settings (Admin Only) */}
        {user && user.role_id === 2 && (
          <div style={{ 
            border: '1px solid #ddd', 
            borderRadius: '5px',
            padding: '20px',
            backgroundColor: 'white',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
          }}>
            <h2>System Settings</h2>
            <form onSubmit={handleSystemSettingsSubmit}>
              <div style={{ marginBottom: '15px' }}>
                <label htmlFor="sessionTimeout" style={{ display: 'block', marginBottom: '5px' }}>Session Timeout (minutes):</label>
                <input
                  type="number"
                  id="sessionTimeout"
                  name="sessionTimeout"
                  value={systemSettings.sessionTimeout}
                  onChange={handleSystemSettingChange}
                  style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
                  min="15"
                  max="480"
                  required
                />
              </div>
              <div style={{ marginBottom: '15px' }}>
                <label htmlFor="language" style={{ display: 'block', marginBottom: '5px' }}>Default Language:</label>
                <select
                  id="language"
                  name="language"
                  value={systemSettings.language}
                  onChange={handleSystemSettingChange}
                  style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
                >
                  <option value="en">English</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                </select>
              </div>
              <div style={{ marginBottom: '15px', display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  id="enableNotifications"
                  name="enableNotifications"
                  checked={systemSettings.enableNotifications}
                  onChange={handleSystemSettingChange}
                  style={{ marginRight: '10px' }}
                />
                <label htmlFor="enableNotifications">Enable Email Notifications</label>
              </div>
              <div style={{ marginBottom: '15px', display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  id="darkMode"
                  name="darkMode"
                  checked={systemSettings.darkMode}
                  onChange={handleSystemSettingChange}
                  style={{ marginRight: '10px' }}
                />
                <label htmlFor="darkMode">Enable Dark Mode</label>
              </div>
              <button 
                type="submit"
                style={{
                  backgroundColor: '#3498db',
                  color: 'white',
                  border: 'none',
                  padding: '10px 15px',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Save System Settings
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
};

export default Settings;
