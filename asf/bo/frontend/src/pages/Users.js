import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Users = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [currentUser, setCurrentUser] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newUser, setNewUser] = useState({ username: '', email: '', password: '', role_id: 1 });
  const navigate = useNavigate();

  // Form validation state
  const [formErrors, setFormErrors] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      const token = localStorage.getItem('token');
      if (!token) {
        navigate('/');
        return;
      }

      try {
        // Get current user
        const userResponse = await axios.get('http://localhost:8000/api/me', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        setCurrentUser(userResponse.data);

        // Only admins can access this page
        if (userResponse.data.role_id !== 2) { // Assuming 2 is admin
          navigate('/dashboard');
          return;
        }

        // Get all users
        const usersResponse = await axios.get('http://localhost:8000/api/users', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        setUsers(usersResponse.data);
      } catch (err) {
        console.error('Failed to fetch data:', err);
        setError('Failed to load users data. You may need to log in again.');
        if (err.response && (err.response.status === 401 || err.response.status === 403)) {
          localStorage.removeItem('token');
          navigate('/');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  const handleDeleteUser = async (userId) => {
    if (!window.confirm('Are you sure you want to delete this user?')) return;
    
    const token = localStorage.getItem('token');
    try {
      await axios.delete(`http://localhost:8000/api/users/${userId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setUsers(users.filter(user => user.id !== userId));
    } catch (err) {
      setError('Failed to delete user');
      console.error(err);
    }
  };

  const validateForm = () => {
    const errors = {};
    if (!newUser.username) errors.username = 'Username is required';
    if (!newUser.email) errors.email = 'Email is required';
    if (!newUser.email.includes('@')) errors.email = 'Enter a valid email';
    if (!newUser.password) errors.password = 'Password is required';
    if (newUser.password && newUser.password.length < 6) errors.password = 'Password must be at least 6 characters';
    
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleAddUser = async () => {
    if (!validateForm()) return;
    
    const token = localStorage.getItem('token');
    try {
      const response = await axios.post('http://localhost:8000/api/users', newUser, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setUsers([...users, response.data]);
      setShowAddModal(false);
      setNewUser({ username: '', email: '', password: '', role_id: 1 });
    } catch (err) {
      setError('Failed to add user');
      console.error(err);
      if (err.response && err.response.data) {
        // Set specific error based on backend response
        if (err.response.data.detail.includes('email')) {
          setFormErrors({...formErrors, email: err.response.data.detail});
        }
      }
    }
  };

  if (loading) {
    return <div style={{ textAlign: 'center', marginTop: '50px' }}>Loading users...</div>;
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
            <li style={{ padding: '10px 0', borderBottom: '1px solid #34495e', cursor: 'pointer' }}
                onClick={() => navigate('/dashboard')}>
              Dashboard
            </li>
            <li style={{ padding: '10px 0', borderBottom: '1px solid #34495e', fontWeight: 'bold', cursor: 'pointer' }}>
              Users
            </li>
            <li style={{ padding: '10px 0', borderBottom: '1px solid #34495e', cursor: 'pointer' }}>
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
          <h1>User Management</h1>
          {currentUser && (
            <div>
              <span style={{ marginRight: '10px' }}>Admin: {currentUser.username}</span>
              <button 
                onClick={() => setShowAddModal(true)}
                style={{
                  backgroundColor: '#27ae60',
                  color: 'white',
                  border: 'none',
                  padding: '8px 15px',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Add User
              </button>
            </div>
          )}
        </div>

        {error && <div style={{ color: 'red', marginBottom: '20px' }}>{error}</div>}

        {/* Users Table */}
        <div style={{ 
          border: '1px solid #ddd', 
          borderRadius: '5px',
          padding: '20px',
          marginBottom: '20px',
          backgroundColor: 'white',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '10px', borderBottom: '1px solid #ddd' }}>ID</th>
                <th style={{ textAlign: 'left', padding: '10px', borderBottom: '1px solid #ddd' }}>Username</th>
                <th style={{ textAlign: 'left', padding: '10px', borderBottom: '1px solid #ddd' }}>Email</th>
                <th style={{ textAlign: 'left', padding: '10px', borderBottom: '1px solid #ddd' }}>Role</th>
                <th style={{ textAlign: 'center', padding: '10px', borderBottom: '1px solid #ddd' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.length === 0 ? (
                <tr>
                  <td colSpan="5" style={{ textAlign: 'center', padding: '10px' }}>No users found</td>
                </tr>
              ) : (
                users.map(user => (
                  <tr key={user.id}>
                    <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>{user.id}</td>
                    <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>{user.username}</td>
                    <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>{user.email}</td>
                    <td style={{ padding: '10px', borderBottom: '1px solid #ddd' }}>
                      <span style={{
                        backgroundColor: user.role_id === 2 ? '#3498db' : '#7f8c8d',
                        color: 'white',
                        padding: '3px 8px',
                        borderRadius: '10px',
                        fontSize: '0.8em'
                      }}>
                        {user.role_id === 1 ? 'User' : 'Admin'}
                      </span>
                    </td>
                    <td style={{ padding: '10px', borderBottom: '1px solid #ddd', textAlign: 'center' }}>
                      <button
                        onClick={() => handleDeleteUser(user.id)}
                        disabled={user.id === currentUser.id}
                        style={{
                          backgroundColor: user.id === currentUser.id ? '#95a5a6' : '#e74c3c',
                          color: 'white',
                          border: 'none',
                          padding: '5px 10px',
                          borderRadius: '3px',
                          cursor: user.id === currentUser.id ? 'not-allowed' : 'pointer'
                        }}
                      >
                        {user.id === currentUser.id ? 'Current User' : 'Delete'}
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add User Modal */}
      {showAddModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '5px',
            width: '400px',
            maxWidth: '90%'
          }}>
            <h2>Add New User</h2>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="username">Username:</label><br />
              <input
                type="text"
                id="username"
                value={newUser.username}
                onChange={e => setNewUser({...newUser, username: e.target.value})}
                style={{ width: '100%', padding: '8px' }}
              />
              {formErrors.username && <div style={{ color: 'red', fontSize: '0.8em' }}>{formErrors.username}</div>}
            </div>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="email">Email:</label><br />
              <input
                type="email"
                id="email"
                value={newUser.email}
                onChange={e => setNewUser({...newUser, email: e.target.value})}
                style={{ width: '100%', padding: '8px' }}
              />
              {formErrors.email && <div style={{ color: 'red', fontSize: '0.8em' }}>{formErrors.email}</div>}
            </div>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="password">Password:</label><br />
              <input
                type="password"
                id="password"
                value={newUser.password}
                onChange={e => setNewUser({...newUser, password: e.target.value})}
                style={{ width: '100%', padding: '8px' }}
              />
              {formErrors.password && <div style={{ color: 'red', fontSize: '0.8em' }}>{formErrors.password}</div>}
            </div>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="role">Role:</label><br />
              <select
                id="role"
                value={newUser.role_id}
                onChange={e => setNewUser({...newUser, role_id: parseInt(e.target.value)})}
                style={{ width: '100%', padding: '8px' }}
              >
                <option value="1">User</option>
                <option value="2">Admin</option>
              </select>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '20px' }}>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setNewUser({ username: '', email: '', password: '', role_id: 1 });
                  setFormErrors({});
                }}
                style={{
                  backgroundColor: '#95a5a6',
                  color: 'white',
                  border: 'none',
                  padding: '8px 15px',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleAddUser}
                style={{
                  backgroundColor: '#27ae60',
                  color: 'white',
                  border: 'none',
                  padding: '8px 15px',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Add User
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Users;