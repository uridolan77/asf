import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import {
  Box, Button, Card, CardContent, Chip, Dialog, DialogActions,
  DialogContent, DialogContentText, DialogTitle, Divider, FormControl,
  Grid, IconButton, InputLabel, MenuItem, Paper, Select, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow, TextField,
  Typography, Alert, Snackbar
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Person as PersonIcon
} from '@mui/icons-material';

import PageLayout from '../components/Layout/PageLayout';

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

  // Actions for the page header
  const pageActions = (
    <Button
      variant="contained"
      color="secondary"
      startIcon={<AddIcon />}
      onClick={() => setShowAddModal(true)}
    >
      Add User
    </Button>
  );

  return (
    <PageLayout
      title="User Management"
      breadcrumbs={[{ label: 'Users', path: '/users' }]}
      loading={loading}
      user={currentUser}
      actions={pageActions}
    >
      {/* Error message */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Users Table */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            System Users
          </Typography>
          <Divider sx={{ mb: 2 }} />

          <TableContainer component={Paper} elevation={0}>
            <Table sx={{ minWidth: 650 }}>
              <TableHead>
                <TableRow>
                  <TableCell>ID</TableCell>
                  <TableCell>Username</TableCell>
                  <TableCell>Email</TableCell>
                  <TableCell>Role</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {users.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center">No users found</TableCell>
                  </TableRow>
                ) : (
                  users.map(user => (
                    <TableRow key={user.id}>
                      <TableCell>{user.id}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <PersonIcon sx={{ mr: 1, color: 'action.active' }} />
                          {user.username}
                        </Box>
                      </TableCell>
                      <TableCell>{user.email}</TableCell>
                      <TableCell>
                        <Chip
                          label={user.role_id === 1 ? 'User' : 'Admin'}
                          color={user.role_id === 2 ? 'primary' : 'info'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell align="center">
                        <IconButton
                          color="error"
                          onClick={() => handleDeleteUser(user.id)}
                          disabled={user.id === currentUser?.id}
                          size="small"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Add User Dialog */}
      <Dialog open={showAddModal} onClose={() => {
        setShowAddModal(false);
        setNewUser({ username: '', email: '', password: '', role_id: 1 });
        setFormErrors({});
      }}>
        <DialogTitle>Add New User</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Create a new user account with the following details.
          </DialogContentText>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                autoFocus
                margin="dense"
                id="username"
                label="Username"
                type="text"
                fullWidth
                variant="outlined"
                value={newUser.username}
                onChange={e => setNewUser({...newUser, username: e.target.value})}
                error={!!formErrors.username}
                helperText={formErrors.username}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                margin="dense"
                id="email"
                label="Email Address"
                type="email"
                fullWidth
                variant="outlined"
                value={newUser.email}
                onChange={e => setNewUser({...newUser, email: e.target.value})}
                error={!!formErrors.email}
                helperText={formErrors.email}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                margin="dense"
                id="password"
                label="Password"
                type="password"
                fullWidth
                variant="outlined"
                value={newUser.password}
                onChange={e => setNewUser({...newUser, password: e.target.value})}
                error={!!formErrors.password}
                helperText={formErrors.password}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel id="role-label">Role</InputLabel>
                <Select
                  labelId="role-label"
                  id="role"
                  value={newUser.role_id}
                  label="Role"
                  onChange={e => setNewUser({...newUser, role_id: parseInt(e.target.value)})}
                >
                  <MenuItem value={1}>User</MenuItem>
                  <MenuItem value={2}>Admin</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setShowAddModal(false);
            setNewUser({ username: '', email: '', password: '', role_id: 1 });
            setFormErrors({});
          }}>Cancel</Button>
          <Button onClick={handleAddUser} variant="contained" color="secondary">
            Add User
          </Button>
        </DialogActions>
      </Dialog>
    </PageLayout>
  );
};

export default Users;