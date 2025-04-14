import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useHistory } from 'react-router-dom';

const Register = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [roleId, setRoleId] = useState('');
  const [roles, setRoles] = useState([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const history = useHistory();

  useEffect(() => {
    // Fetch roles from backend if available, else use static roles
    async function fetchRoles() {
      try {
        const res = await axios.get('/api/roles');
        setRoles(res.data);
      } catch {
        setRoles([
          { id: 1, name: 'User' },
          { id: 2, name: 'Admin' }
        ]);
      }
    }
    fetchRoles();
  }, []);

  const handleRegister = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    try {
      await axios.post('/api/register', {
        username,
        email,
        password,
        role_id: roleId
      });
      setSuccess('Registration successful! Redirecting to login...');
      setTimeout(() => history.push('/login'), 1500);
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
    }
  };

  return (
    <div className="register-container" style={{ width: '350px', margin: '50px auto' }}>
      <h2>Register</h2>
      <form onSubmit={handleRegister}>
        <div style={{ marginBottom: '15px' }}>
          <label htmlFor="username">Username:</label><br />
          <input
            type="text"
            id="username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            required
            style={{ width: '100%' }}
          />
        </div>
        <div style={{ marginBottom: '15px' }}>
          <label htmlFor="email">Email:</label><br />
          <input
            type="email"
            id="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
            style={{ width: '100%' }}
          />
        </div>
        <div style={{ marginBottom: '15px' }}>
          <label htmlFor="password">Password:</label><br />
          <input
            type="password"
            id="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
            style={{ width: '100%' }}
          />
        </div>
        <div style={{ marginBottom: '15px' }}>
          <label htmlFor="role">Role:</label><br />
          <select
            id="role"
            value={roleId}
            onChange={e => setRoleId(e.target.value)}
            required
            style={{ width: '100%' }}
          >
            <option value="">Select a role</option>
            {roles.map(role => (
              <option key={role.id} value={role.id}>{role.name}</option>
            ))}
          </select>
        </div>
        {error && <p style={{ color: 'red' }}>{error}</p>}
        {success && <p style={{ color: 'green' }}>{success}</p>}
        <button type="submit" style={{ width: '100%' }}>Register</button>
      </form>
      <div style={{ marginTop: '10px', textAlign: 'center' }}>
        <span>Already have an account? </span>
        <a href="/login">Login</a>
      </div>
    </div>
  );
};

export default Register;
