// frontend/src/pages/Login.js
import React, { useState } from 'react';
import axios from 'axios';
import { useHistory } from 'react-router-dom';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const history = useHistory();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');

    try {
      // FastAPI OAuth2 expects form data, not JSON
      const params = new URLSearchParams();
      params.append('username', username); // This is actually email in your backend
      params.append('password', password);
      const response = await axios.post('/api/login', params, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });
      if (response.data.access_token) {
        // Store token and redirect
        localStorage.setItem('token', response.data.access_token);
        history.push('/dashboard');
      } else {
        setError('Login failed.');
      }
    } catch (err) {
      setError("Invalid login credentials");
    }
  };

  return (
    <div className="login-container" style={{ width: '300px', margin: '50px auto' }}>
      <h2>Login</h2>
      <form onSubmit={handleLogin}>
        <div style={{ marginBottom: '15px' }}>
          <label htmlFor="username">Username:</label><br />
          <input 
            type="text" 
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
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
            onChange={(e) => setPassword(e.target.value)}
            required 
            style={{ width: '100%' }}
          />
        </div>
        {error && <p style={{color:'red'}}>{error}</p>}
        <button type="submit" style={{ width: '100%' }}>Login</button>
      </form>
      <div style={{ marginTop: '10px', textAlign: 'center' }}>
        <span>Don't have an account? </span>
        <a href="/register">Register</a>
      </div>
    </div>
  );
};

export default Login;