import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const KnowledgeBase = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [selectedKB, setSelectedKB] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    query: '',
    updateSchedule: 'weekly'
  });
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [actionInProgress, setActionInProgress] = useState(false);
  const navigate = useNavigate();

  // Fetch user data and knowledge bases when component loads
  useEffect(() => {
    const fetchInitialData = async () => {
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
        setUser(userResponse.data);

        // Fetch knowledge bases
        await fetchKnowledgeBases();
      } catch (err) {
        console.error('Failed to fetch data:', err);
        setError('Failed to load data. You may need to log in again.');
        if (err.response && (err.response.status === 401 || err.response.status === 403)) {
          handleLogout();
        }
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, [navigate]);

  const fetchKnowledgeBases = async () => {
    const token = localStorage.getItem('token');
    try {
      const response = await axios.get('http://localhost:8000/api/medical/knowledge-base', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.data.success) {
        setKnowledgeBases(response.data.data.knowledge_bases);
      } else {
        setError('Failed to load knowledge bases: ' + response.data.message);
      }
    } catch (err) {
      console.error('Failed to fetch knowledge bases:', err);
      setError('Failed to load knowledge bases');
    }
  };

  const fetchKnowledgeBaseDetails = async (kbId) => {
    setDetailsLoading(true);
    setSelectedKB(null);
    setError('');

    const token = localStorage.getItem('token');
    try {
      const response = await axios.get(`http://localhost:8000/api/medical/knowledge-base/${kbId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.data.success) {
        setSelectedKB(response.data.data);
      } else {
        setError('Failed to load knowledge base details: ' + response.data.message);
      }
    } catch (err) {
      console.error('Failed to fetch knowledge base details:', err);
      setError('Failed to load knowledge base details');
    } finally {
      setDetailsLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleCreateKnowledgeBase = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setActionInProgress(true);

    const token = localStorage.getItem('token');
    try {
      const response = await axios.post('http://localhost:8000/api/medical/knowledge-base', {
        name: formData.name,
        query: formData.query,
        update_schedule: formData.updateSchedule
      }, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.data.success) {
        setSuccess(`Knowledge base "${formData.name}" created successfully`);
        setShowAddModal(false);
        setFormData({
          name: '',
          query: '',
          updateSchedule: 'weekly'
        });
        // Refresh the knowledge bases list
        await fetchKnowledgeBases();
      } else {
        setError('Failed to create knowledge base: ' + response.data.message);
      }
    } catch (err) {
      console.error('Failed to create knowledge base:', err);
      setError('Failed to create knowledge base: ' + (err.response?.data?.message || err.message));
    } finally {
      setActionInProgress(false);
    }
  };

  const handleUpdateKnowledgeBase = async (kbId) => {
    setError('');
    setSuccess('');
    setActionInProgress(true);

    const token = localStorage.getItem('token');
    try {
      const response = await axios.post(`http://localhost:8000/api/medical/knowledge-base/${kbId}/update`, {}, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.data.success) {
        setSuccess('Knowledge base update started successfully');
        // Refresh the selected KB after a short delay to allow the update to start
        setTimeout(() => {
          fetchKnowledgeBaseDetails(kbId);
        }, 1000);
      } else {
        setError('Failed to update knowledge base: ' + response.data.message);
      }
    } catch (err) {
      console.error('Failed to update knowledge base:', err);
      setError('Failed to update knowledge base: ' + (err.response?.data?.message || err.message));
    } finally {
      setActionInProgress(false);
    }
  };

  const handleDeleteKnowledgeBase = async (kbId) => {
    if (!window.confirm('Are you sure you want to delete this knowledge base? This action cannot be undone.')) {
      return;
    }

    setError('');
    setSuccess('');
    setActionInProgress(true);

    const token = localStorage.getItem('token');
    try {
      const response = await axios.delete(`http://localhost:8000/api/medical/knowledge-base/${kbId}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.data.success) {
        setSuccess('Knowledge base deleted successfully');
        setSelectedKB(null);
        // Refresh the knowledge bases list
        await fetchKnowledgeBases();
      } else {
        setError('Failed to delete knowledge base: ' + response.data.message);
      }
    } catch (err) {
      console.error('Failed to delete knowledge base:', err);
      setError('Failed to delete knowledge base: ' + (err.response?.data?.message || err.message));
    } finally {
      setActionInProgress(false);
    }
  };

  if (loading) {
    return <div style={{ textAlign: 'center', marginTop: '50px' }}>Loading knowledge bases...</div>;
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
              style={{ padding: '10px 0', borderBottom: '1px solid #34495e', fontWeight: 'bold', cursor: 'pointer' }}
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
      <div style={{ flex: 1, padding: '20px', backgroundColor: '#f5f7fa' }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '20px',
          padding: '10px',
          backgroundColor: '#fff',
          borderRadius: '5px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}>
          <h1>Knowledge Base Management</h1>
          {user && (
            <div>
              <span style={{ marginRight: '10px' }}>Welcome, {user.username}</span>
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

        <div style={{ display: 'flex', gap: '20px', height: 'calc(100vh - 180px)' }}>
          {/* Knowledge Base List */}
          <div style={{ 
            width: '300px', 
            backgroundColor: '#fff', 
            borderRadius: '5px',
            padding: '15px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            overflowY: 'auto'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
              <h2 style={{ margin: 0 }}>Knowledge Bases</h2>
              <button 
                onClick={() => setShowAddModal(true)}
                style={{
                  backgroundColor: '#27ae60',
                  color: 'white',
                  border: 'none',
                  padding: '5px 10px',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                + New
              </button>
            </div>

            {knowledgeBases.length === 0 ? (
              <p>No knowledge bases found. Create one to get started.</p>
            ) : (
              <ul style={{ listStyleType: 'none', padding: 0 }}>
                {knowledgeBases.map(kb => (
                  <li 
                    key={kb.id}
                    onClick={() => fetchKnowledgeBaseDetails(kb.id)}
                    style={{ 
                      padding: '10px', 
                      borderBottom: '1px solid #eee',
                      cursor: 'pointer',
                      backgroundColor: selectedKB && selectedKB.id === kb.id ? '#f0f7ff' : 'transparent',
                      borderLeft: selectedKB && selectedKB.id === kb.id ? '3px solid #3498db' : '3px solid transparent',
                      paddingLeft: selectedKB && selectedKB.id === kb.id ? '7px' : '10px'
                    }}
                  >
                    <div style={{ fontWeight: 'bold' }}>{kb.name}</div>
                    <div style={{ fontSize: '0.9em', color: '#7f8c8d' }}>{kb.article_count} articles</div>
                    <div style={{ fontSize: '0.8em', color: '#95a5a6' }}>
                      Last updated: {new Date(kb.last_updated).toLocaleDateString()}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Knowledge Base Details */}
          <div style={{ 
            flex: 1,
            backgroundColor: '#fff', 
            borderRadius: '5px',
            padding: '20px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            overflowY: 'auto'
          }}>
            {detailsLoading ? (
              <div style={{ textAlign: 'center', padding: '50px' }}>Loading details...</div>
            ) : selectedKB ? (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                  <h2 style={{ margin: 0 }}>{selectedKB.name}</h2>
                  <div>
                    <button 
                      onClick={() => handleUpdateKnowledgeBase(selectedKB.id)}
                      disabled={actionInProgress}
                      style={{
                        backgroundColor: '#3498db',
                        color: 'white',
                        border: 'none',
                        padding: '8px 12px',
                        borderRadius: '4px',
                        cursor: actionInProgress ? 'not-allowed' : 'pointer',
                        marginRight: '10px',
                        opacity: actionInProgress ? 0.7 : 1
                      }}
                    >
                      {actionInProgress ? 'Updating...' : 'Update Now'}
                    </button>
                    <button 
                      onClick={() => handleDeleteKnowledgeBase(selectedKB.id)}
                      disabled={actionInProgress}
                      style={{
                        backgroundColor: '#e74c3c',
                        color: 'white',
                        border: 'none',
                        padding: '8px 12px',
                        borderRadius: '4px',
                        cursor: actionInProgress ? 'not-allowed' : 'pointer',
                        opacity: actionInProgress ? 0.7 : 1
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>

                <div style={{ 
                  backgroundColor: '#f9f9f9', 
                  padding: '15px', 
                  borderRadius: '5px', 
                  marginBottom: '20px' 
                }}>
                  <div><strong>Query:</strong> {selectedKB.query}</div>
                  <div><strong>Update Schedule:</strong> {selectedKB.update_schedule}</div>
                  <div><strong>Created:</strong> {new Date(selectedKB.created_at).toLocaleString()}</div>
                  <div><strong>Last Updated:</strong> {new Date(selectedKB.last_updated).toLocaleString()}</div>
                  <div><strong>Articles:</strong> {selectedKB.article_count}</div>
                </div>

                {/* Articles Section */}
                <h3>Articles ({selectedKB.articles?.length || 0})</h3>
                <div style={{ marginBottom: '20px', maxHeight: '300px', overflowY: 'auto' }}>
                  {selectedKB.articles?.length > 0 ? (
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Title</th>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Journal</th>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Year</th>
                          <th style={{ textAlign: 'left', padding: '8px', borderBottom: '1px solid #ddd' }}>Relevance</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedKB.articles.map(article => (
                          <tr key={article.id}>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{article.title}</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{article.journal}</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>{article.year}</td>
                            <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>
                              <span style={{ 
                                backgroundColor: getRelevanceColor(article.relevance_score),
                                color: 'white',
                                padding: '3px 6px',
                                borderRadius: '10px',
                                fontSize: '0.8em'
                              }}>
                                {(article.relevance_score * 100).toFixed(0)}%
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p>No articles found in this knowledge base.</p>
                  )}
                </div>

                {/* Concepts Section */}
                <h3>Key Concepts ({selectedKB.concepts?.length || 0})</h3>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                  {selectedKB.concepts?.map(concept => (
                    <div 
                      key={concept.id}
                      style={{
                        backgroundColor: '#f0f7ff',
                        padding: '8px 12px',
                        borderRadius: '20px',
                        display: 'flex',
                        alignItems: 'center',
                        border: '1px solid #d6e9ff'
                      }}
                    >
                      <span>{concept.name}</span>
                      <span style={{ 
                        backgroundColor: '#3498db',
                        color: 'white',
                        borderRadius: '50%',
                        padding: '2px 6px',
                        fontSize: '0.7em',
                        marginLeft: '5px'
                      }}>
                        {concept.related_articles}
                      </span>
                    </div>
                  ))}
                  
                  {!selectedKB.concepts?.length && (
                    <p>No concepts extracted yet.</p>
                  )}
                </div>
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '50px', color: '#7f8c8d' }}>
                <p>Select a knowledge base from the list to view details</p>
                <p>- or -</p>
                <button 
                  onClick={() => setShowAddModal(true)}
                  style={{
                    backgroundColor: '#27ae60',
                    color: 'white',
                    border: 'none',
                    padding: '8px 15px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    marginTop: '10px'
                  }}
                >
                  Create New Knowledge Base
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add Knowledge Base Modal */}
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
            width: '500px',
            maxWidth: '90%'
          }}>
            <h2>Create New Knowledge Base</h2>
            <form onSubmit={handleCreateKnowledgeBase}>
              <div style={{ marginBottom: '15px' }}>
                <label htmlFor="name" style={{ display: 'block', marginBottom: '5px' }}>Name:</label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  placeholder="e.g., Community Acquired Pneumonia Research"
                  required
                  style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
                />
              </div>
              <div style={{ marginBottom: '15px' }}>
                <label htmlFor="query" style={{ display: 'block', marginBottom: '5px' }}>Search Query:</label>
                <textarea
                  id="query"
                  name="query"
                  value={formData.query}
                  onChange={handleInputChange}
                  placeholder="e.g., community acquired pneumonia treatment outcomes"
                  required
                  style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd', height: '100px' }}
                />
              </div>
              <div style={{ marginBottom: '20px' }}>
                <label htmlFor="updateSchedule" style={{ display: 'block', marginBottom: '5px' }}>Update Schedule:</label>
                <select
                  id="updateSchedule"
                  name="updateSchedule"
                  value={formData.updateSchedule}
                  onChange={handleInputChange}
                  style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
                >
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
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
                  type="submit"
                  disabled={actionInProgress}
                  style={{
                    backgroundColor: '#27ae60',
                    color: 'white',
                    border: 'none',
                    padding: '8px 15px',
                    borderRadius: '4px',
                    cursor: actionInProgress ? 'not-allowed' : 'pointer',
                    opacity: actionInProgress ? 0.7 : 1
                  }}
                >
                  {actionInProgress ? 'Creating...' : 'Create Knowledge Base'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function to get color based on relevance score
function getRelevanceColor(score) {
  if (score >= 0.9) return '#27ae60'; // High relevance - green
  if (score >= 0.7) return '#2980b9'; // Medium relevance - blue
  return '#f39c12'; // Low relevance - orange
}

export default KnowledgeBase;