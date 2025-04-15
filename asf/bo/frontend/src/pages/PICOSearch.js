import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const PICOSearch = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);
  const [error, setError] = useState('');
  const [picoFormData, setPicoFormData] = useState({
    condition: '',
    interventions: [''],
    outcomes: [''],
    population: '',
    studyDesign: '',
    years: '',
    maxResults: 20
  });
  const [searchResults, setSearchResults] = useState(null);
  const navigate = useNavigate();

  // Fetch user data when component loads
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

  // Handle form field changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setPicoFormData({
      ...picoFormData,
      [name]: value
    });
  };

  // Handle array field changes (interventions, outcomes)
  const handleArrayFieldChange = (index, field, value) => {
    const newArray = [...picoFormData[field]];
    newArray[index] = value;
    setPicoFormData({
      ...picoFormData,
      [field]: newArray
    });
  };

  // Add a new item to an array field
  const addArrayItem = (field) => {
    setPicoFormData({
      ...picoFormData,
      [field]: [...picoFormData[field], '']
    });
  };

  // Remove an item from an array field
  const removeArrayItem = (field, index) => {
    if (picoFormData[field].length > 1) {
      const newArray = [...picoFormData[field]];
      newArray.splice(index, 1);
      setPicoFormData({
        ...picoFormData,
        [field]: newArray
      });
    }
  };

  // Submit PICO search
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSearchLoading(true);
    setSearchResults(null);

    // Filter out empty values from arrays
    const filteredData = {
      ...picoFormData,
      interventions: picoFormData.interventions.filter(item => item.trim() !== ''),
      outcomes: picoFormData.outcomes.filter(item => item.trim() !== '')
    };

    // Convert years to number if present
    if (filteredData.years) {
      filteredData.years = parseInt(filteredData.years, 10);
    }

    const token = localStorage.getItem('token');
    try {
      const response = await axios.post('http://localhost:8000/api/medical/search/pico', {
        condition: filteredData.condition,
        interventions: filteredData.interventions,
        outcomes: filteredData.outcomes,
        population: filteredData.population || null,
        study_design: filteredData.studyDesign || null,
        years: filteredData.years || null,
        max_results: filteredData.maxResults
      }, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.data.success) {
        setSearchResults(response.data.data);
      } else {
        setError('Search failed: ' + response.data.message);
      }
    } catch (err) {
      console.error('Failed to perform PICO search:', err);
      setError('Failed to perform search. ' + (err.response?.data?.message || err.message));
    } finally {
      setSearchLoading(false);
    }
  };

  // Reset form
  const handleReset = () => {
    setPicoFormData({
      condition: '',
      interventions: [''],
      outcomes: [''],
      population: '',
      studyDesign: '',
      years: '',
      maxResults: 20
    });
    setSearchResults(null);
  };

  if (loading) {
    return <div style={{ textAlign: 'center', marginTop: '50px' }}>Loading...</div>;
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
              style={{ padding: '10px 0', borderBottom: '1px solid #34495e', fontWeight: 'bold', cursor: 'pointer' }}
            >
              PICO Search
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
          <h1>PICO Search</h1>
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

        {/* PICO Search Form */}
        <div style={{ 
          backgroundColor: '#fff', 
          borderRadius: '5px',
          padding: '20px',
          marginBottom: '20px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}>
          <h2>Evidence-Based PICO Search</h2>
          <p style={{ color: '#666', marginBottom: '20px' }}>
            Use the PICO framework to structure your clinical question: 
            <strong>P</strong>opulation, <strong>I</strong>ntervention, 
            <strong>C</strong>omparison, <strong>O</strong>utcome
          </p>

          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="condition" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Condition / Problem <span style={{ color: 'red' }}>*</span>
              </label>
              <input
                type="text"
                id="condition"
                name="condition"
                value={picoFormData.condition}
                onChange={handleInputChange}
                placeholder="e.g., Community Acquired Pneumonia"
                required
                style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
              />
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="population" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Population (P)
              </label>
              <input
                type="text"
                id="population"
                name="population"
                value={picoFormData.population}
                onChange={handleInputChange}
                placeholder="e.g., Adults over 65"
                style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
              />
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Interventions (I) <span style={{ color: 'red' }}>*</span>
              </label>
              {picoFormData.interventions.map((intervention, index) => (
                <div key={`intervention-${index}`} style={{ display: 'flex', marginBottom: '8px' }}>
                  <input
                    type="text"
                    value={intervention}
                    onChange={(e) => handleArrayFieldChange(index, 'interventions', e.target.value)}
                    placeholder="e.g., Amoxicillin"
                    required={index === 0}
                    style={{ flex: 1, padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                  />
                  <button
                    type="button"
                    onClick={() => removeArrayItem('interventions', index)}
                    disabled={picoFormData.interventions.length === 1 && index === 0}
                    style={{
                      padding: '0 10px',
                      marginLeft: '5px',
                      backgroundColor: '#e74c3c',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: picoFormData.interventions.length === 1 && index === 0 ? 'not-allowed' : 'pointer',
                      opacity: picoFormData.interventions.length === 1 && index === 0 ? '0.5' : '1'
                    }}
                  >
                    -
                  </button>
                </div>
              ))}
              <button
                type="button"
                onClick={() => addArrayItem('interventions')}
                style={{
                  padding: '5px 10px',
                  backgroundColor: '#3498db',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  marginTop: '5px'
                }}
              >
                + Add Intervention
              </button>
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Outcomes (O) <span style={{ color: 'red' }}>*</span>
              </label>
              {picoFormData.outcomes.map((outcome, index) => (
                <div key={`outcome-${index}`} style={{ display: 'flex', marginBottom: '8px' }}>
                  <input
                    type="text"
                    value={outcome}
                    onChange={(e) => handleArrayFieldChange(index, 'outcomes', e.target.value)}
                    placeholder="e.g., Mortality, Treatment Success"
                    required={index === 0}
                    style={{ flex: 1, padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
                  />
                  <button
                    type="button"
                    onClick={() => removeArrayItem('outcomes', index)}
                    disabled={picoFormData.outcomes.length === 1 && index === 0}
                    style={{
                      padding: '0 10px',
                      marginLeft: '5px',
                      backgroundColor: '#e74c3c',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: picoFormData.outcomes.length === 1 && index === 0 ? 'not-allowed' : 'pointer',
                      opacity: picoFormData.outcomes.length === 1 && index === 0 ? '0.5' : '1'
                    }}
                  >
                    -
                  </button>
                </div>
              ))}
              <button
                type="button"
                onClick={() => addArrayItem('outcomes')}
                style={{
                  padding: '5px 10px',
                  backgroundColor: '#3498db',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  marginTop: '5px'
                }}
              >
                + Add Outcome
              </button>
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="studyDesign" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Study Design
              </label>
              <select
                id="studyDesign"
                name="studyDesign"
                value={picoFormData.studyDesign}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
              >
                <option value="">Any Study Design</option>
                <option value="Randomized Controlled Trial">Randomized Controlled Trial</option>
                <option value="Systematic Review">Systematic Review</option>
                <option value="Meta-Analysis">Meta-Analysis</option>
                <option value="Cohort Study">Cohort Study</option>
                <option value="Case-Control Study">Case-Control Study</option>
                <option value="Case Series">Case Series</option>
                <option value="Case Report">Case Report</option>
              </select>
            </div>

            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="years" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Publication Years (restrict to last N years)
              </label>
              <select
                id="years"
                name="years"
                value={picoFormData.years}
                onChange={handleInputChange}
                style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
              >
                <option value="">Any Year</option>
                <option value="1">Last 1 Year</option>
                <option value="2">Last 2 Years</option>
                <option value="5">Last 5 Years</option>
                <option value="10">Last 10 Years</option>
              </select>
            </div>

            <div style={{ marginBottom: '25px' }}>
              <label htmlFor="maxResults" style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold' }}>
                Maximum Results
              </label>
              <input
                type="number"
                id="maxResults"
                name="maxResults"
                value={picoFormData.maxResults}
                onChange={handleInputChange}
                min="1"
                max="100"
                style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #ddd' }}
              />
            </div>

            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                type="submit"
                disabled={searchLoading}
                style={{
                  backgroundColor: '#27ae60',
                  color: 'white',
                  border: 'none',
                  padding: '12px 20px',
                  borderRadius: '4px',
                  cursor: searchLoading ? 'not-allowed' : 'pointer',
                  opacity: searchLoading ? '0.7' : '1',
                  fontWeight: 'bold'
                }}
              >
                {searchLoading ? 'Searching...' : 'Search'}
              </button>
              <button
                type="button"
                onClick={handleReset}
                style={{
                  backgroundColor: '#95a5a6',
                  color: 'white',
                  border: 'none',
                  padding: '12px 20px',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Reset
              </button>
            </div>
          </form>
        </div>

        {/* Search Results */}
        {searchResults && (
          <div style={{ 
            backgroundColor: '#fff', 
            borderRadius: '5px',
            padding: '20px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
          }}>
            <h2>Search Results</h2>
            <div style={{ background: '#f9f9f9', padding: '15px', borderRadius: '4px', marginBottom: '20px' }}>
              <h3 style={{ marginTop: 0 }}>PICO Query</h3>
              <p style={{ margin: '0', color: '#555' }}>{searchResults.pico_query}</p>
            </div>
            
            <p>Found {searchResults.total_results} results</p>
            
            {searchResults.articles.map((article) => (
              <div 
                key={article.id} 
                style={{ 
                  borderBottom: '1px solid #eee', 
                  padding: '15px 0', 
                  marginBottom: '10px' 
                }}
              >
                <h3 style={{ marginTop: 0, color: '#2980b9' }}>{article.title}</h3>
                <p style={{ margin: '5px 0' }}><strong>Authors:</strong> {article.authors.join(', ')}</p>
                <p style={{ margin: '5px 0' }}><strong>Journal:</strong> {article.journal} ({article.year})</p>
                <p style={{ margin: '5px 0', color: '#555' }}>{article.abstract}</p>
                <div style={{ display: 'flex', alignItems: 'center', marginTop: '10px' }}>
                  <span 
                    style={{ 
                      backgroundColor: getRelevanceColor(article.relevance_score), 
                      color: 'white', 
                      padding: '3px 8px', 
                      borderRadius: '10px', 
                      fontSize: '0.8em',
                      marginRight: '10px'
                    }}
                  >
                    Relevance: {(article.relevance_score * 100).toFixed(0)}%
                  </span>
                  <button
                    style={{
                      backgroundColor: '#3498db',
                      color: 'white',
                      border: 'none',
                      padding: '5px 10px',
                      borderRadius: '3px',
                      cursor: 'pointer',
                      fontSize: '0.9em'
                    }}
                  >
                    View Details
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Helper function to get color based on relevance score
function getRelevanceColor(score) {
  if (score >= 0.9) return '#27ae60'; // High relevance - green
  if (score >= 0.7) return '#2980b9'; // Medium relevance - blue
  return '#f39c12'; // Low relevance - orange
}

export default PICOSearch;