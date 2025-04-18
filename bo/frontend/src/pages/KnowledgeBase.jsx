import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Grid, Paper } from '@mui/material';

import { useAuth } from '../context/AuthContext.jsx';
import { useNotification } from '../context/NotificationContext.jsx';

import PageLayout from '../components/Layout/PageLayout';
import {
  KnowledgeBaseList,
  KnowledgeBaseDetails,
  CreateKnowledgeBaseDialog
} from '../components/KnowledgeBase';

const KnowledgeBase = () => {
  const { user, api } = useAuth();
  const { showSuccess, showError } = useNotification();
  const [loading, setLoading] = useState(true);
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [selectedKB, setSelectedKB] = useState(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    query: '',
    updateSchedule: 'weekly'
  });
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [actionInProgress, setActionInProgress] = useState(false);
  const navigate = useNavigate();

  // Fetch knowledge bases when component mounts
  useEffect(() => {
    fetchKnowledgeBases();
  }, []);

  // Fetch all knowledge bases
  const fetchKnowledgeBases = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/medical/knowledge-base');
      if (response.data && response.data.data && response.data.data.knowledge_bases) {
        setKnowledgeBases(response.data.data.knowledge_bases);
      }
    } catch (error) {
      console.error('Failed to load knowledge bases:', error);
      showError('Failed to load knowledge bases');
    } finally {
      setLoading(false);
    }
  };

  // Fetch knowledge base details
  const fetchKnowledgeBaseDetails = async (kbId) => {
    setDetailsLoading(true);
    setSelectedKB(null);

    try {
      const response = await api.get(`/api/medical/knowledge-base/${kbId}`);
      if (response.data && response.data.data) {
        setSelectedKB(response.data.data);
      }
    } catch (error) {
      console.error('Failed to load knowledge base details:', error);
      showError('Failed to load knowledge base details');
    } finally {
      setDetailsLoading(false);
    }
  };

  // Handle knowledge base creation
  const handleCreateKnowledgeBase = async () => {
    setActionInProgress(true);

    try {
      await api.post('/api/medical/knowledge-base', {
        name: formData.name,
        query: formData.query,
        update_schedule: formData.updateSchedule
      });
      
      showSuccess(`Knowledge base "${formData.name}" created successfully`);
      setShowAddModal(false);
      setFormData({
        name: '',
        query: '',
        updateSchedule: 'weekly'
      });
      
      // Refresh the knowledge bases list
      await fetchKnowledgeBases();
    } catch (error) {
      console.error('Failed to create knowledge base:', error);
      showError('Failed to create knowledge base');
    } finally {
      setActionInProgress(false);
    }
  };

  // Handle knowledge base update
  const handleUpdateKnowledgeBase = async (kbId) => {
    setActionInProgress(true);

    try {
      await api.post(`/api/medical/knowledge-base/${kbId}/update`);
      showSuccess('Knowledge base update started successfully');
      
      // Refresh the selected KB after a short delay to allow the update to start
      setTimeout(() => {
        fetchKnowledgeBaseDetails(kbId);
      }, 1000);
    } catch (error) {
      console.error('Failed to update knowledge base:', error);
      showError('Failed to update knowledge base');
    } finally {
      setActionInProgress(false);
    }
  };

  // Handle knowledge base deletion
  const handleDeleteKnowledgeBase = async (kbId) => {
    if (!window.confirm('Are you sure you want to delete this knowledge base? This action cannot be undone.')) {
      return;
    }

    setActionInProgress(true);

    try {
      await api.delete(`/api/medical/knowledge-base/${kbId}`);
      showSuccess('Knowledge base deleted successfully');
      setSelectedKB(null);
      
      // Refresh the knowledge bases list
      await fetchKnowledgeBases();
    } catch (error) {
      console.error('Failed to delete knowledge base:', error);
      showError('Failed to delete knowledge base');
    } finally {
      setActionInProgress(false);
    }
  };

  return (
    <PageLayout
      title="Knowledge Base Management"
      breadcrumbs={[{ label: 'Knowledge Base', path: '/knowledge-base' }]}
      loading={loading}
      user={user}
    >
      {/* Main content */}
      <Paper sx={{ p: 0, height: 'calc(100vh - 180px)' }}>
        <Grid container sx={{ height: '100%' }}>
          {/* Knowledge Base List */}
          <Grid item xs={12} md={3} sx={{
            borderRight: '1px solid',
            borderColor: 'divider',
            height: '100%',
            overflow: 'hidden'
          }}>
            <Box sx={{ p: 2, height: '100%' }}>
              <KnowledgeBaseList
                knowledgeBases={knowledgeBases}
                selectedKB={selectedKB}
                onSelectKB={fetchKnowledgeBaseDetails}
                onAddNew={() => setShowAddModal(true)}
              />
            </Box>
          </Grid>

          {/* Knowledge Base Details */}
          <Grid item xs={12} md={9} sx={{ height: '100%', overflow: 'hidden' }}>
            <Box sx={{ p: 3, height: '100%' }}>
              <KnowledgeBaseDetails
                selectedKB={selectedKB}
                loading={detailsLoading}
                actionInProgress={actionInProgress}
                onUpdate={handleUpdateKnowledgeBase}
                onDelete={handleDeleteKnowledgeBase}
                onCreateNew={() => setShowAddModal(true)}
              />
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Create Knowledge Base Dialog */}
      <CreateKnowledgeBaseDialog
        open={showAddModal}
        onClose={() => setShowAddModal(false)}
        formData={formData}
        onChange={setFormData}
        onSubmit={handleCreateKnowledgeBase}
        actionInProgress={actionInProgress}
      />
    </PageLayout>
  );
};

export default KnowledgeBase;