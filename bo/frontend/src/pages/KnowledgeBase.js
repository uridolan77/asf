import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Grid, Paper } from '@mui/material';

import apiService from '../services/api';
import { useNotification } from '../context/NotificationContext.jsx';
import useApi from '../hooks/useApi';

import PageLayout from '../components/Layout/PageLayout';
import {
  KnowledgeBaseList,
  KnowledgeBaseDetails,
  CreateKnowledgeBaseDialog
} from '../components/KnowledgeBase';

const KnowledgeBase = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [selectedKB, setSelectedKB] = useState(null);
  const { showSuccess, showError } = useNotification();
  const [showAddModal, setShowAddModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    query: '',
    updateSchedule: 'weekly'
  });
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [actionInProgress, setActionInProgress] = useState(false);
  const navigate = useNavigate();

  // Use API hook for fetching user data
  const {
    data: userData,
    loading: userLoading,
    error: userError,
    execute: fetchUser
  } = useApi(apiService.auth.me, {
    loadOnMount: true,
    onSuccess: (data) => {
      setUser(data);
    },
    onError: (error) => {
      showError('Failed to load user data: ' + error);
    }
  });

  // Use API hook for fetching knowledge bases
  const {
    data: knowledgeBasesData,
    loading: knowledgeBasesLoading,
    error: knowledgeBasesError,
    execute: fetchKnowledgeBases
  } = useApi(apiService.knowledgeBase.getAll, {
    loadOnMount: true,
    onSuccess: (data) => {
      if (data && data.knowledge_bases) {
        setKnowledgeBases(data.knowledge_bases);
      }
    },
    onError: (error) => {
      showError('Failed to load knowledge bases: ' + error);
    }
  });

  // Set overall loading state
  useEffect(() => {
    setLoading(userLoading || knowledgeBasesLoading);
  }, [userLoading, knowledgeBasesLoading]);



  // Use API hook for fetching knowledge base details
  const {
    loading: detailsApiLoading,
    error: detailsError,
    execute: fetchDetails
  } = useApi(apiService.knowledgeBase.getById, {
    onSuccess: (data) => {
      setSelectedKB(data.data);
    },
    onError: (error) => {
      showError('Failed to load knowledge base details: ' + error);
    }
  });

  // Fetch knowledge base details
  const fetchKnowledgeBaseDetails = (kbId) => {
    setDetailsLoading(true);
    setSelectedKB(null);

    fetchDetails(kbId).finally(() => {
      setDetailsLoading(false);
    });
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/');
  };

  // This function is replaced by the onChange prop in CreateKnowledgeBaseDialog

  // Use API hook for creating knowledge base
  const {
    loading: createLoading,
    error: createError,
    execute: createKnowledgeBase
  } = useApi(apiService.knowledgeBase.create, {
    onSuccess: (data) => {
      showSuccess(`Knowledge base "${formData.name}" created successfully`);
      setShowAddModal(false);
      setFormData({
        name: '',
        query: '',
        updateSchedule: 'weekly'
      });
      // Refresh the knowledge bases list
      fetchKnowledgeBases();
    },
    onError: (error) => {
      showError('Failed to create knowledge base: ' + error);
    }
  });

  // Handle knowledge base creation
  const handleCreateKnowledgeBase = () => {
    setActionInProgress(true);

    createKnowledgeBase({
      name: formData.name,
      query: formData.query,
      update_schedule: formData.updateSchedule
    }).finally(() => {
      setActionInProgress(false);
    });
  };

  // Use API hook for updating knowledge base
  const {
    loading: updateLoading,
    error: updateError,
    execute: updateKnowledgeBase
  } = useApi(apiService.knowledgeBase.update, {
    onSuccess: (data) => {
      showSuccess('Knowledge base update started successfully');
      // Refresh the selected KB after a short delay to allow the update to start
      setTimeout(() => {
        fetchKnowledgeBaseDetails(selectedKB.id);
      }, 1000);
    },
    onError: (error) => {
      showError('Failed to update knowledge base: ' + error);
    }
  });

  // Handle knowledge base update
  const handleUpdateKnowledgeBase = (kbId) => {
    setActionInProgress(true);

    updateKnowledgeBase(kbId).finally(() => {
      setActionInProgress(false);
    });
  };

  // Use API hook for deleting knowledge base
  const {
    loading: deleteLoading,
    error: deleteError,
    execute: deleteKnowledgeBase
  } = useApi(apiService.knowledgeBase.delete, {
    onSuccess: (data) => {
      showSuccess('Knowledge base deleted successfully');
      setSelectedKB(null);
      // Refresh the knowledge bases list
      fetchKnowledgeBases();
    },
    onError: (error) => {
      showError('Failed to delete knowledge base: ' + error);
    }
  });

  // Handle knowledge base deletion
  const handleDeleteKnowledgeBase = (kbId) => {
    if (!window.confirm('Are you sure you want to delete this knowledge base? This action cannot be undone.')) {
      return;
    }

    setActionInProgress(true);

    deleteKnowledgeBase(kbId).finally(() => {
      setActionInProgress(false);
    });
  };

  return (
    <PageLayout
      title="Knowledge Base Management"
      breadcrumbs={[{ label: 'Knowledge Base', path: '/knowledge-base' }]}
      loading={loading}
      user={user}
    >
      {/* Notifications are handled by NotificationContext */}

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