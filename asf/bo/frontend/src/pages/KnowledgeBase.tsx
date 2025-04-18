import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Grid, Paper, CircularProgress, Alert } from '@mui/material';

import { useAuth } from '../hooks/useAuth';
import { useKnowledgeBase } from '../hooks/useKnowledgeBase';
import { useNotification } from '../context/NotificationContext';

import PageLayout from '../components/Layout/PageLayout';
import {
  KnowledgeBaseList,
  KnowledgeBaseDetails,
  CreateKnowledgeBaseDialog
} from '../components/KnowledgeBase';

// Types
interface FormData {
  name: string;
  query: string;
  updateSchedule: string;
}

const KnowledgeBase: React.FC = () => {
  // State
  const [selectedKBId, setSelectedKBId] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState<boolean>(false);
  const [formData, setFormData] = useState<FormData>({
    name: '',
    query: '',
    updateSchedule: 'weekly'
  });
  
  // Hooks
  const navigate = useNavigate();
  const { showSuccess, showError } = useNotification();
  
  // Auth hook
  const { 
    user, 
    isLoadingUser, 
    isErrorUser, 
    errorUser 
  } = useAuth();
  
  // Knowledge base hooks
  const {
    knowledgeBases,
    isLoadingKnowledgeBases,
    isErrorKnowledgeBases,
    errorKnowledgeBases,
    refetchKnowledgeBases,
    createKnowledgeBase,
    isCreatingKnowledgeBase
  } = useKnowledgeBase();
  
  // Get knowledge base details
  const {
    data: selectedKB,
    isLoading: isLoadingKBDetails,
    isError: isErrorKBDetails,
    error: errorKBDetails,
    refetch: refetchKBDetails
  } = useKnowledgeBase().getKnowledgeBaseDetails(selectedKBId || '');
  
  // Update knowledge base
  const {
    mutate: updateKB,
    isPending: isUpdatingKB
  } = useKnowledgeBase().updateKnowledgeBase(selectedKBId || '');
  
  // Delete knowledge base
  const {
    mutate: deleteKB,
    isPending: isDeletingKB
  } = useKnowledgeBase().deleteKnowledgeBase(selectedKBId || '');
  
  // Determine if any action is in progress
  const actionInProgress = isCreatingKnowledgeBase || isUpdatingKB || isDeletingKB;
  
  // Overall loading state
  const isLoading = isLoadingUser || isLoadingKnowledgeBases;
  
  // Handle knowledge base selection
  const handleSelectKB = (kbId: string) => {
    setSelectedKBId(kbId);
  };
  
  // Handle knowledge base creation
  const handleCreateKnowledgeBase = () => {
    createKnowledgeBase({
      name: formData.name,
      query: formData.query,
      update_schedule: formData.updateSchedule
    });
    
    // Reset form and close modal
    setShowAddModal(false);
    setFormData({
      name: '',
      query: '',
      updateSchedule: 'weekly'
    });
  };
  
  // Handle knowledge base update
  const handleUpdateKnowledgeBase = (kbId: string) => {
    if (!kbId) return;
    
    updateKB();
  };
  
  // Handle knowledge base deletion
  const handleDeleteKnowledgeBase = (kbId: string) => {
    if (!kbId) return;
    
    if (!window.confirm('Are you sure you want to delete this knowledge base? This action cannot be undone.')) {
      return;
    }
    
    deleteKB();
  };
  
  return (
    <PageLayout
      title="Knowledge Base Management"
      breadcrumbs={[{ label: 'Knowledge Base', path: '/knowledge-base' }]}
      loading={isLoading}
      user={user}
    >
      {isErrorKnowledgeBases && (
        <Alert 
          severity="error" 
          sx={{ mb: 2 }}
          action={
            <button onClick={() => refetchKnowledgeBases()}>
              Retry
            </button>
          }
        >
          Failed to load knowledge bases: {errorKnowledgeBases?.message || 'Unknown error'}
        </Alert>
      )}
      
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
              {isLoadingKnowledgeBases ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <KnowledgeBaseList
                  knowledgeBases={knowledgeBases}
                  selectedKB={selectedKB}
                  onSelectKB={handleSelectKB}
                  onAddNew={() => setShowAddModal(true)}
                />
              )}
            </Box>
          </Grid>

          {/* Knowledge Base Details */}
          <Grid item xs={12} md={9} sx={{ height: '100%', overflow: 'hidden' }}>
            <Box sx={{ p: 3, height: '100%' }}>
              <KnowledgeBaseDetails
                selectedKB={selectedKB}
                loading={isLoadingKBDetails}
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
