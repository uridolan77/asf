import { useApiQuery, useApiPost } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface Entity {
  id: string;
  text: string;
  label: string;
  start: number;
  end: number;
  confidence?: number;
  umls_id?: string;
  umls_name?: string;
}

export interface Relation {
  id: string;
  relation: string;
  head: string;
  head_id: string;
  tail: string;
  tail_id: string;
  confidence?: number;
}

export interface Section {
  id: string;
  title: string;
  content: string;
  start: number;
  end: number;
}

export interface Reference {
  id: string;
  text: string;
  doi?: string;
  url?: string;
  authors?: string[];
  year?: number;
  title?: string;
  journal?: string;
}

export interface Summary {
  abstract?: string;
  key_findings?: string;
  conclusion?: string;
  clinical_implications?: string;
}

export interface SynthesisResult {
  document_id: string;
  title: string;
  content: string;
  entities: Entity[];
  relations: Relation[];
  sections: Section[];
  references: Reference[];
  summary: Summary;
  processing_time: number;
  entity_count: number;
  relation_count: number;
}

export interface SynthesisParams {
  document_id: string;
  options?: {
    extract_entities?: boolean;
    extract_relations?: boolean;
    detect_sections?: boolean;
    parse_references?: boolean;
    generate_summary?: boolean;
    use_enhanced_synthesizer?: boolean;
    use_biomedlm?: boolean;
  };
}

export interface TemporalAnalysisParams {
  text1: string;
  text2: string;
  context?: string;
}

export interface TemporalAnalysisResult {
  confidence: number;
  explanation: string;
  temporal_relation: string;
  contradictions?: {
    text: string;
    explanation: string;
    severity: string;
  }[];
}

export interface BiasAssessmentParams {
  text: string;
  bias_types?: string[];
}

export interface BiasAssessmentResult {
  overall_score: number;
  explanation: string;
  bias_types: {
    type: string;
    score: number;
    explanation: string;
    examples: string[];
  }[];
}

/**
 * Hook for medical research synthesis operations
 */
export function useMedicalResearchSynthesis() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Temporal analysis
  const calculateTemporalConfidence = () => {
    return useApiPost<TemporalAnalysisResult, TemporalAnalysisParams>(
      '/api/ml/temporal-analysis',
      {
        onError: (error) => {
          showError(`Temporal analysis failed: ${error.message}`);
        }
      }
    );
  };

  // Bias assessment
  const assessBias = () => {
    return useApiPost<BiasAssessmentResult, BiasAssessmentParams>(
      '/api/ml/bias-assessment',
      {
        onError: (error) => {
          showError(`Bias assessment failed: ${error.message}`);
        }
      }
    );
  };

  // Research synthesis
  const synthesizeResearch = () => {
    return useApiPost<SynthesisResult, SynthesisParams>(
      '/api/medical/research/synthesize',
      {
        onSuccess: (data) => {
          showSuccess('Research synthesis completed successfully');
        },
        onError: (error) => {
          showError(`Research synthesis failed: ${error.message}`);
        }
      }
    );
  };

  // Get synthesis status
  const getSynthesisStatus = (documentId: string) => {
    return useApiQuery<{ status: string; progress: number; message?: string }>(
      `/api/medical/research/status/${documentId}`,
      ['medical', 'research', 'status', documentId],
      {
        staleTime: 5000, // 5 seconds
        refetchInterval: (data) => {
          // Refetch every 2 seconds if status is 'processing', otherwise stop polling
          return data?.status === 'processing' ? 2000 : false;
        },
        enabled: !!documentId,
        onError: (error) => {
          console.error(`Failed to fetch synthesis status: ${error.message}`);
        }
      }
    );
  };

  // Get synthesis result
  const getSynthesisResult = (documentId: string) => {
    return useApiQuery<SynthesisResult>(
      `/api/medical/research/result/${documentId}`,
      ['medical', 'research', 'result', documentId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!documentId,
        onError: (error) => {
          showError(`Failed to fetch synthesis result: ${error.message}`);
        }
      }
    );
  };

  // Get synthesis history
  const getSynthesisHistory = () => {
    return useApiQuery<{ documents: Array<{ id: string; title: string; processed_at: string }> }>(
      '/api/medical/research/history',
      ['medical', 'research', 'history'],
      {
        staleTime: 60000, // 1 minute
        onError: (error) => {
          showError(`Failed to fetch synthesis history: ${error.message}`);
        }
      }
    );
  };

  // Get synthesis settings
  const getSynthesisSettings = () => {
    return useApiQuery<{
      extract_entities: boolean;
      extract_relations: boolean;
      detect_sections: boolean;
      parse_references: boolean;
      generate_summary: boolean;
      use_enhanced_synthesizer: boolean;
      use_biomedlm: boolean;
    }>(
      '/api/medical/research/settings',
      ['medical', 'research', 'settings'],
      {
        staleTime: 3600000, // 1 hour
        onError: (error) => {
          showError(`Failed to fetch synthesis settings: ${error.message}`);
        }
      }
    );
  };

  // Update synthesis settings
  const updateSynthesisSettings = () => {
    return useApiPost<any, any>(
      '/api/medical/research/settings',
      {
        onSuccess: (data) => {
          showSuccess('Synthesis settings updated successfully');
          queryClient.invalidateQueries({ queryKey: ['medical', 'research', 'settings'] });
        },
        onError: (error) => {
          showError(`Failed to update synthesis settings: ${error.message}`);
        }
      }
    );
  };

  return {
    // Temporal analysis
    calculateTemporalConfidence,
    
    // Bias assessment
    assessBias,
    
    // Research synthesis
    synthesizeResearch,
    getSynthesisStatus,
    getSynthesisResult,
    getSynthesisHistory,
    getSynthesisSettings,
    updateSynthesisSettings
  };
}
