import { useApiQuery, useApiPost } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface TemporalAnalysisParams {
  publication_date: string;
  reference_date?: string;
  domain?: string;
  include_details?: boolean;
}

export interface TemporalAnalysisResult {
  confidence_score: number;
  temporal_relevance: number;
  temporal_context: string;
  temporal_factors: Record<string, number>;
  details?: {
    time_decay: number;
    domain_evolution: number;
    citation_impact: number;
    guideline_changes: number;
    methodology_advances: number;
  };
  recommendations?: string[];
}

export interface BiasAssessmentParams {
  query: string;
  domains?: string[];
  max_results?: number;
  assessment_type?: 'publication' | 'methodology' | 'reporting' | 'comprehensive';
}

export interface BiasAssessmentResult {
  overall_bias_score: number;
  domains_assessed: string[];
  bias_by_domain: Record<string, number>;
  bias_factors: Array<{
    factor: string;
    score: number;
    description: string;
  }>;
  recommendations: string[];
  articles_analyzed: number;
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
  };
}

export interface SynthesisResult {
  document_id: string;
  title: string;
  entities: Array<{
    id: string;
    text: string;
    type: string;
    start: number;
    end: number;
    confidence?: number;
  }>;
  relations: Array<{
    id: string;
    type: string;
    source_id: string;
    target_id: string;
    confidence?: number;
  }>;
  sections: Array<{
    id: string;
    title: string;
    content: string;
    start: number;
    end: number;
  }>;
  summary?: {
    abstract: string;
    key_findings: string;
    clinical_implications: string;
  };
}

/**
 * Hook for medical research operations
 */
export function useMedicalResearch() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Temporal analysis
  const calculateTemporalConfidence = (params: TemporalAnalysisParams) => {
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
  const assessBias = (params: BiasAssessmentParams) => {
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
  const synthesizeResearch = (params: SynthesisParams) => {
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

  return {
    // Temporal analysis
    calculateTemporalConfidence,
    
    // Bias assessment
    assessBias,
    
    // Research synthesis
    synthesizeResearch,
    getSynthesisStatus,
    getSynthesisResult
  };
}
