import { useApiQuery, useApiPost } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';

// Types
interface ContradictionPair {
  article1: {
    id: string;
    title: string;
    claim: string;
  };
  article2: {
    id: string;
    title: string;
    claim: string;
  };
  contradiction_score: number;
  explanation: string;
}

interface ContradictionAnalysisParams {
  query: string;
  max_results?: number;
  threshold?: number;
  use_biomedlm?: boolean;
  use_tsmixer?: boolean;
  use_lorentz?: boolean;
  use_temporal?: boolean;
}

interface ContradictionAnalysisResponse {
  contradiction_pairs: ContradictionPair[];
  query: string;
  threshold: number;
  models_used: string[];
}

interface CAPAnalysisResponse {
  treatments: Array<{
    name: string;
    efficacy_score: number;
    recommendation_level: string;
    patient_groups: string[];
    contraindications: string[];
  }>;
  diagnostic_criteria: Array<{
    criterion: string;
    sensitivity: number;
    specificity: number;
    recommendation: string;
  }>;
  recent_findings: Array<{
    title: string;
    summary: string;
    year: number;
    impact: string;
    source: string;
  }>;
  meta: {
    last_updated: string;
    guidelines_source: string;
  };
}

interface ScreeningParams {
  query: string;
  max_results?: number;
  stage?: string;
  criteria?: Record<string, string[]>;
}

interface BiasAssessmentParams {
  query: string;
  max_results?: number;
  domains?: string[];
}

/**
 * Hook for medical analysis operations
 */
export function useMedicalAnalysis() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();

  // Analyze contradictions
  const analyzeContradictions = (params: ContradictionAnalysisParams) => {
    return useApiPost<ContradictionAnalysisResponse, ContradictionAnalysisParams>(
      '/api/medical/analysis/contradictions',
      {
        onError: (error) => {
          showError(`Contradiction analysis failed: ${error.message}`);
        }
      }
    );
  };

  // Get CAP analysis
  const {
    data: capAnalysis,
    isLoading: isLoadingCAPAnalysis,
    isError: isErrorCAPAnalysis,
    error: errorCAPAnalysis,
    refetch: refetchCAPAnalysis
  } = useApiQuery<CAPAnalysisResponse>(
    '/api/medical/analysis/cap',
    ['medical', 'analysis', 'cap'],
    {
      staleTime: 60 * 60 * 1000, // 1 hour
      refetchOnWindowFocus: false,
      onError: (error) => {
        showError(`Failed to fetch CAP analysis: ${error.message}`);
      }
    }
  );

  // Screen articles using PRISMA framework
  const screenArticles = (params: ScreeningParams) => {
    return useApiPost<any, ScreeningParams>(
      '/api/medical/screening/prisma',
      {
        onError: (error) => {
          showError(`Article screening failed: ${error.message}`);
        }
      }
    );
  };

  // Assess bias in medical research
  const assessBias = (params: BiasAssessmentParams) => {
    return useApiPost<any, BiasAssessmentParams>(
      '/api/medical/screening/bias-assessment',
      {
        onError: (error) => {
          showError(`Bias assessment failed: ${error.message}`);
        }
      }
    );
  };

  // Get clinical data
  const getClinicalData = (condition: string) => {
    return useApiQuery<any>(
      `/api/medical/clinical-data/${condition}`,
      ['medical', 'clinical-data', condition],
      {
        staleTime: 60 * 60 * 1000, // 1 hour
        enabled: !!condition,
        onError: (error) => {
          showError(`Failed to fetch clinical data: ${error.message}`);
        }
      }
    );
  };

  return {
    // Contradiction analysis
    analyzeContradictions,

    // CAP analysis
    capAnalysis,
    isLoadingCAPAnalysis,
    isErrorCAPAnalysis,
    errorCAPAnalysis,
    refetchCAPAnalysis,

    // Article screening
    screenArticles,

    // Bias assessment
    assessBias,

    // Clinical data
    getClinicalData
  };
}
