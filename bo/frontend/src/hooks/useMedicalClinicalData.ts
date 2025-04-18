import { useApiQuery, useApiPost } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';

// Types
interface ClinicalTerm {
  id: string;
  term: string;
  code: string;
  system: string;
  definition?: string;
  synonyms?: string[];
  parent_concepts?: string[];
  child_concepts?: string[];
}

interface ClinicalTrialSummary {
  id: string;
  title: string;
  status: string;
  phase: string;
  conditions: string[];
  interventions: string[];
  start_date: string;
  completion_date?: string;
  enrollment: number;
  url: string;
}

interface ClinicalDataSearchParams {
  query: string;
  filter_type?: string;
  max_results?: number;
  include_trials?: boolean;
  include_terminology?: boolean;
}

interface TerminologySearchParams {
  term: string;
  system?: string;
  max_results?: number;
  include_hierarchy?: boolean;
  include_mappings?: boolean;
}

interface TrialSearchParams {
  condition?: string;
  intervention?: string;
  status?: string;
  phase?: string;
  max_results?: number;
}

/**
 * Hook for medical clinical data operations
 */
export function useMedicalClinicalData() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();

  // Search clinical data (combined terminology and trials)
  const searchClinicalData = (params: ClinicalDataSearchParams) => {
    return useApiPost<any, ClinicalDataSearchParams>(
      '/api/medical/clinical-data/search',
      {
        onError: (error) => {
          showError(`Clinical data search failed: ${error.message}`);
        }
      }
    );
  };

  // Search medical terminology
  const searchTerminology = (params: TerminologySearchParams) => {
    return useApiPost<{ terms: ClinicalTerm[] }, TerminologySearchParams>(
      '/api/medical/terminology/search',
      {
        onError: (error) => {
          showError(`Terminology search failed: ${error.message}`);
        }
      }
    );
  };

  // Get terminology details
  const getTerminologyDetails = (termId: string, system?: string) => {
    return useApiQuery<ClinicalTerm>(
      `/api/medical/terminology/${termId}${system ? `?system=${system}` : ''}`,
      ['medical', 'terminology', termId, system],
      {
        staleTime: 24 * 60 * 60 * 1000, // 24 hours
        enabled: !!termId,
        onError: (error) => {
          showError(`Failed to fetch terminology details: ${error.message}`);
        }
      }
    );
  };

  // Search clinical trials
  const searchClinicalTrials = (params: TrialSearchParams) => {
    return useApiPost<{ trials: ClinicalTrialSummary[] }, TrialSearchParams>(
      '/api/medical/clinical-trials/search',
      {
        onError: (error) => {
          showError(`Clinical trials search failed: ${error.message}`);
        }
      }
    );
  };

  // Get clinical trial details
  const getClinicalTrialDetails = (trialId: string) => {
    return useApiQuery<any>(
      `/api/medical/clinical-trials/${trialId}`,
      ['medical', 'clinical-trials', trialId],
      {
        staleTime: 60 * 60 * 1000, // 1 hour
        enabled: !!trialId,
        onError: (error) => {
          showError(`Failed to fetch clinical trial details: ${error.message}`);
        }
      }
    );
  };

  // Map terminology to clinical trials
  const mapTerminologyToTrials = (termId: string, system?: string) => {
    return useApiQuery<{ trials: ClinicalTrialSummary[] }>(
      `/api/medical/terminology/${termId}/trials${system ? `?system=${system}` : ''}`,
      ['medical', 'terminology', termId, 'trials', system],
      {
        staleTime: 60 * 60 * 1000, // 1 hour
        enabled: !!termId,
        onError: (error) => {
          showError(`Failed to map terminology to trials: ${error.message}`);
        }
      }
    );
  };

  // Get semantic expansion for a term
  const getSemanticExpansion = (term: string) => {
    return useApiQuery<{ expanded_terms: ClinicalTerm[] }>(
      `/api/medical/terminology/expand?term=${encodeURIComponent(term)}`,
      ['medical', 'terminology', 'expand', term],
      {
        staleTime: 24 * 60 * 60 * 1000, // 24 hours
        enabled: !!term,
        onError: (error) => {
          showError(`Failed to get semantic expansion: ${error.message}`);
        }
      }
    );
  };

  return {
    // Combined search
    searchClinicalData,

    // Terminology
    searchTerminology,
    getTerminologyDetails,
    getSemanticExpansion,

    // Clinical trials
    searchClinicalTrials,
    getClinicalTrialDetails,
    mapTerminologyToTrials
  };
}
