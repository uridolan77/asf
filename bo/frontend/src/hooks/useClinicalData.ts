import { useApiQuery, useApiPost } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface ClinicalTerm {
  id: string;
  term: string;
  code: string;
  system: string;
  definition?: string;
  synonyms?: string[];
  parent_concepts?: string[];
  child_concepts?: string[];
}

export interface ClinicalTrial {
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

export interface TermSearchParams {
  term: string;
  max_trials?: number;
  include_hierarchy?: boolean;
  include_mappings?: boolean;
}

export interface TermSearchResponse {
  term: string;
  concepts: ClinicalTerm[];
  trials: ClinicalTrial[];
}

export interface ConceptSearchParams {
  concept_id: string;
  terminology?: string;
  max_trials?: number;
}

export interface ConceptSearchResponse {
  concept: ClinicalTerm;
  trials: ClinicalTrial[];
}

export interface SemanticSearchParams {
  term: string;
  include_similar?: boolean;
  max_trials?: number;
}

export interface SemanticSearchResponse {
  term: string;
  expanded_terms: string[];
  trials: ClinicalTrial[];
}

export interface TrialMappingParams {
  condition: string;
  max_concepts?: number;
}

export interface TrialMappingResponse {
  condition: string;
  concepts: ClinicalTerm[];
  mappings: Array<{
    concept_id: string;
    concept_name: string;
    trial_count: number;
  }>;
}

/**
 * Hook for clinical data operations
 */
export function useClinicalData() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Search for a term and find related concepts and trials
  const searchConceptAndTrials = (params: TermSearchParams) => {
    return useApiPost<TermSearchResponse, TermSearchParams>(
      '/api/medical/clinical-data/term-search',
      {
        onError: (error) => {
          showError(`Term search failed: ${error.message}`);
        }
      }
    );
  };

  // Search for trials by concept ID
  const getTrialsByConceptId = (params: ConceptSearchParams) => {
    return useApiPost<ConceptSearchResponse, ConceptSearchParams>(
      '/api/medical/clinical-data/concept-trials',
      {
        onError: (error) => {
          showError(`Concept search failed: ${error.message}`);
        }
      }
    );
  };

  // Search for trials with semantic expansion
  const findTrialsWithSemanticExpansion = (params: SemanticSearchParams) => {
    return useApiPost<SemanticSearchResponse, SemanticSearchParams>(
      '/api/medical/clinical-data/semantic-search',
      {
        onError: (error) => {
          showError(`Semantic search failed: ${error.message}`);
        }
      }
    );
  };

  // Map condition to concepts and find related trials
  const mapConditionToConcepts = (params: TrialMappingParams) => {
    return useApiPost<TrialMappingResponse, TrialMappingParams>(
      '/api/medical/clinical-data/trial-mapping',
      {
        onError: (error) => {
          showError(`Trial mapping failed: ${error.message}`);
        }
      }
    );
  };

  // Get concept details
  const getConceptDetails = (conceptId: string, system?: string) => {
    return useApiQuery<ClinicalTerm>(
      `/api/medical/clinical-data/concept/${conceptId}${system ? `?system=${system}` : ''}`,
      ['medical', 'clinical-data', 'concept', conceptId, system],
      {
        staleTime: 24 * 60 * 60 * 1000, // 24 hours
        enabled: !!conceptId,
        onError: (error) => {
          showError(`Failed to fetch concept details: ${error.message}`);
        }
      }
    );
  };

  // Get trial details
  const getTrialDetails = (trialId: string) => {
    return useApiQuery<ClinicalTrial>(
      `/api/medical/clinical-data/trial/${trialId}`,
      ['medical', 'clinical-data', 'trial', trialId],
      {
        staleTime: 60 * 60 * 1000, // 1 hour
        enabled: !!trialId,
        onError: (error) => {
          showError(`Failed to fetch trial details: ${error.message}`);
        }
      }
    );
  };

  // Get term suggestions
  const getTermSuggestions = (query: string) => {
    return useApiQuery<{ suggestions: string[] }>(
      `/api/medical/clinical-data/term-suggestions?query=${encodeURIComponent(query)}`,
      ['medical', 'clinical-data', 'term-suggestions', query],
      {
        staleTime: 60 * 60 * 1000, // 1 hour
        enabled: !!query && query.length >= 2,
        onError: (error) => {
          console.error(`Failed to fetch term suggestions: ${error.message}`);
        }
      }
    );
  };

  return {
    // Term search
    searchConceptAndTrials,
    
    // Concept search
    getTrialsByConceptId,
    
    // Semantic search
    findTrialsWithSemanticExpansion,
    
    // Trial mapping
    mapConditionToConcepts,
    
    // Details
    getConceptDetails,
    getTrialDetails,
    
    // Suggestions
    getTermSuggestions
  };
}
