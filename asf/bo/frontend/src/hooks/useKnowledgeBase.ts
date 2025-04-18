import { useApiQuery, useApiPost, useApiPut, useApiDelete } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
interface KnowledgeBase {
  id: string;
  name: string;
  query: string;
  update_schedule: string;
  created_at: string;
  article_count: number;
  last_updated: string;
}

interface KnowledgeBaseDetails extends KnowledgeBase {
  articles: Array<{
    id: string;
    title: string;
    journal: string;
    year: number;
    relevance_score: number;
  }>;
  concepts: Array<{
    id: string;
    name: string;
    related_articles: number;
  }>;
}

interface CreateKnowledgeBaseParams {
  name: string;
  query: string;
  update_schedule: string;
}

export interface SearchParams {
  query: string;
  max_results?: number;
  page?: number;
  page_size?: number;
  search_method?: 'semantic' | 'keyword' | 'hybrid';
  use_graph_rag?: boolean;
  filters?: {
    date_range?: [string, string];
    source_types?: string[];
    domains?: string[];
    min_confidence?: number;
  };
}

export interface SearchResult {
  id: string;
  title: string;
  content: string;
  source: string;
  source_type: string;
  url?: string;
  date?: string;
  confidence: number;
  metadata?: Record<string, any>;
}

export interface SearchResponse {
  total_count: number;
  page: number;
  page_size: number;
  results: SearchResult[];
  query_expansion?: string[];
  execution_time?: number;
}

export interface KnowledgeEntry {
  id: string;
  title: string;
  content: string;
  source: string;
  source_type: string;
  url?: string;
  date?: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

export interface SaveEntryParams {
  title: string;
  content: string;
  source: string;
  source_type: string;
  url?: string;
  date?: string;
  tags?: string[];
  metadata?: Record<string, any>;
}

/**
 * Hook for knowledge base operations
 */
export function useKnowledgeBase() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Get all knowledge bases
  const {
    data: knowledgeBases = [],
    isLoading: isLoadingKnowledgeBases,
    isError: isErrorKnowledgeBases,
    error: errorKnowledgeBases,
    refetch: refetchKnowledgeBases
  } = useApiQuery<KnowledgeBase[]>(
    '/api/knowledge-base/list',
    ['knowledgeBase', 'list'],
    {
      staleTime: 60000, // 1 minute
      refetchOnWindowFocus: true,
      onError: (error) => {
        showError(`Failed to fetch knowledge bases: ${error.message}`);
      }
    }
  );

  // Get knowledge base details
  const getKnowledgeBaseDetails = (kbId: string) => {
    return useApiQuery<KnowledgeBaseDetails>(
      `/api/knowledge-base/${kbId}`,
      ['knowledgeBase', kbId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!kbId, // Only run query if kbId is provided
        onError: (error) => {
          showError(`Failed to fetch knowledge base details: ${error.message}`);
        }
      }
    );
  };

  // Create knowledge base
  const {
    mutate: createKnowledgeBase,
    isPending: isCreatingKnowledgeBase,
    isError: isErrorCreatingKnowledgeBase,
    error: errorCreatingKnowledgeBase
  } = useApiPost<KnowledgeBase, CreateKnowledgeBaseParams>(
    '/api/knowledge-base/create',
    {
      onSuccess: (data) => {
        showSuccess('Knowledge base created successfully');
        queryClient.invalidateQueries({ queryKey: ['knowledgeBase', 'list'] });
      },
      onError: (error) => {
        showError(`Failed to create knowledge base: ${error.message}`);
      }
    }
  );

  // Update knowledge base
  const updateKnowledgeBase = (kbId: string) => {
    return useApiPost<KnowledgeBase, void>(
      `/api/knowledge-base/update/${kbId}`,
      {
        onSuccess: (data) => {
          showSuccess('Knowledge base updated successfully');
          queryClient.invalidateQueries({ queryKey: ['knowledgeBase', 'list'] });
          queryClient.invalidateQueries({ queryKey: ['knowledgeBase', kbId] });
        },
        onError: (error) => {
          showError(`Failed to update knowledge base: ${error.message}`);
        }
      }
    );
  };

  // Delete knowledge base
  const deleteKnowledgeBase = (kbId: string) => {
    return useApiDelete<void, void>(
      `/api/knowledge-base/${kbId}`,
      {
        onSuccess: () => {
          showSuccess('Knowledge base deleted successfully');
          queryClient.invalidateQueries({ queryKey: ['knowledgeBase', 'list'] });
        },
        onError: (error) => {
          showError(`Failed to delete knowledge base: ${error.message}`);
        }
      }
    );
  };

  // Export knowledge base
  const exportKnowledgeBase = (kbId: string, format: string) => {
    return useApiPost<{ url: string; format: string }, { format: string }>(
      `/api/knowledge-base/${kbId}/export`,
      {
        onSuccess: (data) => {
          showSuccess(`Knowledge base exported successfully as ${format}`);
        },
        onError: (error) => {
          showError(`Failed to export knowledge base: ${error.message}`);
        }
      }
    );
  };

  // Search within knowledge base
  const searchKnowledgeBase = (params: SearchParams) => {
    return useApiPost<SearchResponse, SearchParams>(
      '/api/knowledge-base/search',
      {
        onError: (error) => {
          showError(`Search failed: ${error.message}`);
        }
      }
    );
  };

  // Get all entries
  const getAllEntries = (page: number = 1, pageSize: number = 20) => {
    return useApiQuery<{ total_count: number; page: number; page_size: number; entries: KnowledgeEntry[] }>(
      `/api/knowledge-base/entries?page=${page}&page_size=${pageSize}`,
      ['knowledge-base', 'entries', page, pageSize],
      {
        staleTime: 60000, // 1 minute
        keepPreviousData: true,
        onError: (error) => {
          showError(`Failed to fetch knowledge base entries: ${error.message}`);
        }
      }
    );
  };

  // Get entry by ID
  const getEntryById = (entryId: string) => {
    return useApiQuery<KnowledgeEntry>(
      `/api/knowledge-base/entries/${entryId}`,
      ['knowledge-base', 'entries', entryId],
      {
        staleTime: 60000, // 1 minute
        enabled: !!entryId,
        onError: (error) => {
          showError(`Failed to fetch knowledge base entry: ${error.message}`);
        }
      }
    );
  };

  // Save entry
  const saveEntry = (params: SaveEntryParams) => {
    return useApiPost<KnowledgeEntry, SaveEntryParams>(
      '/api/knowledge-base/entries',
      {
        onSuccess: (data) => {
          showSuccess('Entry saved successfully');
          // Invalidate queries to refetch data
          queryClient.invalidateQueries({ queryKey: ['knowledge-base', 'entries'] });
        },
        onError: (error) => {
          showError(`Failed to save entry: ${error.message}`);
        }
      }
    );
  };

  // Update entry
  const updateEntry = (entryId: string, params: Partial<SaveEntryParams>) => {
    return useApiPost<KnowledgeEntry, Partial<SaveEntryParams>>(
      `/api/knowledge-base/entries/${entryId}`,
      {
        onSuccess: (data) => {
          showSuccess('Entry updated successfully');
          // Invalidate queries to refetch data
          queryClient.invalidateQueries({ queryKey: ['knowledge-base', 'entries'] });
          queryClient.invalidateQueries({ queryKey: ['knowledge-base', 'entries', entryId] });
        },
        onError: (error) => {
          showError(`Failed to update entry: ${error.message}`);
        }
      }
    );
  };

  // Delete entry
  const deleteEntry = (entryId: string) => {
    return useApiPost<{ success: boolean }, {}>(
      `/api/knowledge-base/entries/${entryId}/delete`,
      {
        onSuccess: (data) => {
          showSuccess('Entry deleted successfully');
          // Invalidate queries to refetch data
          queryClient.invalidateQueries({ queryKey: ['knowledge-base', 'entries'] });
        },
        onError: (error) => {
          showError(`Failed to delete entry: ${error.message}`);
        }
      }
    );
  };

  // Get all tags
  const getAllTags = () => {
    return useApiQuery<{ tags: string[] }>(
      '/api/knowledge-base/tags',
      ['knowledge-base', 'tags'],
      {
        staleTime: 300000, // 5 minutes
        onError: (error) => {
          showError(`Failed to fetch tags: ${error.message}`);
        }
      }
    );
  };

  return {
    // Knowledge bases list
    knowledgeBases,
    isLoadingKnowledgeBases,
    isErrorKnowledgeBases,
    errorKnowledgeBases,
    refetchKnowledgeBases,

    // Knowledge base details
    getKnowledgeBaseDetails,

    // Operations
    createKnowledgeBase,
    isCreatingKnowledgeBase,
    isErrorCreatingKnowledgeBase,
    errorCreatingKnowledgeBase,
    updateKnowledgeBase,
    deleteKnowledgeBase,
    exportKnowledgeBase,
    searchKnowledgeBase,

    // Entries
    getAllEntries,
    getEntryById,
    saveEntry,
    updateEntry,
    deleteEntry,

    // Tags
    getAllTags
  };
}
