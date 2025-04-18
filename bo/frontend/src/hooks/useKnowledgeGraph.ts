import { useApiQuery, useApiPost } from './useApi';
import { useQueryClient } from '@tanstack/react-query';
import { useNotification } from '../context/NotificationContext';
import { useFeatureFlags } from '../context/FeatureFlagContext';

// Types
export interface GraphNode {
  id: string;
  type: string;
  label: string;
  confidence?: number;
  properties?: Record<string, any>;
  isCluster?: boolean;
  nodeCount?: number;
  entityType?: string;
}

export interface GraphLink {
  source: string;
  target: string;
  type: string;
  confidence: number;
  properties?: Record<string, any>;
}

export interface KnowledgeGraph {
  nodes: GraphNode[];
  links: GraphLink[];
  metadata?: {
    document_id?: string;
    document_title?: string;
    created_at?: string;
    entity_count?: number;
    relation_count?: number;
  };
}

export interface GraphQueryParams {
  query: string;
  limit?: number;
  include_properties?: boolean;
  confidence_threshold?: number;
  entity_types?: string[];
  relation_types?: string[];
}

export interface GraphExportParams {
  graph_id: string;
  format: 'json' | 'graphml' | 'csv';
  include_properties?: boolean;
}

export interface GraphLayoutParams {
  graph_id: string;
  layout_algorithm: string;
  options?: Record<string, any>;
}

export interface GraphClusterParams {
  graph_id: string;
  cluster_algorithm: string;
  min_cluster_size?: number;
  options?: Record<string, any>;
}

/**
 * Hook for knowledge graph operations
 */
export function useKnowledgeGraph() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  const { isEnabled } = useFeatureFlags();
  const useMockData = isEnabled('useMockData');

  // Get document knowledge graph
  const getDocumentGraph = (documentId: string) => {
    return useApiQuery<KnowledgeGraph>(
      `/api/document-processing/documents/${documentId}/graph`,
      ['document-processing', 'documents', documentId, 'graph'],
      {
        staleTime: 300000, // 5 minutes
        enabled: !!documentId,
        onError: (error) => {
          showError(`Failed to fetch document knowledge graph: ${error.message}`);
        }
      }
    );
  };

  // Get entity knowledge graph
  const getEntityGraph = (entityId: string, depth: number = 2) => {
    return useApiQuery<KnowledgeGraph>(
      `/api/knowledge-graph/entities/${entityId}?depth=${depth}`,
      ['knowledge-graph', 'entities', entityId, depth],
      {
        staleTime: 300000, // 5 minutes
        enabled: !!entityId,
        onError: (error) => {
          showError(`Failed to fetch entity knowledge graph: ${error.message}`);
        }
      }
    );
  };

  // Query knowledge graph
  const queryGraph = (params: GraphQueryParams) => {
    return useApiPost<KnowledgeGraph, GraphQueryParams>(
      '/api/knowledge-graph/query',
      {
        onError: (error) => {
          showError(`Graph query failed: ${error.message}`);
        }
      }
    );
  };

  // Export knowledge graph
  const exportGraph = (params: GraphExportParams) => {
    return useApiPost<{ url: string }, GraphExportParams>(
      '/api/knowledge-graph/export',
      {
        onSuccess: (data) => {
          showSuccess('Graph exported successfully');
        },
        onError: (error) => {
          showError(`Graph export failed: ${error.message}`);
        }
      }
    );
  };

  // Apply layout to knowledge graph
  const applyGraphLayout = (params: GraphLayoutParams) => {
    return useApiPost<KnowledgeGraph, GraphLayoutParams>(
      '/api/knowledge-graph/layout',
      {
        onSuccess: (data) => {
          showSuccess('Graph layout applied successfully');
        },
        onError: (error) => {
          showError(`Failed to apply graph layout: ${error.message}`);
        }
      }
    );
  };

  // Cluster knowledge graph
  const clusterGraph = (params: GraphClusterParams) => {
    return useApiPost<KnowledgeGraph, GraphClusterParams>(
      '/api/knowledge-graph/cluster',
      {
        onSuccess: (data) => {
          showSuccess('Graph clustering applied successfully');
        },
        onError: (error) => {
          showError(`Failed to cluster graph: ${error.message}`);
        }
      }
    );
  };

  // Get entity types
  const getEntityTypes = () => {
    return useApiQuery<{ entity_types: string[] }>(
      '/api/knowledge-graph/entity-types',
      ['knowledge-graph', 'entity-types'],
      {
        staleTime: 3600000, // 1 hour
        onError: (error) => {
          showError(`Failed to fetch entity types: ${error.message}`);
        }
      }
    );
  };

  // Get relation types
  const getRelationTypes = () => {
    return useApiQuery<{ relation_types: string[] }>(
      '/api/knowledge-graph/relation-types',
      ['knowledge-graph', 'relation-types'],
      {
        staleTime: 3600000, // 1 hour
        onError: (error) => {
          showError(`Failed to fetch relation types: ${error.message}`);
        }
      }
    );
  };

  return {
    getDocumentGraph,
    getEntityGraph,
    queryGraph,
    exportGraph,
    applyGraphLayout,
    clusterGraph,
    getEntityTypes,
    getRelationTypes
  };
}
