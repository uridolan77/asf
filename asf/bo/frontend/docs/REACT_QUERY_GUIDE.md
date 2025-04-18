# React Query Implementation Guide

This guide provides a comprehensive approach for implementing React Query in the BO frontend application to standardize data fetching, caching, and state management.

## Why React Query?

- **Automatic Caching**: Caches data and automatically revalidates
- **Loading & Error States**: Built-in loading and error states
- **Pagination & Infinite Scroll**: Simplified pagination and infinite scroll
- **Prefetching**: Prefetch data before it's needed
- **Optimistic Updates**: Update UI before server confirms changes
- **Automatic Refetching**: Refetch data when window is refocused or network is reconnected
- **Devtools**: Powerful devtools for debugging

## Setup

The project already has React Query configured in `index.tsx`:

```tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
    },
  },
});

root.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>
);
```

## Implementation Steps

### 1. Create API Client

First, create a typed API client using Axios:

```tsx
// src/services/apiClient.ts
import axios, { AxiosRequestConfig, AxiosResponse } from 'axios';

const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for auth
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // Handle errors (401, 403, etc.)
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default apiClient;
```

### 2. Create Custom Hooks

Create custom hooks for data fetching:

```tsx
// src/hooks/useApi.ts
import { useQuery, useMutation, QueryKey, UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import apiClient from '../services/apiClient';
import { AxiosRequestConfig } from 'axios';
import { ApiError } from '../types/api';

// Type for the options parameter
interface UseApiQueryOptions<TData, TError> extends Omit<UseQueryOptions<TData, TError, TData, QueryKey>, 'queryKey' | 'queryFn'> {
  config?: AxiosRequestConfig;
}

/**
 * Hook for GET requests using React Query
 */
export function useApiQuery<TData = unknown, TError = ApiError>(
  endpoint: string,
  queryKey: QueryKey,
  options?: UseApiQueryOptions<TData, TError>
) {
  const { config, ...queryOptions } = options || {};
  
  return useQuery<TData, TError>({
    queryKey,
    queryFn: () => apiClient.get<TData>(endpoint, config),
    ...queryOptions,
  });
}

/**
 * Hook for POST/PUT/PATCH/DELETE requests using React Query
 */
export function useApiMutation<TData = unknown, TVariables = unknown, TError = ApiError>(
  method: 'post' | 'put' | 'patch' | 'delete',
  endpoint: string,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
  return useMutation<TData, TError, TVariables>({
    mutationFn: (variables: TVariables) => {
      if (method === 'delete') {
        return apiClient.delete<TData>(endpoint);
      }
      return apiClient[method]<TData>(endpoint, variables);
    },
    ...options,
  });
}

// Convenience hooks for different HTTP methods
export const useApiGet = useApiQuery;

export function useApiPost<TData = unknown, TVariables = unknown, TError = ApiError>(
  endpoint: string,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
  return useApiMutation<TData, TVariables, TError>('post', endpoint, options);
}

export function useApiPut<TData = unknown, TVariables = unknown, TError = ApiError>(
  endpoint: string,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
  return useApiMutation<TData, TVariables, TError>('put', endpoint, options);
}

export function useApiPatch<TData = unknown, TVariables = unknown, TError = ApiError>(
  endpoint: string,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
  return useApiMutation<TData, TVariables, TError>('patch', endpoint, options);
}

export function useApiDelete<TData = unknown, TVariables = unknown, TError = ApiError>(
  endpoint: string,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
  return useApiMutation<TData, TVariables, TError>('delete', endpoint, options);
}
```

### 3. Create Domain-Specific Hooks

Create hooks for specific domains:

```tsx
// src/hooks/useMCPProviders.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { MCPProvider, ApiError } from '../types/api';
import { useApiQuery, useApiPost, useApiPut, useApiDelete } from './useApi';
import { useNotification } from '../context/NotificationContext';

export function useMCPProviders() {
  const queryClient = useQueryClient();
  const { showSuccess, showError } = useNotification();
  
  // Fetch all providers
  const providersQuery = useApiQuery<MCPProvider[]>(
    '/llm/mcp/providers',
    ['mcpProviders'],
    {
      staleTime: 60000, // 1 minute
      onError: (error) => {
        showError(`Failed to fetch MCP providers: ${error.message}`);
      }
    }
  );
  
  // Create provider
  const createProvider = useApiPost<MCPProvider, Omit<MCPProvider, 'id'>>(
    '/llm/mcp/providers',
    {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['mcpProviders'] });
        showSuccess('Provider created successfully');
      },
      onError: (error) => {
        showError(`Failed to create provider: ${error.message}`);
      }
    }
  );
  
  // Update provider
  const updateProvider = useApiPut<MCPProvider, MCPProvider>(
    '/llm/mcp/providers',
    {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['mcpProviders'] });
        showSuccess('Provider updated successfully');
      },
      onError: (error) => {
        showError(`Failed to update provider: ${error.message}`);
      }
    }
  );
  
  // Delete provider
  const deleteProvider = useApiDelete<void, { id: string }>(
    '/llm/mcp/providers',
    {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['mcpProviders'] });
        showSuccess('Provider deleted successfully');
      },
      onError: (error) => {
        showError(`Failed to delete provider: ${error.message}`);
      }
    }
  );
  
  return {
    providers: providersQuery.data || [],
    isLoading: providersQuery.isLoading,
    isError: providersQuery.isError,
    error: providersQuery.error,
    refetch: providersQuery.refetch,
    createProvider: createProvider.mutate,
    updateProvider: updateProvider.mutate,
    deleteProvider: deleteProvider.mutate,
    isCreating: createProvider.isPending,
    isUpdating: updateProvider.isPending,
    isDeleting: deleteProvider.isPending
  };
}
```

### 4. Using React Query in Components

```tsx
// src/components/LLM/MCP/MCPProviderManagement.tsx
import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Alert,
  CircularProgress,
  Tooltip
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  Check as CheckIcon,
  Warning as WarningIcon
} from '@mui/icons-material';

import { useMCPProviders } from '../../../hooks/useMCPProviders';
import { MCPProvider } from '../../../types/api';
import MCPConfigDialog from './MCPConfigDialog';
import { ContentLoader } from '../../UI/LoadingIndicators';

const MCPProviderManagement: React.FC = () => {
  // State
  const [configDialogOpen, setConfigDialogOpen] = useState<boolean>(false);
  const [configDialogMode, setConfigDialogMode] = useState<'add' | 'edit'>('add');
  const [selectedProvider, setSelectedProvider] = useState<MCPProvider | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState<boolean>(false);
  const [providerToDelete, setProviderToDelete] = useState<string | null>(null);
  
  // Fetch providers using React Query
  const {
    providers,
    isLoading,
    isError,
    error,
    refetch,
    createProvider,
    updateProvider,
    deleteProvider,
    isCreating,
    isUpdating,
    isDeleting
  } = useMCPProviders();
  
  // Handle provider creation
  const handleCreateProvider = () => {
    setSelectedProvider(null);
    setConfigDialogMode('add');
    setConfigDialogOpen(true);
  };
  
  // Handle provider edit
  const handleEditProvider = (provider: MCPProvider) => {
    setSelectedProvider(provider);
    setConfigDialogMode('edit');
    setConfigDialogOpen(true);
  };
  
  // Handle provider save
  const handleSaveProvider = (provider: MCPProvider) => {
    if (configDialogMode === 'add') {
      createProvider(provider);
    } else {
      updateProvider(provider);
    }
    setConfigDialogOpen(false);
  };
  
  // Handle provider delete
  const handleDeleteProvider = (id: string) => {
    deleteProvider({ id });
  };
  
  // Render loading state
  if (isLoading) {
    return <ContentLoader />;
  }
  
  // Render error state
  if (isError) {
    return (
      <Alert 
        severity="error" 
        action={
          <Button color="inherit" size="small" onClick={() => refetch()}>
            Retry
          </Button>
        }
      >
        Failed to load providers: {error?.message || 'Unknown error'}
      </Alert>
    );
  }
  
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">MCP Providers</Typography>
        <Box>
          <Button
            variant="contained"
            color="primary"
            onClick={handleCreateProvider}
          >
            Add Provider
          </Button>
          <IconButton onClick={() => refetch()} sx={{ ml: 1 }}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>
      
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Transport</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {providers.map((provider) => (
              <TableRow key={provider.id}>
                <TableCell>{provider.id}</TableCell>
                <TableCell>{provider.name}</TableCell>
                <TableCell>{provider.transport_type}</TableCell>
                <TableCell>
                  <Chip
                    label={provider.status}
                    color={provider.status === 'connected' ? 'success' : 'error'}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <IconButton onClick={() => handleEditProvider(provider)}>
                    <EditIcon />
                  </IconButton>
                  <IconButton onClick={() => handleDeleteProvider(provider.id)}>
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <MCPConfigDialog
        open={configDialogOpen}
        mode={configDialogMode}
        provider={selectedProvider}
        onClose={() => setConfigDialogOpen(false)}
        onSave={handleSaveProvider}
      />
    </Box>
  );
};

export default MCPProviderManagement;
```

### 5. Optimistic Updates

For a better user experience, implement optimistic updates:

```tsx
// Optimistic update example
const updateTodo = useMutation({
  mutationFn: (todo) => apiClient.put(`/todos/${todo.id}`, todo),
  // When mutate is called:
  onMutate: async (newTodo) => {
    // Cancel any outgoing refetches
    await queryClient.cancelQueries({ queryKey: ['todos'] });
    
    // Snapshot the previous value
    const previousTodos = queryClient.getQueryData(['todos']);
    
    // Optimistically update to the new value
    queryClient.setQueryData(['todos'], (old) => 
      old.map(todo => todo.id === newTodo.id ? newTodo : todo)
    );
    
    // Return a context object with the snapshot
    return { previousTodos };
  },
  // If the mutation fails, use the context returned from onMutate to roll back
  onError: (err, newTodo, context) => {
    queryClient.setQueryData(['todos'], context.previousTodos);
  },
  // Always refetch after error or success:
  onSettled: () => {
    queryClient.invalidateQueries({ queryKey: ['todos'] });
  },
});
```

### 6. Pagination

Implement pagination with React Query:

```tsx
// Pagination example
function Todos() {
  const [page, setPage] = useState(0);
  
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['todos', page],
    queryFn: () => fetchTodos(page),
    keepPreviousData: true, // Keep previous data while fetching new data
  });
  
  return (
    <div>
      {isLoading ? (
        <div>Loading...</div>
      ) : isError ? (
        <div>Error: {error.message}</div>
      ) : (
        <>
          {data.todos.map(todo => (
            <div key={todo.id}>{todo.title}</div>
          ))}
          <div>
            <button
              onClick={() => setPage(old => Math.max(old - 1, 0))}
              disabled={page === 0}
            >
              Previous Page
            </button>
            <button
              onClick={() => setPage(old => old + 1)}
              disabled={!data.hasMore}
            >
              Next Page
            </button>
          </div>
        </>
      )}
    </div>
  );
}
```

### 7. Infinite Scroll

Implement infinite scroll with React Query:

```tsx
// Infinite scroll example
function InfiniteTodos() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
    error
  } = useInfiniteQuery({
    queryKey: ['infiniteTodos'],
    queryFn: ({ pageParam = 0 }) => fetchTodos(pageParam),
    getNextPageParam: (lastPage) => lastPage.nextCursor,
  });
  
  return (
    <div>
      {isLoading ? (
        <div>Loading...</div>
      ) : isError ? (
        <div>Error: {error.message}</div>
      ) : (
        <>
          {data.pages.map((page, i) => (
            <React.Fragment key={i}>
              {page.todos.map(todo => (
                <div key={todo.id}>{todo.title}</div>
              ))}
            </React.Fragment>
          ))}
          <div>
            <button
              onClick={() => fetchNextPage()}
              disabled={!hasNextPage || isFetchingNextPage}
            >
              {isFetchingNextPage
                ? 'Loading more...'
                : hasNextPage
                ? 'Load More'
                : 'Nothing more to load'}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
```

## Best Practices

### 1. Query Keys

Use consistent query keys to ensure proper caching:

```tsx
// Simple key
useQuery({ queryKey: ['todos'], queryFn: fetchTodos });

// Key with parameters
useQuery({ queryKey: ['todo', todoId], queryFn: () => fetchTodo(todoId) });

// Key with filters
useQuery({
  queryKey: ['todos', { status, page }],
  queryFn: () => fetchTodos({ status, page })
});
```

### 2. Prefetching

Prefetch data before it's needed:

```tsx
// Prefetch on hover
const prefetchTodo = (todoId) => {
  queryClient.prefetchQuery({
    queryKey: ['todo', todoId],
    queryFn: () => fetchTodo(todoId)
  });
};

// In your component
<div onMouseEnter={() => prefetchTodo(todo.id)}>
  {todo.title}
</div>
```

### 3. Query Invalidation

Invalidate queries when data changes:

```tsx
// Invalidate all queries with 'todos' in the key
queryClient.invalidateQueries({ queryKey: ['todos'] });

// Invalidate a specific query
queryClient.invalidateQueries({ queryKey: ['todo', todoId] });

// Invalidate multiple queries
queryClient.invalidateQueries({
  predicate: (query) => 
    query.queryKey[0] === 'todos' || query.queryKey[0] === 'todoCount'
});
```

### 4. Dependent Queries

Handle dependent queries:

```tsx
// Get user
const { data: user } = useQuery({
  queryKey: ['user', userId],
  queryFn: () => fetchUser(userId)
});

// Get projects for user
const { data: projects } = useQuery({
  queryKey: ['projects', user?.id],
  queryFn: () => fetchProjects(user.id),
  // Only run this query if user is available
  enabled: !!user,
});
```

### 5. Query Cancellation

Cancel queries when components unmount:

```tsx
// This is handled automatically by React Query, but you can also do it manually
useEffect(() => {
  const controller = new AbortController();
  const signal = controller.signal;
  
  fetchData(signal);
  
  return () => {
    controller.abort();
  };
}, []);
```

### 6. Error Handling

Handle errors consistently:

```tsx
const { data, isLoading, isError, error } = useQuery({
  queryKey: ['todos'],
  queryFn: fetchTodos,
  onError: (error) => {
    // Log error or show notification
    showError(`Failed to fetch todos: ${error.message}`);
  }
});

// In your component
if (isError) {
  return (
    <Alert 
      severity="error" 
      action={
        <Button color="inherit" size="small" onClick={() => refetch()}>
          Retry
        </Button>
      }
    >
      {error.message}
    </Alert>
  );
}
```

## Migration Strategy

1. **Start with API client**: Create a typed API client
2. **Create custom hooks**: Implement useApi hooks
3. **Migrate simple components**: Start with components that make simple API calls
4. **Migrate complex components**: Move on to components with more complex data requirements
5. **Implement optimistic updates**: Add optimistic updates for better UX
6. **Add prefetching**: Implement prefetching for common user flows

## Resources

- [TanStack Query Documentation](https://tanstack.com/query/latest/docs/react/overview)
- [React Query Patterns](https://tkdodo.eu/blog/practical-react-query)
- [React Query Examples](https://github.com/tannerlinsley/react-query/tree/master/examples)
