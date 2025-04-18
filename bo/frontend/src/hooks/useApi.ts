// Generic React Query hook for data fetching
import { useQuery, useMutation, QueryKey, UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import api from '../services/apiClient';
import { AxiosRequestConfig } from 'axios';
import { ApiError } from '../types/api';

// Type for the options parameter
interface UseApiQueryOptions<TData, TError> extends Omit<UseQueryOptions<TData, TError, TData, QueryKey>, 'queryKey' | 'queryFn'> {
  config?: AxiosRequestConfig;
}

/**
 * Hook for GET requests using React Query
 * 
 * @param endpoint - API endpoint
 * @param queryKey - Query key for React Query cache
 * @param options - Additional options for React Query and axios
 */
export function useApiQuery<TData = unknown, TError = ApiError>(
  endpoint: string,
  queryKey: QueryKey,
  options?: UseApiQueryOptions<TData, TError>
) {
  const { config, ...queryOptions } = options || {};
  
  return useQuery<TData, TError>({
    queryKey,
    queryFn: () => api.get<TData>(endpoint, config),
    ...queryOptions,
  });
}

/**
 * Hook for POST/PUT/PATCH/DELETE requests using React Query
 * 
 * @param method - HTTP method
 * @param endpoint - API endpoint
 * @param options - Additional options for React Query
 */
export function useApiMutation<TData = unknown, TVariables = unknown, TError = ApiError>(
  method: 'post' | 'put' | 'patch' | 'delete',
  endpoint: string,
  options?: Omit<UseMutationOptions<TData, TError, TVariables>, 'mutationFn'>
) {
  return useMutation<TData, TError, TVariables>({
    mutationFn: (variables: TVariables) => {
      if (method === 'delete') {
        return api.delete<TData>(endpoint);
      }
      return api[method]<TData>(endpoint, variables);
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