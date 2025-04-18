import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

/**
 * Default query client configuration
 */
const defaultQueryClientConfig = {
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      suspense: false, // We'll handle suspense manually for now
    },
    mutations: {
      retry: 1,
    },
  },
};

interface QueryProviderProps {
  children: React.ReactNode;
  config?: typeof defaultQueryClientConfig;
}

/**
 * QueryProvider component
 * 
 * Sets up React Query for the application with default configuration
 * and provides the React Query client to all child components.
 */
const QueryProvider: React.FC<QueryProviderProps> = ({ 
  children, 
  config = defaultQueryClientConfig 
}) => {
  const queryClient = React.useMemo(() => new QueryClient(config), [config]);

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {process.env.NODE_ENV !== 'production' && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  );
};

export default QueryProvider;
