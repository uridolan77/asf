# Data Fetching Strategy Guide

This guide explains the standardized data fetching strategy implemented in the BO frontend application, including the approach to handling mock data vs. real API integration.

## Overview

The application uses a consistent data fetching strategy based on React Query, with support for toggling between mock data and real API calls using feature flags and Mock Service Worker (MSW).

## Key Components

### 1. React Query

React Query is used for all data fetching, providing:

- Automatic caching and revalidation
- Loading and error states
- Pagination and infinite scrolling
- Prefetching and background updates
- Optimistic updates
- Devtools for debugging

### 2. Feature Flags

Feature flags are used to toggle between mock data and real API calls:

- `useMockData`: When enabled, the application uses mock data instead of real API calls
- Other feature flags control various aspects of the application

### 3. Mock Service Worker (MSW)

MSW is used to intercept API requests and return mock data:

- Runs in the browser during development
- Provides realistic mock responses
- Can be toggled on/off using feature flags

## Implementation Details

### API Client

The API client (`apiClient.ts`) provides a consistent interface for making API calls:

```typescript
import api from '../services/apiClient';

// Make a GET request
const data = await api.get<MyDataType>('/api/endpoint');

// Make a POST request
const result = await api.post<ResultType>('/api/endpoint', { data });
```

### React Query Hooks

The application provides several hooks for data fetching:

#### 1. Generic API Hooks

```typescript
import { useApiQuery, useApiPost } from '../hooks/useApi';

// Query hook
const { data, isLoading, isError, error, refetch } = useApiQuery<MyDataType>(
  '/api/endpoint',
  ['queryKey'],
  {
    // Options
    staleTime: 60000,
    refetchOnWindowFocus: true
  }
);

// Mutation hook
const { mutate, isPending, isError, error } = useApiPost<ResultType, InputType>(
  '/api/endpoint',
  {
    // Options
    onSuccess: (data) => {
      // Handle success
    },
    onError: (error) => {
      // Handle error
    }
  }
);
```

#### 2. Domain-Specific Hooks

```typescript
import { useLLM } from '../hooks/useLLM';

// LLM hook
const {
  providers,
  isLoadingProviders,
  isErrorProviders,
  errorProviders,
  refetchProviders,
  createProvider,
  isCreatingProvider,
  updateProvider,
  isUpdatingProvider,
  deleteProvider,
  isDeletingProvider
} = useLLM();
```

### Feature Flag Context

The feature flag context (`FeatureFlagContext.tsx`) provides a way to toggle features:

```typescript
import { useFeatureFlags } from '../context/FeatureFlagContext';

// In your component
const { isEnabled } = useFeatureFlags();
const useMockData = isEnabled('useMockData');

// Conditionally render based on feature flag
{useMockData && (
  <Alert severity="info">
    Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
  </Alert>
)}
```

### Mock Service Worker

MSW (`mocks/handlers.ts`) provides mock API responses:

```typescript
// In mocks/handlers.ts
rest.get(`${API_URL}/endpoint`, (req, res, ctx) => {
  return res(
    ctx.status(200),
    ctx.json({
      success: true,
      data: {
        // Mock data
      }
    })
  );
});
```

## Best Practices

### 1. Use Domain-Specific Hooks

Always use domain-specific hooks for data fetching:

```typescript
// Good
const { providers, isLoadingProviders } = useLLM();

// Avoid
const { data: providers, isLoading } = useApiQuery('/api/llm/mcp/providers', ['llm', 'mcp', 'providers']);
```

### 2. Handle Loading and Error States

Always handle loading and error states:

```typescript
if (isLoading) {
  return <CircularProgress />;
}

if (isError) {
  return (
    <Alert severity="error" action={<Button onClick={refetch}>Retry</Button>}>
      {error.message}
    </Alert>
  );
}
```

### 3. Use Optimistic Updates

Use optimistic updates for a better user experience:

```typescript
const { mutate } = useApiPost('/api/endpoint', {
  onMutate: async (newData) => {
    // Cancel any outgoing refetches
    await queryClient.cancelQueries({ queryKey: ['key'] });
    
    // Snapshot the previous value
    const previousData = queryClient.getQueryData(['key']);
    
    // Optimistically update to the new value
    queryClient.setQueryData(['key'], (old) => [...old, newData]);
    
    // Return a context object with the snapshot
    return { previousData };
  },
  onError: (err, newData, context) => {
    // If the mutation fails, use the context to roll back
    queryClient.setQueryData(['key'], context.previousData);
  },
  onSettled: () => {
    // Always refetch after error or success
    queryClient.invalidateQueries({ queryKey: ['key'] });
  }
});
```

### 4. Use Feature Flags for Mock Data

Use feature flags to toggle between mock data and real API calls:

```typescript
const { isEnabled } = useFeatureFlags();
const useMockData = isEnabled('useMockData');

// In your component
{useMockData && (
  <Alert severity="info">
    Using mock data. Toggle the "Use Mock Data" feature flag to use real API data.
  </Alert>
)}
```

## Migration Guide

### 1. Migrating from useEffect + useState

**Before:**

```jsx
const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await apiService.endpoint();
      setData(response.data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  fetchData();
}, []);
```

**After:**

```jsx
const {
  data,
  isLoading,
  isError,
  error,
  refetch
} = useApiQuery('/api/endpoint', ['endpoint']);
```

### 2. Migrating from Custom useApi Hook

**Before:**

```jsx
const {
  data,
  loading,
  error,
  execute
} = useApi(apiService.endpoint, {
  loadOnMount: true,
  onSuccess: (data) => {
    // Handle success
  },
  onError: (error) => {
    // Handle error
  }
});
```

**After:**

```jsx
const {
  data,
  isLoading,
  isError,
  error,
  refetch
} = useApiQuery('/api/endpoint', ['endpoint'], {
  onSuccess: (data) => {
    // Handle success
  },
  onError: (error) => {
    // Handle error
  }
});
```

### 3. Migrating Mock Data

**Before:**

```jsx
// In a service file
const USE_MOCK_DATA = true;

export const fetchData = async () => {
  if (USE_MOCK_DATA) {
    // Generate mock data
    const mockData = generateMockData();
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    return mockData;
  }
  
  // Real API call
  const response = await axios.get('/api/endpoint');
  return response.data;
};
```

**After:**

```jsx
// In mocks/handlers.ts
rest.get(`${API_URL}/endpoint`, (req, res, ctx) => {
  return res(
    ctx.status(200),
    ctx.json({
      success: true,
      data: generateMockData()
    })
  );
});

// In your component
const { isEnabled } = useFeatureFlags();
const useMockData = isEnabled('useMockData');

// Use React Query as normal
const { data, isLoading } = useApiQuery('/api/endpoint', ['endpoint']);
```

## Resources

- [TanStack Query Documentation](https://tanstack.com/query/latest/docs/react/overview)
- [Mock Service Worker Documentation](https://mswjs.io/docs/)
- [Feature Flags Best Practices](https://martinfowler.com/articles/feature-toggles.html)
