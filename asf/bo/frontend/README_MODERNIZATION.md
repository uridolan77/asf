# Frontend Modernization Guide

This guide provides instructions for migrating the existing frontend to the new modern stack with TypeScript, React Query, Zustand, and Cypress for E2E testing.

## Key Enhancements

1. **TypeScript** - For type safety and improved developer experience
2. **React Query** - For data fetching, caching, and state management
3. **Zustand** - For global state management
4. **Form Handling with react-hook-form + Yup** - For consistent form validation
5. **Cypress** - For E2E testing of critical features

## Migration Steps

### Step 1: Convert JavaScript files to TypeScript

Convert `.js` files to `.tsx` (for components) or `.ts` (for utilities):

1. Rename the file extension
2. Add type annotations to function parameters, component props, and return values
3. Define interfaces for data structures and component props
4. Use React.FC for functional components

Example:
```tsx
// Before: Component.js
function UserCard({ user, onDelete }) {
  return (...)
}

// After: UserCard.tsx
interface UserCardProps {
  user: User;
  onDelete: (userId: string) => void;
}

const UserCard: React.FC<UserCardProps> = ({ user, onDelete }) => {
  return (...)
};
```

### Step 2: Replace API calls with React Query hooks

Replace direct API calls or custom hooks with React Query hooks:

1. For GET requests, use `useApiQuery` from `hooks/useApi.ts`
2. For mutation operations (POST/PUT/DELETE), use the appropriate hook from `hooks/useApi.ts`

Example:
```tsx
// Before
const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);

useEffect(() => {
  fetchData('/api/users')
    .then(res => setData(res))
    .finally(() => setLoading(false));
}, []);

// After
const { data, isLoading, error } = useApiQuery<UserData[]>('/api/users', ['users']);
```

### Step 3: Move global state to Zustand stores

Replace Context-based global state with Zustand stores:

1. Identify global state that's currently in Context
2. Create appropriate Zustand stores in the `store` directory
3. Replace Context providers and consumers with store hooks

Example:
```tsx
// Before
const { user } = useAuth();

// After
const user = useAuthStore(state => state.user);
```

### Step 4: Standardize form handling

Convert forms to use the standardized pattern with react-hook-form and Yup:

1. Create a validation schema with Yup
2. Use the `useFormState` hook from `hooks/useFormState.ts`
3. Connect form fields to react-hook-form

Example:
```tsx
// Define schema
const validationSchema = yup.object({
  email: validationSchemas.email,
  password: validationSchemas.password,
});

// In component
interface LoginFormData {
  email: string;
  password: string;
}

const { form, isSubmitting, getErrorMessage } = useFormState<LoginFormData>({
  validationSchema,
});

const { register, handleSubmit } = form;

const onSubmit = handleSubmit((data) => {
  // Handle form submission
});
```

### Step 5: Add E2E tests for critical features

Add Cypress tests for critical features:

1. Authentication flows
2. LLM management UI
3. Claim visualizer
4. Any other critical user flows

Run tests with `npm run test:e2e` or open the Cypress test runner with `npm run test:e2e:open`.

## Development Workflow

1. Start the development server with `npm run dev`
2. TypeScript errors will show in the terminal and browser console
3. Use React Query DevTools (visible in development mode) to debug query state
4. Write tests as you develop new features

## Best Practices

1. **Use TypeScript Strictly**: Enable strict mode in tsconfig.json and fix all type errors
2. **Follow React Query Patterns**: 
   - Use query keys consistently
   - Set appropriate staleTime for your data
   - Use prefetching for anticipated user actions
3. **Keep Zustand Stores Focused**: Create separate stores for different domains
4. **Write Tests for Critical Paths**: Focus on business-critical user flows
5. **Use Data-testid Attributes**: Add data-testid to elements you want to test