# TypeScript Migration Guide

This guide provides a step-by-step approach for migrating JavaScript components to TypeScript in the BO frontend application.

## Why TypeScript?

- **Type Safety**: Catch errors at compile time rather than runtime
- **Better IDE Support**: Improved autocompletion, navigation, and refactoring
- **Self-Documenting Code**: Types serve as documentation
- **Enhanced Maintainability**: Easier to understand and refactor code

## Prerequisites

The project already has TypeScript configured with:
- `tsconfig.json` for TypeScript configuration
- `vite.config.ts` for build configuration
- Required dependencies in `package.json`

## Migration Process

### 1. Preparation

Before converting a file, understand its purpose and dependencies:

- What does the component do?
- What props does it accept?
- What state does it manage?
- What API calls does it make?

### 2. File Conversion Steps

1. **Rename the file**:
   - Change `.js` or `.jsx` to `.tsx` for React components
   - Change `.js` to `.ts` for utility functions

2. **Add type definitions**:
   - Define prop types using interfaces
   - Add return types to functions
   - Type state variables and hooks

3. **Fix type errors**:
   - Address any TypeScript errors
   - Use type assertions when necessary (but sparingly)
   - Add null checks where needed

4. **Update imports**:
   - Update imports to use the new file extensions
   - Import types from libraries or create your own

### 3. Example: Converting a Component

**Before (Component.jsx):**
```jsx
import React, { useState, useEffect } from 'react';
import { Button, TextField } from '@mui/material';

const MyComponent = ({ initialValue, onSubmit }) => {
  const [value, setValue] = useState(initialValue);
  const [error, setError] = useState(null);

  useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  const handleSubmit = async () => {
    try {
      await onSubmit(value);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div>
      <TextField 
        value={value} 
        onChange={(e) => setValue(e.target.value)} 
        error={!!error}
        helperText={error}
      />
      <Button onClick={handleSubmit}>Submit</Button>
    </div>
  );
};

export default MyComponent;
```

**After (Component.tsx):**
```tsx
import React, { useState, useEffect, ChangeEvent } from 'react';
import { Button, TextField } from '@mui/material';

interface MyComponentProps {
  initialValue: string;
  onSubmit: (value: string) => Promise<void>;
}

const MyComponent: React.FC<MyComponentProps> = ({ initialValue, onSubmit }) => {
  const [value, setValue] = useState<string>(initialValue);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  const handleSubmit = async (): Promise<void> => {
    try {
      await onSubmit(value);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
    setValue(e.target.value);
  };

  return (
    <div>
      <TextField 
        value={value} 
        onChange={handleChange} 
        error={!!error}
        helperText={error}
      />
      <Button onClick={handleSubmit}>Submit</Button>
    </div>
  );
};

export default MyComponent;
```

### 4. Common Type Definitions

Create a `types` directory with common type definitions:

```tsx
// src/types/api.ts
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface ApiError {
  message: string;
  status?: number;
  details?: Record<string, any>;
}
```

### 5. Typing React Query

When using React Query, add proper types:

```tsx
import { useQuery, useMutation } from '@tanstack/react-query';
import { User, ApiError } from '../types/api';

// For queries
const { data, isLoading, error } = useQuery<User[], ApiError>({
  queryKey: ['users'],
  queryFn: fetchUsers
});

// For mutations
const mutation = useMutation<void, ApiError, { id: string }>({
  mutationFn: deleteUser,
  onSuccess: () => {
    // Handle success
  }
});
```

### 6. Typing Forms with React Hook Form

```tsx
import { useForm, SubmitHandler } from 'react-hook-form';

interface FormValues {
  name: string;
  email: string;
  age: number;
}

const { register, handleSubmit, formState: { errors } } = useForm<FormValues>();

const onSubmit: SubmitHandler<FormValues> = (data) => {
  console.log(data);
};
```

## Best Practices

1. **Start with simple components**: Begin with smaller, simpler components
2. **Create type files**: Group related types in dedicated files
3. **Use interfaces for objects**: Prefer interfaces for object types
4. **Use type for unions/intersections**: Use type for complex type combinations
5. **Avoid `any`**: Use specific types or `unknown` instead of `any`
6. **Use TypeScript's utility types**: Leverage `Partial<T>`, `Pick<T>`, `Omit<T>`, etc.
7. **Type assertions as a last resort**: Use type assertions (`as Type`) only when necessary

## Troubleshooting Common Issues

### 1. Event Handlers

```tsx
// Typing event handlers
const handleChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
  setValue(e.target.value);
};

// For form submissions
const handleFormSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
  e.preventDefault();
  // Form submission logic
};
```

### 2. Children Props

```tsx
// For components that accept children
interface LayoutProps {
  children: React.ReactNode;
  title?: string;
}

const Layout: React.FC<LayoutProps> = ({ children, title }) => {
  // Component implementation
};
```

### 3. Refs

```tsx
// Using refs
const inputRef = useRef<HTMLInputElement>(null);

// Later in your component
<input ref={inputRef} />;
```

### 4. Default Props

```tsx
// Using default props
interface ButtonProps {
  label: string;
  variant?: 'primary' | 'secondary';
  onClick?: () => void;
}

const Button: React.FC<ButtonProps> = ({ 
  label, 
  variant = 'primary', 
  onClick 
}) => {
  // Component implementation
};
```

## Migration Priority

1. **Core components**: Shared components used throughout the application
2. **API services**: Data fetching and API interaction code
3. **Utility functions**: Helper functions and utilities
4. **Page components**: Individual page components
5. **Context providers**: Global state management

## Testing TypeScript Components

When writing tests for TypeScript components, ensure you:

1. Type your test props and mocks
2. Use proper type assertions in your tests
3. Test type-specific edge cases (null, undefined, etc.)

```tsx
import { render, screen } from '@testing-library/react';
import UserProfile from './UserProfile';
import { User } from '../types/api';

const mockUser: User = {
  id: '1',
  username: 'testuser',
  email: 'test@example.com',
  role: 'user'
};

test('renders user profile', () => {
  render(<UserProfile user={mockUser} />);
  expect(screen.getByText(mockUser.username)).toBeInTheDocument();
});
```

## Resources

- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
- [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/)
