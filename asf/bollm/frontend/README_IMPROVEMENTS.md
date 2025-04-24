# Frontend Improvements

This document outlines the improvements made to the frontend codebase to enhance maintainability, user experience, and performance.

## Table of Contents

1. [Theme Provider](#theme-provider)
2. [Loading States](#loading-states)
3. [Error Handling](#error-handling)
4. [Animations](#animations)
5. [API Service](#api-service)
6. [Usage Examples](#usage-examples)

## Theme Provider

A centralized theme provider has been implemented to ensure consistent styling across the application.

### Key Features

- **Centralized Theme Configuration**: All theme settings are defined in a single file (`src/theme/index.js`)
- **Consistent Color Palette**: Predefined color palette for primary, secondary, success, error, warning, and info colors
- **Typography System**: Standardized typography settings for all text elements
- **Component Styling**: Global styling overrides for Material UI components
- **Responsive Design**: Theme settings that work across different screen sizes

### Implementation

The theme provider is implemented in `App.js`:

```jsx
import { ThemeProvider, CssBaseline } from '@mui/material';
import theme from './theme';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {/* Application content */}
    </ThemeProvider>
  );
}
```

## Loading States

Consistent loading states have been implemented for all asynchronous operations.

### Key Features

- **Reusable Loading Components**: A set of loading components for different contexts (`src/components/UI/LoadingIndicators.js`)
- **Consistent API**: All loading components follow a consistent API
- **Context-Aware**: Different loading indicators for different contexts (full page, content area, buttons, etc.)
- **Customizable**: All loading components accept customization props

### Loading Components

- `FullPageLoader`: For full-page loading states
- `ContentLoader`: For loading content within a specific area
- `ButtonLoader`: For loading states within buttons
- `CardSkeleton`: For loading card content
- `TableSkeleton`: For loading table content
- `TopProgressBar`: For indicating background loading
- `FormFieldSkeleton`: For loading form fields

### Usage Example

```jsx
import { ButtonLoader } from '../UI/LoadingIndicators';

<Button
  variant="contained"
  color="primary"
  startIcon={isLoading ? <ButtonLoader size={20} /> : <SearchIcon />}
  disabled={isLoading}
>
  {isLoading ? 'Searching...' : 'Search'}
</Button>
```

## Error Handling

A centralized error handling system has been implemented to ensure consistent error handling across the application.

### Key Features

- **Centralized Error Handling**: All error handling logic is defined in a single file (`src/utils/errorHandler.js`)
- **Consistent Error Messages**: Standardized error messages for common error scenarios
- **Error Logging**: Centralized error logging
- **Authentication Error Handling**: Special handling for authentication errors

### Error Handling Utilities

- `getErrorMessage`: Extract a user-friendly error message from various error types
- `handleAuthError`: Handle authentication errors (401, 403)
- `logError`: Log errors to console (can be extended to log to a service)
- `createErrorObject`: Create a standardized error object

### Usage Example

```jsx
import { getErrorMessage } from '../utils/errorHandler';

try {
  // API call
} catch (error) {
  const errorMessage = getErrorMessage(error);
  showError(errorMessage);
}
```

## Animations

Subtle animations have been added to enhance the user experience.

### Key Features

- **Reusable Animation Components**: A set of animation components for different contexts (`src/components/UI/Animations.js`)
- **Consistent API**: All animation components follow a consistent API
- **Performance Optimized**: Animations are optimized for performance
- **Customizable**: All animation components accept customization props

### Animation Components

- `FadeIn`: Fade in animation
- `GrowIn`: Grow animation
- `SlideIn`: Slide in animation
- `ZoomIn`: Zoom in animation
- `StaggeredList`: Staggered animation for lists
- `PageTransition`: Animated page transition
- `HoverAnimation`: Hover animation
- `AnimatedCounter`: Animated counter

### Usage Example

```jsx
import { FadeIn, StaggeredList } from '../UI/Animations';

<FadeIn>
  <StaggeredList>
    {items.map(item => (
      <ListItem key={item.id}>{item.name}</ListItem>
    ))}
  </StaggeredList>
</FadeIn>
```

## API Service

A centralized API service has been implemented to ensure consistent API calls across the application.

### Key Features

- **Centralized API Configuration**: All API configuration is defined in a single file (`src/services/api.js`)
- **Consistent Error Handling**: Standardized error handling for all API calls
- **Authentication**: Automatic token handling for authenticated requests
- **Organized Endpoints**: API endpoints are organized by domain

### API Service Methods

- `auth`: Authentication endpoints (login, logout, me)
- `users`: User management endpoints (getAll, getById, create, update, delete)
- `medical`: Medical search endpoints (picoSearch, advancedSearch, terminologySearch)
- `knowledgeBase`: Knowledge base endpoints (getAll, getById, create, update, delete)

### Custom Hook

A custom hook (`useApi`) has been created to simplify API calls:

- **Loading State**: Automatic loading state management
- **Error State**: Automatic error state management
- **Success Handling**: Callback for successful API calls
- **Error Handling**: Callback for failed API calls
- **Authentication Error Handling**: Special handling for authentication errors

### Usage Example

```jsx
import useApi from '../hooks/useApi';
import apiService from '../services/api';

const { data, loading, error, execute } = useApi(apiService.medical.picoSearch, {
  loadOnMount: false,
  onSuccess: (data) => {
    // Handle success
  },
  onError: (error) => {
    // Handle error
  }
});

// Execute the API call
const handleSearch = () => {
  execute(searchParams);
};
```

## Notification System

A centralized notification system has been implemented to ensure consistent notifications across the application.

### Key Features

- **Centralized Notification Management**: All notification logic is defined in a single file (`src/context/NotificationContext.js`)
- **Consistent Notification Styling**: Standardized notification styling
- **Multiple Notification Types**: Support for success, error, info, and warning notifications
- **Automatic Dismissal**: Notifications are automatically dismissed after a configurable duration

### Notification Methods

- `showSuccess`: Show a success notification
- `showError`: Show an error notification
- `showInfo`: Show an info notification
- `showWarning`: Show a warning notification

### Usage Example

```jsx
import { useNotification } from '../context/NotificationContext';

const { showSuccess, showError } = useNotification();

// Show a success notification
showSuccess('Operation completed successfully');

// Show an error notification
showError('An error occurred');
```

## Usage Examples

### Theme Provider

```jsx
// In a component
import { useTheme } from '@mui/material/styles';

const MyComponent = () => {
  const theme = useTheme();
  
  return (
    <div style={{ color: theme.palette.primary.main }}>
      Themed content
    </div>
  );
};
```

### Loading States

```jsx
// Full page loader
import { FullPageLoader } from '../UI/LoadingIndicators';

<FullPageLoader open={isLoading} message="Loading data..." />

// Content loader
import { ContentLoader } from '../UI/LoadingIndicators';

{isLoading ? (
  <ContentLoader height={400} message="Loading content..." />
) : (
  <Content data={data} />
)}

// Button loader
import { ButtonLoader } from '../UI/LoadingIndicators';

<Button
  variant="contained"
  startIcon={isLoading ? <ButtonLoader size={20} /> : <SaveIcon />}
  disabled={isLoading}
>
  {isLoading ? 'Saving...' : 'Save'}
</Button>
```

### Error Handling

```jsx
// Using the error handler directly
import { getErrorMessage } from '../utils/errorHandler';

try {
  // API call
} catch (error) {
  const errorMessage = getErrorMessage(error);
  showError(errorMessage);
}

// Using the API service with error handling
import apiService from '../services/api';

const result = await apiService.medical.picoSearch(searchParams);
if (!result.success) {
  showError(result.error);
}
```

### Animations

```jsx
// Fade in animation
import { FadeIn } from '../UI/Animations';

<FadeIn>
  <Content />
</FadeIn>

// Staggered list animation
import { StaggeredList } from '../UI/Animations';

<StaggeredList staggerDelay={100}>
  {items.map(item => (
    <ListItem key={item.id}>{item.name}</ListItem>
  ))}
</StaggeredList>

// Hover animation
import { HoverAnimation } from '../UI/Animations';

<HoverAnimation>
  <Card>
    Card content
  </Card>
</HoverAnimation>
```

### API Service

```jsx
// Using the API service directly
import apiService from '../services/api';

const handleSearch = async () => {
  setLoading(true);
  const result = await apiService.medical.picoSearch(searchParams);
  setLoading(false);
  
  if (result.success) {
    setData(result.data);
  } else {
    showError(result.error);
  }
};

// Using the useApi hook
import useApi from '../hooks/useApi';
import apiService from '../services/api';

const { data, loading, error, execute } = useApi(apiService.medical.picoSearch, {
  onSuccess: (data) => {
    setData(data);
  },
  onError: (error) => {
    showError(error);
  }
});

const handleSearch = () => {
  execute(searchParams);
};
```

### Notification System

```jsx
// Using the notification context
import { useNotification } from '../context/NotificationContext';

const { showSuccess, showError, showInfo, showWarning } = useNotification();

// Show a success notification
showSuccess('Operation completed successfully');

// Show an error notification
showError('An error occurred');

// Show an info notification
showInfo('Information message');

// Show a warning notification
showWarning('Warning message');
```
