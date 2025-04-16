# Frontend Enhancement Installation Guide

This guide provides the commands needed to install all dependencies for the frontend enhancements including TypeScript, React Query, Zustand, and Cypress for E2E testing.

## Prerequisites

- Node.js (version 16 or higher)
- npm (version 7 or higher)

## Installation Steps

1. Navigate to the frontend directory:

```bash
cd c:\code\asf\asf\bo\frontend
```

2. Install all dependencies:

```bash
npm install
```

This will install all dependencies defined in the package.json file, including:
- TypeScript and type definitions
- Vite and related plugins
- React Query
- Zustand
- Cypress for E2E testing
- ESLint and related plugins

## Alternative: Install Dependencies Individually

If you prefer to install dependencies one by one or are having trouble with the package.json update, you can run these commands:

### Core Dependencies (TypeScript, Vite)

```bash
npm install -D typescript @types/react @types/react-dom @types/node
npm install -D vite @vitejs/plugin-react
```

### React Query

```bash
npm install @tanstack/react-query @tanstack/react-query-devtools
```

### Zustand State Management

```bash
npm install zustand
```

### Form Handling (already in your package.json)

```bash
npm install react-hook-form @hookform/resolvers yup
```

### E2E Testing with Cypress

```bash
npm install -D cypress
```

### ESLint for TypeScript

```bash
npm install -D eslint @typescript-eslint/eslint-plugin @typescript-eslint/parser eslint-plugin-react-hooks eslint-plugin-react-refresh
```

### Testing Library

```bash
npm install -D vitest
```

## Running the Application

After installation, you can:

1. Start the development server:
```bash
npm run dev
```

2. Run tests:
```bash
npm run test
```

3. Run E2E tests:
```bash
npm run test:e2e
```

4. Open Cypress test runner:
```bash
npm run test:e2e:open
```

5. Build for production:
```bash
npm run build
```

## Troubleshooting

If you encounter any issues during installation:

1. Clear npm cache:
```bash
npm cache clean --force
```

2. Delete node_modules folder and package-lock.json:
```bash
rm -rf node_modules package-lock.json
```

3. Reinstall dependencies:
```bash
npm install
```