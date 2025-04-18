/**
 * Initialize Mock Service Worker
 * 
 * This file is responsible for initializing MSW in development mode
 * when the VITE_USE_MOCK_DATA environment variable is set to 'true'.
 */

async function initMocks() {
  // Check if we should use mock data
  const useMockData = import.meta.env.VITE_USE_MOCK_DATA === 'true';
  
  if (useMockData && process.env.NODE_ENV !== 'production') {
    console.log('🔶 Mock Service Worker enabled');
    
    // Import the worker dynamically
    const { worker } = await import('./browser');
    
    // Start the worker
    await worker.start({
      onUnhandledRequest: 'bypass', // Don't warn about unhandled requests
    });
    
    return worker;
  }
  
  return null;
}

export default initMocks;
