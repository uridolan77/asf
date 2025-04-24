// Store for feature toggles and app configuration
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface FeatureFlag {
  id: string;
  name: string;
  enabled: boolean;
  description: string;
}

interface AppConfigState {
  // Feature toggles
  featureFlags: FeatureFlag[];

  // Theme settings
  darkMode: boolean;

  // User preferences
  sidebarCollapsed: boolean;

  // Actions
  toggleFeature: (featureId: string) => void;
  setDarkMode: (enabled: boolean) => void;
  toggleSidebar: () => void;

  // Batch update feature flags
  updateFeatureFlags: (flags: FeatureFlag[]) => void;
}

export const useAppConfigStore = create<AppConfigState>()(
  persist(
    (set) => ({
      featureFlags: [
        { id: 'advanced-ml', name: 'Advanced ML Features', enabled: false, description: 'Enable advanced machine learning features' },
        { id: 'beta-visualizer', name: 'Beta Claim Visualizer', enabled: false, description: 'Use the beta version of the claim visualizer' },
        { id: 'export-formats', name: 'Advanced Export Formats', enabled: true, description: 'Enable additional export formats' },
        { id: 'reporting-feature', name: 'Reporting System', enabled: true, description: 'Enable the reporting system' },
      ],

      darkMode: window.matchMedia('(prefers-color-scheme: dark)').matches,
      sidebarCollapsed: false,

      toggleFeature: (featureId) =>
        set((state) => ({
          featureFlags: state.featureFlags.map(flag =>
            flag.id === featureId ? { ...flag, enabled: !flag.enabled } : flag
          )
        })),

      setDarkMode: (enabled) => set({ darkMode: enabled }),

      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

      updateFeatureFlags: (flags) => set({ featureFlags: flags }),
    }),
    {
      name: 'app-config-storage',
      partialize: (state) => ({
        darkMode: state.darkMode,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    }
  )
);