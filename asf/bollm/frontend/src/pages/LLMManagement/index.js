// LLMManagement index.js - exports all LLM Management components
import Dashboard from './Dashboard';
import GatewayDashboard from './GatewayDashboard';
import DSPyDashboard from './DSPyDashboard';
import BiomedLMDashboard from './BiomedLMDashboard';
import CLPEFTDashboard from './CLPEFTDashboard';
import MCPDashboard from './MCPDashboard';
import UsageDashboard from './UsageDashboard';
import PlaceholderPage from './PlaceholderPage';

// Main LLM Management component (legacy tabbed interface)
import LLMManagement from './LLMManagement';

// Export all components
export {
  Dashboard,
  GatewayDashboard,
  DSPyDashboard,
  BiomedLMDashboard,
  CLPEFTDashboard,
  MCPDashboard,
  UsageDashboard,
  PlaceholderPage
};

// Use the PlaceholderPage for all not-yet-implemented pages
export const ProvidersListPage = (props) => <PlaceholderPage title="LLM Providers List" {...props} />;
export const ProviderConfigPage = (props) => <PlaceholderPage title="Provider Configuration" {...props} />;
export const ProviderKeysPage = (props) => <PlaceholderPage title="Provider API Keys" {...props} />;
export const ProviderHealthPage = (props) => <PlaceholderPage title="Provider Health Status" {...props} />;
export const ModelsListPage = (props) => <PlaceholderPage title="LLM Models List" {...props} />;
export const ModelPerformancePage = (props) => <PlaceholderPage title="Model Performance" {...props} />;
export const RequestLogsPage = (props) => <PlaceholderPage title="Request Logs" {...props} />;
export const RequestAnalysisPage = (props) => <PlaceholderPage title="Request Analysis" {...props} />;
export const ChatPlaygroundPage = (props) => <PlaceholderPage title="Chat Completion Playground" {...props} />;
export const CacheStatsPage = (props) => <PlaceholderPage title="Cache Statistics" {...props} />;
export const CacheConfigPage = (props) => <PlaceholderPage title="Cache Configuration" {...props} />;
export const SemanticCachePage = (props) => <PlaceholderPage title="Semantic Cache" {...props} />;
export const MetricsPage = (props) => <PlaceholderPage title="LLM Metrics" {...props} />;
export const TracesPage = (props) => <PlaceholderPage title="Request Traces" {...props} />;
export const AlertsPage = (props) => <PlaceholderPage title="Alerts" {...props} />;
export const PluginsPage = (props) => <PlaceholderPage title="LLM Plugins" {...props} />;
export const InterventionsPage = (props) => <PlaceholderPage title="LLM Interventions" {...props} />;
export const GatewayConfigPage = (props) => <PlaceholderPage title="Gateway Configuration" {...props} />;
export const APIKeysPage = (props) => <PlaceholderPage title="API Keys Management" {...props} />;
export const EnvironmentPage = (props) => <PlaceholderPage title="Environment Variables" {...props} />;

// Export the legacy LLMManagement component as default for backward compatibility
export default LLMManagement;
