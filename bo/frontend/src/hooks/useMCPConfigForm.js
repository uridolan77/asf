import { useForm } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';

/**
 * Schema for MCP provider configuration validation
 */
const mcpConfigSchema = yup.object({
  provider_id: yup.string()
    .required('Provider ID is required')
    .matches(/^[a-z0-9_-]+$/, 'Provider ID must contain only lowercase letters, numbers, underscores, and hyphens')
    .min(3, 'Provider ID must be at least 3 characters')
    .max(50, 'Provider ID must be at most 50 characters'),
  display_name: yup.string()
    .required('Display name is required')
    .max(100, 'Display name must be at most 100 characters'),
  transport_type: yup.string()
    .required('Transport type is required')
    .oneOf(['stdio', 'grpc', 'http'], 'Transport type must be one of: stdio, grpc, http'),
  enable_streaming: yup.boolean(),
  timeout_seconds: yup.number()
    .required('Timeout is required')
    .positive('Timeout must be positive')
    .integer('Timeout must be an integer')
    .min(1, 'Timeout must be at least 1 second')
    .max(300, 'Timeout must be at most 300 seconds'),
  max_retries: yup.number()
    .required('Max retries is required')
    .integer('Max retries must be an integer')
    .min(0, 'Max retries must be at least 0')
    .max(10, 'Max retries must be at most 10'),
  enabled: yup.boolean(),
  transport_config: yup.object().shape({
    stdio: yup.object().when('$transport_type', {
      is: 'stdio',
      then: yup.object({
        command: yup.string().required('Command is required'),
        args: yup.array().of(yup.string()),
        env: yup.object(),
        cwd: yup.string().nullable()
      })
    }),
    grpc: yup.object().when('$transport_type', {
      is: 'grpc',
      then: yup.object({
        endpoint: yup.string().required('Endpoint is required'),
        use_tls: yup.boolean(),
        ca_cert: yup.string().nullable(),
        client_cert: yup.string().nullable(),
        client_key: yup.string().nullable(),
        metadata: yup.object()
      })
    }),
    http: yup.object().when('$transport_type', {
      is: 'http',
      then: yup.object({
        base_url: yup.string().required('Base URL is required').url('Base URL must be a valid URL'),
        headers: yup.object(),
        verify_ssl: yup.boolean()
      })
    })
  })
});

/**
 * Default values for MCP provider configuration
 */
const defaultValues = {
  provider_id: '',
  display_name: '',
  transport_type: 'stdio',
  enable_streaming: true,
  timeout_seconds: 60,
  max_retries: 3,
  enabled: true,
  transport_config: {
    stdio: {
      command: 'npx',
      args: ['@anthropic/mcp-starter', '--no-color'],
      env: {},
      cwd: null
    },
    grpc: {
      endpoint: 'localhost:50051',
      use_tls: false,
      ca_cert: null,
      client_cert: null,
      client_key: null,
      metadata: {}
    },
    http: {
      base_url: 'https://api.example.com/mcp',
      headers: {
        'Content-Type': 'application/json'
      },
      verify_ssl: true
    }
  }
};

/**
 * Custom hook for MCP configuration form
 * 
 * This hook provides form handling for MCP provider configuration
 * with validation and default values.
 */
export const useMCPConfigForm = (provider = null) => {
  // Create form with validation
  const form = useForm({
    resolver: yupResolver(mcpConfigSchema),
    defaultValues: provider || defaultValues,
    context: {
      transport_type: provider?.transport_type || 'stdio'
    }
  });

  // Watch transport type to update context
  const transportType = form.watch('transport_type');
  
  // Update context when transport type changes
  if (transportType !== form.formState.context.transport_type) {
    form.formState.context.transport_type = transportType;
  }

  return {
    form,
    defaultValues
  };
};
