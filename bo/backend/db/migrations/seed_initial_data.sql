-- seed_initial_data.sql
-- Migration script to seed initial data from YAML configuration

-- Insert OpenAI provider
INSERT INTO providers (provider_id, display_name, provider_type, description, enabled)
VALUES ('openai_gpt4_default', 'OpenAI GPT-4', 'openai', 'OpenAI GPT-4 and GPT-3.5 provider', TRUE);

-- Insert OpenAI models
INSERT INTO provider_models (model_id, provider_id, display_name, model_type, context_window, max_tokens, enabled)
VALUES 
('gpt-3.5-turbo', 'openai_gpt4_default', 'GPT-3.5 Turbo', 'chat', 4096, 4096, TRUE),
('gpt-4', 'openai_gpt4_default', 'GPT-4', 'chat', 8192, 4096, TRUE);

-- Insert connection parameters (non-sensitive)
INSERT INTO connection_parameters (provider_id, param_name, param_value, is_sensitive, environment)
VALUES 
('openai_gpt4_default', 'base_url', '', FALSE, 'development'),
('openai_gpt4_default', 'api_key_env_var', 'OPENAI_API_KEY', FALSE, 'development');

-- Insert global configurations
INSERT INTO configurations (config_key, config_value, config_type, description, environment)
VALUES 
('llm_gateway.batch_concurrency_limit', '8', 'integer', 'Maximum number of concurrent LLM requests', 'development'),
('llm_gateway.default_provider', 'openai_gpt4_default', 'string', 'Default LLM provider', 'development'),
('llm_gateway.default_model', 'gpt-3.5-turbo', 'string', 'Default LLM model', 'development');
