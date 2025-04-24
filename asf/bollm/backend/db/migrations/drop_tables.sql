-- drop_tables.sql
-- Migration script to drop tables for API keys and configuration

-- Drop triggers first
DROP TRIGGER IF EXISTS api_key_usage_before_insert;

-- Drop tables in reverse order of creation (to handle foreign key constraints)
DROP TABLE IF EXISTS api_key_usage;
DROP TABLE IF EXISTS audit_logs;
DROP TABLE IF EXISTS user_settings;
DROP TABLE IF EXISTS configurations;
DROP TABLE IF EXISTS connection_parameters;
DROP TABLE IF EXISTS api_keys;
DROP TABLE IF EXISTS provider_models;
DROP TABLE IF EXISTS providers;
