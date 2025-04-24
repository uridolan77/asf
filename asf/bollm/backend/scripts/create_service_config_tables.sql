-- SQL Script to create all tables for the LLM Service Configuration system
-- This script creates a comprehensive set of tables for storing service configurations

-- Drop tables if they exist (in reverse order of dependencies)
DROP TABLE IF EXISTS progress_tracking_configurations;
DROP TABLE IF EXISTS events_configurations;
DROP TABLE IF EXISTS observability_configurations;
DROP TABLE IF EXISTS resilience_configurations;
DROP TABLE IF EXISTS caching_configurations;
DROP TABLE IF EXISTS service_configurations;

-- Create the main service_configurations table
CREATE TABLE service_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    service_id VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Feature toggles
    enable_caching BOOLEAN DEFAULT TRUE,
    enable_resilience BOOLEAN DEFAULT TRUE,
    enable_observability BOOLEAN DEFAULT TRUE,
    enable_events BOOLEAN DEFAULT TRUE,
    enable_progress_tracking BOOLEAN DEFAULT TRUE,
    
    -- User ownership and sharing
    created_by_user_id INT NOT NULL,
    is_public BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE KEY uix_service_config (service_id, name, created_by_user_id)
);

-- Create the caching_configurations table
CREATE TABLE caching_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    service_config_id INT NOT NULL,
    
    -- Caching settings
    similarity_threshold FLOAT DEFAULT 0.92,
    max_entries INT DEFAULT 10000,
    ttl_seconds INT DEFAULT 3600,
    persistence_type VARCHAR(20) DEFAULT 'disk',
    persistence_config JSON,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (service_config_id) REFERENCES service_configurations(id) ON DELETE CASCADE
);

-- Create the resilience_configurations table
CREATE TABLE resilience_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    service_config_id INT NOT NULL,
    
    -- Resilience settings
    max_retries INT DEFAULT 3,
    retry_delay FLOAT DEFAULT 1.0,
    backoff_factor FLOAT DEFAULT 2.0,
    circuit_breaker_failure_threshold INT DEFAULT 5,
    circuit_breaker_reset_timeout INT DEFAULT 30,
    timeout_seconds FLOAT DEFAULT 30.0,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (service_config_id) REFERENCES service_configurations(id) ON DELETE CASCADE
);

-- Create the observability_configurations table
CREATE TABLE observability_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    service_config_id INT NOT NULL,
    
    -- Observability settings
    metrics_enabled BOOLEAN DEFAULT TRUE,
    tracing_enabled BOOLEAN DEFAULT TRUE,
    logging_level VARCHAR(10) DEFAULT 'INFO',
    export_metrics BOOLEAN DEFAULT FALSE,
    metrics_export_url VARCHAR(255),
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (service_config_id) REFERENCES service_configurations(id) ON DELETE CASCADE
);

-- Create the events_configurations table
CREATE TABLE events_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    service_config_id INT NOT NULL,
    
    -- Events settings
    max_event_history INT DEFAULT 100,
    publish_to_external BOOLEAN DEFAULT FALSE,
    external_event_url VARCHAR(255),
    event_types_filter JSON,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (service_config_id) REFERENCES service_configurations(id) ON DELETE CASCADE
);

-- Create the progress_tracking_configurations table
CREATE TABLE progress_tracking_configurations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    service_config_id INT NOT NULL,
    
    -- Progress tracking settings
    max_active_operations INT DEFAULT 100,
    operation_ttl_seconds INT DEFAULT 3600,
    publish_updates BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign key
    FOREIGN KEY (service_config_id) REFERENCES service_configurations(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX idx_service_configurations_service_id ON service_configurations(service_id);
CREATE INDEX idx_service_configurations_created_by ON service_configurations(created_by_user_id);
CREATE INDEX idx_service_configurations_public ON service_configurations(is_public);
CREATE INDEX idx_caching_configurations_service_config ON caching_configurations(service_config_id);
CREATE INDEX idx_resilience_configurations_service_config ON resilience_configurations(service_config_id);
CREATE INDEX idx_observability_configurations_service_config ON observability_configurations(service_config_id);
CREATE INDEX idx_events_configurations_service_config ON events_configurations(service_config_id);
CREATE INDEX idx_progress_tracking_configurations_service_config ON progress_tracking_configurations(service_config_id);

-- Insert a default configuration
INSERT INTO service_configurations (
    service_id, 
    name, 
    description, 
    enable_caching, 
    enable_resilience, 
    enable_observability, 
    enable_events, 
    enable_progress_tracking, 
    created_by_user_id, 
    is_public
) VALUES (
    'enhanced_llm_service',
    'Default Configuration',
    'Default configuration for the LLM service',
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    1,  -- Assuming user ID 1 exists
    TRUE
);

-- Get the ID of the inserted configuration
SET @config_id = LAST_INSERT_ID();

-- Insert default caching configuration
INSERT INTO caching_configurations (
    service_config_id,
    similarity_threshold,
    max_entries,
    ttl_seconds,
    persistence_type
) VALUES (
    @config_id,
    0.92,
    10000,
    3600,
    'disk'
);

-- Insert default resilience configuration
INSERT INTO resilience_configurations (
    service_config_id,
    max_retries,
    retry_delay,
    backoff_factor,
    circuit_breaker_failure_threshold,
    circuit_breaker_reset_timeout,
    timeout_seconds
) VALUES (
    @config_id,
    3,
    1.0,
    2.0,
    5,
    30,
    30.0
);

-- Insert default observability configuration
INSERT INTO observability_configurations (
    service_config_id,
    metrics_enabled,
    tracing_enabled,
    logging_level,
    export_metrics
) VALUES (
    @config_id,
    TRUE,
    TRUE,
    'INFO',
    FALSE
);

-- Insert default events configuration
INSERT INTO events_configurations (
    service_config_id,
    max_event_history,
    publish_to_external
) VALUES (
    @config_id,
    100,
    FALSE
);

-- Insert default progress tracking configuration
INSERT INTO progress_tracking_configurations (
    service_config_id,
    max_active_operations,
    operation_ttl_seconds,
    publish_updates
) VALUES (
    @config_id,
    100,
    3600,
    TRUE
);
