-- Medical Clients Tables Migration

-- Main medical clients table
CREATE TABLE IF NOT EXISTS medical_clients (
    client_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    base_url VARCHAR(255),
    api_version VARCHAR(50),
    logo_url VARCHAR(255),
    documentation_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Medical client configuration table
CREATE TABLE IF NOT EXISTS medical_client_configs (
    config_id INT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    api_key VARCHAR(255),
    email VARCHAR(255),
    username VARCHAR(100),
    password VARCHAR(255),
    token VARCHAR(255),
    token_expiry TIMESTAMP NULL,
    rate_limit INT,
    rate_limit_period VARCHAR(20),
    timeout INT DEFAULT 30,
    retry_count INT DEFAULT 3,
    use_cache BOOLEAN DEFAULT TRUE,
    cache_ttl INT DEFAULT 3600,
    additional_config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES medical_clients(client_id) ON DELETE CASCADE
);

-- Medical client status table
CREATE TABLE IF NOT EXISTS medical_client_status (
    status_id INT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    status ENUM('connected', 'disconnected', 'error', 'unknown') DEFAULT 'unknown',
    response_time FLOAT,
    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (client_id) REFERENCES medical_clients(client_id) ON DELETE CASCADE
);

-- Medical client status history table
CREATE TABLE IF NOT EXISTS medical_client_status_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    status ENUM('connected', 'disconnected', 'error', 'unknown') NOT NULL,
    response_time FLOAT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (client_id) REFERENCES medical_clients(client_id) ON DELETE CASCADE
);

-- Medical client usage statistics table
CREATE TABLE IF NOT EXISTS medical_client_usage_stats (
    stat_id INT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    requests_count INT DEFAULT 0,
    successful_requests INT DEFAULT 0,
    failed_requests INT DEFAULT 0,
    cached_requests INT DEFAULT 0,
    total_response_time FLOAT DEFAULT 0,
    average_response_time FLOAT DEFAULT 0,
    FOREIGN KEY (client_id) REFERENCES medical_clients(client_id) ON DELETE CASCADE,
    UNIQUE KEY (client_id, date)
);

-- Insert default medical clients
INSERT INTO medical_clients (client_id, name, description, base_url, api_version, documentation_url)
VALUES
    ('ncbi', 'NCBI', 'National Center for Biotechnology Information', 'https://api.ncbi.nlm.nih.gov', '2.0', 'https://www.ncbi.nlm.nih.gov/home/develop/api/'),
    ('umls', 'UMLS', 'Unified Medical Language System', 'https://uts-ws.nlm.nih.gov/rest', '2.0', 'https://documentation.uts.nlm.nih.gov/rest/home.html'),
    ('clinical_trials', 'ClinicalTrials.gov', 'Clinical trials database', 'https://clinicaltrials.gov/api', '1.0', 'https://clinicaltrials.gov/api/gui'),
    ('cochrane', 'Cochrane Library', 'Systematic reviews database', 'https://www.cochranelibrary.com/api', '1.0', 'https://www.cochranelibrary.com/help/api'),
    ('crossref', 'Crossref', 'DOI registration agency', 'https://api.crossref.org', '1.0', 'https://github.com/CrossRef/rest-api-doc'),
    ('snomed', 'SNOMED CT', 'Clinical terminology', 'https://browser.ihtsdotools.org/snowstorm/snomed-ct/v2', '2.0', 'https://github.com/IHTSDO/snowstorm/wiki/Using-the-REST-API');

-- Insert default client configurations
INSERT INTO medical_client_configs (client_id, timeout, retry_count, use_cache, cache_ttl)
VALUES
    ('ncbi', 30, 3, TRUE, 3600),
    ('umls', 30, 3, TRUE, 3600),
    ('clinical_trials', 30, 3, TRUE, 3600),
    ('cochrane', 30, 3, TRUE, 3600),
    ('crossref', 30, 3, TRUE, 3600),
    ('snomed', 30, 3, TRUE, 3600);

-- Insert default client status
INSERT INTO medical_client_status (client_id, status)
VALUES
    ('ncbi', 'unknown'),
    ('umls', 'unknown'),
    ('clinical_trials', 'unknown'),
    ('cochrane', 'unknown'),
    ('crossref', 'unknown'),
    ('snomed', 'unknown');
