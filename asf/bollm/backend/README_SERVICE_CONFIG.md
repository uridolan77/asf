# LLM Service Configurations

This module provides a complete system for managing LLM service configurations, allowing you to create, edit, apply, and delete configurations with all the advanced features of the LLM Gateway.

## Features

- **Configuration Management**: Create, read, update, and delete service configurations
- **Feature Toggles**: Enable/disable specific features like caching, resilience, observability, events, and progress tracking
- **Detailed Settings**: Configure detailed settings for each feature
- **Configuration Sharing**: Share configurations with other users
- **Apply Configurations**: Apply configurations to the service with a single click

## Database Models

The service configurations are stored in the following database tables:

- `service_configurations`: Main configuration table
- `caching_configurations`: Caching-specific settings
- `resilience_configurations`: Resilience-specific settings
- `observability_configurations`: Observability-specific settings
- `events_configurations`: Events-specific settings
- `progress_tracking_configurations`: Progress tracking-specific settings

## API Endpoints

The service configurations API provides the following endpoints:

- `POST /api/llm/service/configurations`: Create a new configuration
- `GET /api/llm/service/configurations`: List configurations
- `GET /api/llm/service/configurations/{config_id}`: Get a specific configuration
- `PUT /api/llm/service/configurations/{config_id}`: Update a configuration
- `DELETE /api/llm/service/configurations/{config_id}`: Delete a configuration
- `POST /api/llm/service/configurations/{config_id}/apply`: Apply a configuration

## Setup and Usage

### Database Configuration

The service configurations are stored in a MySQL database. The default connection settings are:

```
DB_USER = 'root'
DB_PASSWORD = 'Dt%g_9W3z0*!I'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'bo_admin'
```

You can override these settings by setting the following environment variables:
- `BO_DB_USER`: Database username
- `BO_DB_PASSWORD`: Database password
- `BO_DB_HOST`: Database host
- `BO_DB_PORT`: Database port
- `BO_DB_NAME`: Database name

### Initialize the Database

There are two ways to initialize the database tables for service configurations:

#### Option 1: Using the Python ORM-based Initialization

```bash
python -m asf.bollm.backend.init_service_config_db
```

This script will:
1. Connect to the MySQL database
2. Check which tables need to be created
3. Create only the tables that don't exist yet using SQLAlchemy ORM
4. Report the status of each table

#### Option 2: Using the SQL Script (Recommended)

For a more comprehensive setup that includes indexes and default data:

```bash
python -m asf.bollm.backend.scripts.setup_service_config_db
```

This script will:
1. Connect to the MySQL database
2. Create the database if it doesn't exist
3. Execute the SQL script to create all tables with proper indexes
4. Insert default configuration data

The SQL script (`create_service_config_tables.sql`) creates:
- All required tables with proper relationships
- Indexes for better query performance
- A default configuration with settings for all features

### Run the Server

To run the server with the service configurations API enabled, run:

```bash
python -m asf.bollm.backend.run_service_config_server
```

This will start the server on port 8000 by default. You can change the port by setting the `SERVICE_CONFIG_PORT` environment variable.

### Troubleshooting Database Issues

If you encounter database connection errors like:

```
Access denied for user 'root'@'localhost' (using password: YES)
```

Make sure:
1. MySQL is running
2. The user and password in the connection string are correct
3. The database `bo_admin` exists
4. The user has permissions to create tables in the database

You can create the database if it doesn't exist:

```sql
CREATE DATABASE IF NOT EXISTS bo_admin;
```

### Frontend Integration

The frontend provides a complete UI for managing service configurations:

1. **Configurations List Page**: View all configurations, create new ones, and perform actions like edit, delete, and apply
2. **Configuration Edit Page**: Edit a specific configuration with detailed settings for each feature

## Troubleshooting

If you encounter CORS errors or 404 Not Found errors, make sure:

1. The server is running on the correct port (8000 by default)
2. The database tables have been created
3. The API endpoints are properly registered
4. CORS is properly configured to allow requests from your frontend origin

## Development

### Adding New Features

To add a new feature to the service configurations:

1. Add the feature toggle to the `ServiceConfiguration` model
2. Create a new configuration model for the feature
3. Add the feature to the Pydantic models
4. Update the API endpoints to handle the new feature
5. Update the frontend to display and edit the new feature

### Testing

To test the API endpoints, you can use the Swagger UI at `http://localhost:8000/docs` when the server is running.
