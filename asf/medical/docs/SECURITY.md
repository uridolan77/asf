# ASF Medical Research Synthesizer Security Guide

This document provides security guidelines and best practices for the ASF Medical Research Synthesizer.

## Table of Contents

1. [Authentication and Authorization](#authentication-and-authorization)
2. [Rate Limiting](#rate-limiting)
3. [CSRF Protection](#csrf-protection)
4. [JWT Security](#jwt-security)
5. [Input Validation](#input-validation)
6. [Database Security](#database-security)
7. [API Security](#api-security)
8. [Dependency Management](#dependency-management)
9. [Logging and Monitoring](#logging-and-monitoring)
10. [Deployment Security](#deployment-security)
11. [Security Testing](#security-testing)

## Authentication and Authorization

### JWT Authentication

The ASF Medical Research Synthesizer uses JWT (JSON Web Tokens) for authentication. The system implements:

- **Access Tokens**: Short-lived tokens (default: 30 minutes) for API access
- **Refresh Tokens**: Longer-lived tokens (default: 7 days) for obtaining new access tokens
- **Token Validation**: Tokens are validated for expiration, signature, and token type
- **Role-Based Access Control**: Users have roles (e.g., "user", "admin") that determine their access rights

### Best Practices

1. **Always use the provided authentication dependencies**:
   - `get_current_user`: Ensures the user is authenticated
   - `get_current_active_user`: Ensures the user is authenticated and active
   - `get_admin_user`: Ensures the user is authenticated, active, and has admin role
   - `has_role(role)`: Ensures the user has a specific role
   - `has_any_role(roles)`: Ensures the user has any of the specified roles

2. **Protect sensitive endpoints**:
   - Use the admin middleware for admin-only routes
   - Add appropriate role checks to sensitive endpoints

3. **Never expose sensitive user information**:
   - Use the User model for API responses, which excludes password hashes
   - Avoid logging sensitive information

## Rate Limiting

The system implements rate limiting to prevent abuse and brute force attacks:

- **Login Rate Limiting**: Limits the number of login attempts per IP address
- **Burst Protection**: Allows a small burst of requests before rate limiting
- **IP-Based Blocking**: Blocks IP addresses after too many failed login attempts

### Best Practices

1. **Use the provided rate limiting middleware**:
   - `add_login_rate_limit_middleware`: Adds rate limiting to login endpoints

2. **Configure rate limits appropriately**:
   - Set lower limits for sensitive operations
   - Set higher limits for normal operations

3. **Monitor rate limit events**:
   - Watch for patterns of rate limit hits that might indicate attacks

## CSRF Protection

The system implements CSRF (Cross-Site Request Forgery) protection using the Double Submit Cookie pattern:

- **CSRF Token Cookie**: A secure, SameSite cookie containing a random token
- **CSRF Token Header**: The same token must be included in the X-CSRF-Token header for non-GET requests
- **Exempt Paths**: Some paths (e.g., login, API docs) are exempt from CSRF protection

### Best Practices

1. **Include the CSRF token in all forms and AJAX requests**:
   ```javascript
   // Example JavaScript for including CSRF token in AJAX requests
   const csrfToken = document.cookie
     .split('; ')
     .find(row => row.startsWith('csrf_token='))
     .split('=')[1];
     
   fetch('/api/endpoint', {
     method: 'POST',
     headers: {
       'Content-Type': 'application/json',
       'X-CSRF-Token': csrfToken
     },
     body: JSON.stringify(data)
   });
   ```

2. **Exempt only necessary paths**:
   - Only exempt paths that genuinely need to be exempt (e.g., login)
   - Document all exemptions and their justification

## JWT Security

The system implements several security measures for JWT tokens:

- **Short-Lived Access Tokens**: Access tokens expire after a short time (default: 30 minutes)
- **Refresh Token Rotation**: New refresh tokens are issued when refreshing access tokens
- **Token Type Validation**: Tokens are validated for their intended use (access vs. refresh)
- **Secure Signing**: Tokens are signed with a strong secret key using HS256 algorithm

### Best Practices

1. **Keep the SECRET_KEY secure**:
   - Use a strong, random secret key
   - Rotate the secret key periodically
   - Never commit the secret key to version control

2. **Handle token expiration gracefully**:
   - Implement automatic token refresh in client applications
   - Provide clear error messages for expired tokens

3. **Validate tokens properly**:
   - Always validate the token signature
   - Always check the token expiration
   - Always check the token type

## Input Validation

The system uses Pydantic models for input validation:

- **Request Models**: Define the expected structure and types of request data
- **Validation Rules**: Define validation rules for input data
- **Error Messages**: Provide clear error messages for validation failures

### Best Practices

1. **Use Pydantic models for all input data**:
   - Define models with appropriate types and validation rules
   - Use strict validation where appropriate

2. **Validate all user input**:
   - Never trust user input
   - Validate input data before using it

3. **Sanitize input data**:
   - Remove or escape potentially dangerous characters
   - Use parameterized queries for database operations

## Database Security

The system implements several database security measures:

- **ORM Usage**: Uses SQLAlchemy ORM to prevent SQL injection
- **Connection Pooling**: Uses connection pooling for efficient database connections
- **Parameterized Queries**: Uses parameterized queries for all database operations
- **Transaction Management**: Uses transactions for database operations

### Best Practices

1. **Use the provided repository classes**:
   - `EnhancedBaseRepository`: Provides a secure interface to database operations

2. **Use transactions for multi-step operations**:
   - Ensure data consistency
   - Roll back on errors

3. **Limit database permissions**:
   - Use a database user with only the necessary permissions
   - Use different users for different environments

## API Security

The system implements several API security measures:

- **Input Validation**: Validates all input data
- **Output Validation**: Validates all output data
- **Error Handling**: Provides clear error messages without exposing sensitive information
- **Rate Limiting**: Limits the rate of API requests
- **Authentication**: Requires authentication for sensitive operations

### Best Practices

1. **Document API endpoints**:
   - Use OpenAPI documentation
   - Document security requirements

2. **Use appropriate HTTP methods**:
   - GET for retrieving data
   - POST for creating data
   - PUT/PATCH for updating data
   - DELETE for deleting data

3. **Use appropriate HTTP status codes**:
   - 200 OK for successful operations
   - 201 Created for successful creation
   - 400 Bad Request for client errors
   - 401 Unauthorized for authentication errors
   - 403 Forbidden for authorization errors
   - 404 Not Found for missing resources
   - 429 Too Many Requests for rate limiting
   - 500 Internal Server Error for server errors

## Dependency Management

The system uses dependency management to ensure secure and up-to-date dependencies:

- **Pinned Dependencies**: Dependencies are pinned to specific versions
- **Dependency Scanning**: Dependencies are scanned for vulnerabilities
- **Dependency Updates**: Dependencies are updated regularly

### Best Practices

1. **Pin dependencies to specific versions**:
   - Use exact versions in requirements.txt
   - Use version ranges only when necessary

2. **Scan dependencies for vulnerabilities**:
   - Use tools like pip-audit or GitHub Dependabot
   - Update vulnerable dependencies promptly

3. **Update dependencies regularly**:
   - Keep dependencies up to date
   - Test thoroughly after updates

## Logging and Monitoring

The system implements comprehensive logging and monitoring:

- **Structured Logging**: Uses structured logging for easy analysis
- **Log Levels**: Uses appropriate log levels for different events
- **Log Rotation**: Rotates logs to prevent disk space issues
- **Monitoring**: Monitors system health and performance

### Best Practices

1. **Log security events**:
   - Authentication attempts (successful and failed)
   - Authorization failures
   - Rate limiting events
   - Suspicious activity

2. **Avoid logging sensitive information**:
   - Never log passwords or tokens
   - Mask sensitive information in logs

3. **Monitor logs for security events**:
   - Set up alerts for suspicious activity
   - Review logs regularly

## Deployment Security

The system is designed to be deployed securely:

- **HTTPS**: Uses HTTPS for all communication
- **CORS**: Configures CORS to restrict cross-origin requests
- **Environment Variables**: Uses environment variables for configuration
- **Containerization**: Can be deployed in containers for isolation

### Best Practices

1. **Use HTTPS in production**:
   - Obtain and configure SSL/TLS certificates
   - Use HTTP Strict Transport Security (HSTS)
   - Redirect HTTP to HTTPS

2. **Configure CORS properly**:
   - Restrict allowed origins to trusted domains
   - Use appropriate CORS headers

3. **Secure environment variables**:
   - Use a secure method for managing environment variables
   - Never commit environment variables to version control

## Security Testing

The system includes security tests to ensure security measures are working correctly:

- **Authentication Tests**: Test authentication functionality
- **Authorization Tests**: Test authorization functionality
- **Rate Limiting Tests**: Test rate limiting functionality
- **CSRF Protection Tests**: Test CSRF protection functionality
- **Input Validation Tests**: Test input validation functionality

### Best Practices

1. **Run security tests regularly**:
   - Include security tests in CI/CD pipeline
   - Run security tests before deployment

2. **Perform security audits**:
   - Conduct regular security audits
   - Address security issues promptly

3. **Stay informed about security best practices**:
   - Follow security blogs and newsletters
   - Attend security conferences and webinars
