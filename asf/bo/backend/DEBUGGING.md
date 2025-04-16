# Debugging Guide for BO Backend

This guide provides instructions for debugging the BO backend, with a focus on the LLM Gateway functionality.

## Running the Backend in Debug Mode

To run the backend with enhanced debugging:

```bash
cd asf\bo\backend
python run_debug.py
```

This will start the backend with DEBUG level logging and additional diagnostic information.

## Using the Diagnostic Tools

### Web-based Debugging

The backend includes several debug endpoints that can be accessed through the API:

1. **Configuration Inspection**:
   - `GET /api/llm/debug/config` - View the current LLM Gateway configuration

2. **Environment Information**:
   - `GET /api/llm/debug/environment` - View environment variables and system information

3. **OpenAI Connection Test**:
   - `POST /api/llm/debug/test-openai` - Test the OpenAI API connection
   - You can optionally provide an API key in the request body: `{"api_key": "your-api-key"}`

4. **Full Diagnostics**:
   - `GET /api/llm/debug/diagnostics` - Run a comprehensive diagnostic suite

5. **Log Viewing**:
   - `GET /api/llm/debug/logs` - View recent log entries
   - You can specify the number of lines: `GET /api/llm/debug/logs?lines=200`

### Command-line Diagnostic Tools

Several command-line tools are available for diagnosing issues:

1. **OpenAI Connection Test**:
   ```bash
   cd asf
   python -m medical.llm_gateway.test_openai_connection_detailed --api-key YOUR_API_KEY
   ```

2. **Full Diagnostics**:
   ```bash
   cd asf
   python -m medical.llm_gateway.diagnostic
   ```

3. **Simple OpenAI Test**:
   ```bash
   cd asf
   python -m medical.llm_gateway.test_openai_connection YOUR_API_KEY
   ```

## Logging Configuration

The backend uses Python's logging module with configuration in multiple places:

1. **Debug Configuration**:
   - `asf\bo\backend\debug_config.py` - Sets up DEBUG level logging

2. **Core Logging**:
   - `asf\medical\core\logging_config.py` - Main logging configuration

3. **Log Files**:
   - Application logs: `asf\medical\logs\app.log`
   - Error logs: `asf\medical\logs\error.log`

## Common Issues and Solutions

### OpenAI API Connection Issues

1. **API Key Problems**:
   - Check if the API key is set in the configuration file or environment variable
   - Verify the API key is valid and has not expired
   - Use the `/api/llm/debug/test-openai` endpoint to test the connection

2. **Network Issues**:
   - Check if your network can reach the OpenAI API
   - Verify proxy settings if applicable

3. **Rate Limiting**:
   - Check if you've hit OpenAI's rate limits
   - Look for "rate_limit" errors in the logs

### Configuration Issues

1. **Missing or Invalid Configuration**:
   - Check if the configuration file exists at `asf\bo\backend\config\llm\llm_gateway_config.yaml`
   - Verify the configuration format is valid YAML
   - Use the `/api/llm/debug/config` endpoint to inspect the configuration

2. **Provider Configuration**:
   - Ensure at least one provider is configured and enabled
   - Check that the provider type matches the implementation

### Environment Issues

1. **Missing Environment Variables**:
   - Check if required environment variables are set
   - Use the `/api/llm/debug/environment` endpoint to inspect environment variables

2. **Python Path Issues**:
   - Ensure the project root is in the Python path
   - Check for import errors in the logs

## Advanced Debugging

For more advanced debugging:

1. **Add Custom Logging**:
   - Add temporary logging statements to specific files
   - Use `logger.debug()` for detailed information

2. **Use Python Debugger**:
   - Add `import pdb; pdb.set_trace()` at specific points in the code
   - Run the backend with `python -m pdb run_debug.py`

3. **Inspect Network Traffic**:
   - Use tools like Wireshark or Fiddler to inspect API calls
   - Check for HTTPS issues or connection problems

## Getting Help

If you're still having issues:

1. Check the logs for specific error messages
2. Run the full diagnostics and review the results
3. Consult the API documentation for the LLM Gateway
4. Reach out to the development team with the diagnostic results
