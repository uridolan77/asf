@echo off
REM Script to run the backend server with LLM Gateway enabled but observability disabled

REM Set environment variables to disable observability components
set DISABLE_MCP_WEBSOCKET_TASKS=1
set DISABLE_PROMETHEUS=1
set DISABLE_TRACING=1
set DISABLE_METRICS=1
set DISABLE_OTLP=1
set DISABLE_OBSERVABILITY=1
set LOG_LEVEL=debug

REM Enable LLM Gateway (by not setting DISABLE_LLM_GATEWAY)
set DISABLE_LLM_GATEWAY=

REM Run the server
python run_with_llm.py
