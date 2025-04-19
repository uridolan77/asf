@echo off
REM Script to run the backend server without the LLM Gateway module

REM Set environment variables to disable all observability components
set DISABLE_MCP_WEBSOCKET_TASKS=1
set DISABLE_PROMETHEUS=1
set DISABLE_TRACING=1
set DISABLE_METRICS=1
set DISABLE_OTLP=1
set DISABLE_OBSERVABILITY=1
set LOG_LEVEL=debug

REM Disable LLM Gateway
set DISABLE_LLM_GATEWAY=1

REM Run the server
python run_no_llm.py
