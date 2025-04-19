@echo off
REM Script to run the backend server with all tracing and observability components disabled

REM Set environment variables to disable all observability components
set DISABLE_MCP_WEBSOCKET_TASKS=1
set DISABLE_PROMETHEUS=1
set DISABLE_TRACING=1
set DISABLE_METRICS=1
set DISABLE_OTLP=1
set DISABLE_OBSERVABILITY=1
set LOG_LEVEL=debug

REM Run the server
python run_simple.py
