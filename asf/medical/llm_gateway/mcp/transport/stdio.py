"""
StdIO transport implementation for MCP communication.

This module provides a transport that launches and communicates with
an MCP server process using standard input/output streams.
"""

import asyncio
import os
import shlex
import signal
import shutil
import sys
from asyncio.subprocess import Process
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import structlog

from asf.medical.llm_gateway.transport.base import BaseTransport, ConnectionError

logger = structlog.get_logger("mcp_transport.stdio")


class StdioTransport(BaseTransport):
    """
    Transport implementation that communicates with an MCP server
    via stdin/stdout of a subprocess.
    
    Features:
    - Process lifecycle management
    - Signal handling (graceful termination)
    - Timeouts and health monitoring
    - Environment variable management for auth
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StdIO transport.
        
        Args:
            config: Configuration dictionary with keys:
                - command: Command to run
                - args: List of command arguments
                - env: Environment variables for the process
                - cwd: Working directory (optional)
                - timeout_seconds: Process startup timeout
                - termination_timeout: Seconds to wait for graceful termination
                - process_poll_interval: Seconds between process health checks
                - api_key_env_var: Environment variable name for API key
        """
        super().__init__(config)
        
        # Extract configuration
        self.command = config.get("command")
        self.args = config.get("args", [])
        self.env = config.get("env", {})
        self.cwd = config.get("cwd")
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.termination_timeout = config.get("termination_timeout", 5)
        self.process_poll_interval = config.get("process_poll_interval", 10)
        self.api_key_env_var = config.get("api_key_env_var")
        
        # Process state
        self.process: Optional[Process] = None
        self.last_heartbeat = datetime.utcnow()
        self._monitor_task: Optional[asyncio.Task] = None
        
        self.logger = logger.bind(
            command=self.command,
            transport_type="stdio"
        )
    
    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[Tuple[asyncio.StreamReader, asyncio.StreamWriter], None]:
        """
        Start the MCP server process and establish stdin/stdout streams.
        
        Returns:
            Tuple of (reader, writer) for the process
        
        Raises:
            ConnectionError: If the process fails to start
            TimeoutError: If the process fails to start within timeout
        """
        process = await self._start_process()
        
        try:
            # Start monitoring task if not already running
            if self._monitor_task is None or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(self._monitor_process())
            
            # Update heartbeat timestamp
            self.last_heartbeat = datetime.utcnow()
            
            # Yield the streams to the caller
            yield process.stdout, process.stdin
        
        finally:
            # Process will be closed in _monitor_process if it ends abnormally
            # For normal operation, we keep the process running for reuse
            pass
    
    async def _start_process(self) -> Process:
        """
        Start the MCP server process.
        
        Returns:
            The Process object
        
        Raises:
            ConnectionError: If process fails to start
        """
        if self.process is not None and self.process.returncode is None:
            # Process is still running, reuse it
            self.logger.debug("Reusing existing MCP process")
            return self.process
        
        # Resolve the command path
        cmd_path = shutil.which(self.command)
        if cmd_path is None:
            raise ConnectionError(
                f"Command not found: {self.command}",
                transport_type="stdio"
            )
        
        # Prepare environment variables
        env = os.environ.copy()
        env.update(self.env)
        env.update(self._get_auth_env())
        
        # Log startup
        safe_env = {k: "***" if k.lower().endswith("key") or k.lower().endswith("token") else v 
                   for k, v in env.items() if k not in os.environ}
        self.logger.info(
            "Starting MCP server process",
            command=cmd_path,
            args=self.args,
            env=safe_env,
            cwd=self.cwd or os.getcwd()
        )
        
        try:
            # Start the process asynchronously
            self.process = await asyncio.create_subprocess_exec(
                cmd_path,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.cwd
            )
            
            # Start logging stderr in the background
            asyncio.create_task(self._log_stderr())
            
            self.logger.info(
                "MCP server process started",
                pid=self.process.pid
            )
            
            return self.process
        
        except Exception as e:
            self.logger.error(
                "Failed to start MCP server process",
                error=str(e),
                exc_info=True
            )
            raise ConnectionError(
                f"Failed to start process: {str(e)}",
                transport_type="stdio",
                original_error=e
            )
    
    async def _log_stderr(self) -> None:
        """Log the stderr output from the process."""
        if self.process is None or self.process.stderr is None:
            return
        
        while True:
            try:
                line = await self.process.stderr.readline()
                if not line:
                    break
                
                stderr_line = line.decode('utf-8', errors='replace').rstrip()
                self.logger.debug(
                    "MCP server stderr",
                    pid=self.process.pid,
                    output=stderr_line
                )
            except Exception as e:
                self.logger.warning(
                    "Error reading MCP server stderr",
                    pid=self.process.pid if self.process else None,
                    error=str(e)
                )
                break
    
    async def _monitor_process(self) -> None:
        """
        Monitor the health of the process and restart it if needed.
        
        This runs as a background task to ensure the process stays alive.
        """
        while True:
            try:
                await asyncio.sleep(self.process_poll_interval)
                
                # Check if process is still running
                if self.process is None or self.process.returncode is not None:
                    self.logger.warning(
                        "MCP server process terminated unexpectedly",
                        returncode=self.process.returncode if self.process else None
                    )
                    # Process will be restarted on next connect() call
                    break
                
                # Check for inactivity timeout
                inactivity_time = (datetime.utcnow() - self.last_heartbeat).total_seconds()
                max_inactivity = self.config.get("max_process_inactivity_seconds", 3600)  # Default: 1 hour
                
                if inactivity_time > max_inactivity:
                    self.logger.warning(
                        "MCP server process inactive, terminating",
                        pid=self.process.pid,
                        inactivity_seconds=inactivity_time
                    )
                    await self.terminate_process()
                    break
            
            except asyncio.CancelledError:
                self.logger.info(
                    "Process monitor task cancelled",
                    pid=self.process.pid if self.process else None
                )
                break
            
            except Exception as e:
                self.logger.error(
                    "Error in process monitor task",
                    pid=self.process.pid if self.process else None,
                    error=str(e),
                    exc_info=True
                )
    
    async def terminate_process(self) -> None:
        """Gracefully terminate the process."""
        if self.process is None or self.process.returncode is not None:
            return
        
        self.logger.info(
            "Terminating MCP server process",
            pid=self.process.pid
        )
        
        try:
            # Send SIGTERM for graceful shutdown
            self.process.terminate()
            
            # Wait for process to terminate
            try:
                await asyncio.wait_for(
                    self.process.wait(),
                    timeout=self.termination_timeout
                )
                self.logger.info(
                    "MCP server process terminated gracefully",
                    pid=self.process.pid,
                    returncode=self.process.returncode
                )
            except asyncio.TimeoutError:
                # Force kill if it doesn't terminate gracefully
                self.logger.warning(
                    "MCP server process did not terminate gracefully, sending SIGKILL",
                    pid=self.process.pid
                )
                self.process.kill()
                await self.process.wait()
                self.logger.info(
                    "MCP server process killed",
                    pid=self.process.pid,
                    returncode=self.process.returncode
                )
        
        except Exception as e:
            self.logger.error(
                "Error terminating MCP server process",
                pid=self.process.pid,
                error=str(e),
                exc_info=True
            )
    
    def _get_auth_env(self) -> Dict[str, str]:
        """Get API keys/auth tokens for the MCP server environment."""
        auth_env = {}
        
        # Look for specific key in config
        if self.api_key_env_var:
            api_key = os.environ.get(self.api_key_env_var)
            if api_key:
                auth_env["MCP_API_KEY"] = api_key
                self.logger.debug(
                    "Using specific API key environment variable",
                    env_var=self.api_key_env_var
                )
                return auth_env
        
        # Fallback to generic key
        generic_key = os.environ.get("MCP_SERVER_API_KEY")
        if generic_key:
            auth_env["MCP_API_KEY"] = generic_key
            self.logger.debug("Using generic MCP_SERVER_API_KEY for auth")
            return auth_env
        
        self.logger.warning("No API key environment variable configured or found")
        return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the transport."""
        status = "available"
        message = "Process is running"
        
        # Check if process exists and is running
        if self.process is None:
            status = "initializing"
            message = "Process not started yet"
        elif self.process.returncode is not None:
            status = "unavailable"
            message = f"Process terminated with code {self.process.returncode}"
        
        return {
            "status": status,
            "transport_type": "stdio",
            "message": message,
            "pid": self.process.pid if self.process and hasattr(self.process, 'pid') else None,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "command": self.command,
            "args": self.args
        }
    
    async def cleanup(self) -> None:
        """Clean up resources and terminate the process."""
        # Cancel monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Terminate process
        await self.terminate_process()