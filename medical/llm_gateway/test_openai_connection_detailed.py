#!/usr/bin/env python
# test_openai_connection_detailed.py

"""
Detailed test script for OpenAI API connection through the LLM Gateway.
This script provides verbose logging and step-by-step verification of the connection process.
"""

import asyncio
import logging
import os
import sys
import yaml
import json
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

# Import gateway components
from asf.medical.llm_gateway.core.models import (
    GatewayConfig,
    InterventionConfig,
    InterventionContext,
    LLMConfig,
    LLMRequest,
    ProviderConfig,
)
from asf.medical.llm_gateway.providers.openai_client import OpenAIClient

class OpenAIConnectionTester:
    """Detailed tester for OpenAI API connection."""
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the tester."""
        self.api_key = api_key
        self.config_path = config_path
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "steps": [],
            "success": False,
            "error": None
        }
    
    def add_step(self, name: str, success: bool, message: str, details: Optional[Dict[str, Any]] = None):
        """Add a step to the results."""
        self.results["steps"].append({
            "name": name,
            "success": success,
            "message": message,
            "details": details or {}
        })
        
        if success:
            logger.info(f"✅ {name}: {message}")
        else:
            logger.error(f"❌ {name}: {message}")
    
    async def run_test(self):
        """Run the connection test."""
        logger.info("Starting detailed OpenAI connection test")
        
        try:
            # Step 1: Check API key
            await self.check_api_key()
            
            # Step 2: Load configuration
            await self.load_configuration()
            
            # Step 3: Initialize OpenAI client
            await self.initialize_client()
            
            # Step 4: Send test request
            await self.send_test_request()
            
            # Set overall success
            self.results["success"] = all(step["success"] for step in self.results["steps"])
            
        except Exception as e:
            logger.exception("Unexpected error during test")
            self.results["success"] = False
            self.results["error"] = str(e)
            self.results["traceback"] = traceback.format_exc()
        
        # Print summary
        if self.results["success"]:
            logger.info("✅ OpenAI connection test PASSED!")
        else:
            logger.error("❌ OpenAI connection test FAILED!")
            if self.results["error"]:
                logger.error(f"Error: {self.results['error']}")
        
        return self.results
    
    async def check_api_key(self):
        """Check if API key is available."""
        step_name = "API Key Check"
        
        # Check command-line provided key
        if self.api_key:
            api_key = self.api_key
            source = "command-line argument"
        else:
            # Check environment variable
            api_key = os.environ.get("OPENAI_API_KEY")
            source = "OPENAI_API_KEY environment variable"
            
            # Check other common environment variables
            if not api_key:
                for env_var in ["AZURE_OPENAI_API_KEY", "OPENAI_KEY"]:
                    api_key = os.environ.get(env_var)
                    if api_key:
                        source = f"{env_var} environment variable"
                        break
        
        if not api_key:
            self.add_step(
                step_name,
                False,
                "API key not found",
                {"checked_sources": ["command-line argument", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_KEY"]}
            )
            return
        
        # Store API key for later use
        self.api_key = api_key
        
        # Mask API key for logging
        masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 9 else "***"
        
        self.add_step(
            step_name,
            True,
            f"API key found from {source}",
            {"api_key_masked": masked_key}
        )
    
    async def load_configuration(self):
        """Load configuration for the test."""
        step_name = "Configuration Loading"
        
        try:
            # Try to find configuration file
            if self.config_path:
                config_path = self.config_path
            else:
                # Look in standard locations
                possible_paths = [
                    os.path.join(os.getcwd(), "llm_gateway_config.yaml"),
                    os.path.join(os.getcwd(), "config", "llm_gateway_config.yaml"),
                    os.path.join(os.getcwd(), "asf", "bo", "backend", "config", "llm", "llm_gateway_config.yaml"),
                    os.path.join(project_root, "bo", "backend", "config", "llm", "llm_gateway_config.yaml"),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        config_path = path
                        break
                else:
                    # Create minimal configuration
                    logger.warning("Configuration file not found, using minimal configuration")
                    self.gateway_config = GatewayConfig(
                        gateway_id="test_gateway",
                        description="Test Gateway",
                        default_provider="openai_test",
                        allowed_providers=["openai_test"],
                        default_timeout_seconds=30,
                        max_retries=2,
                        retry_delay_seconds=1,
                        additional_config={}
                    )
                    
                    self.provider_config = ProviderConfig(
                        provider_id="openai_test",
                        provider_type="openai",
                        description="Test OpenAI provider",
                        models={"gpt-3.5-turbo": {}},
                        connection_params={
                            "api_key": self.api_key,
                            "max_retries": 2,
                            "timeout_seconds": 30,
                        }
                    )
                    
                    self.add_step(
                        step_name,
                        True,
                        "Using minimal configuration",
                        {"config_source": "generated"}
                    )
                    return
            
            # Load configuration
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create gateway config
            self.gateway_config = GatewayConfig(**config_dict)
            
            # Find OpenAI provider
            providers = config_dict.get("additional_config", {}).get("providers", {})
            openai_providers = [
                provider_id for provider_id, config in providers.items()
                if config.get("provider_type") == "openai"
            ]
            
            if not openai_providers:
                # Create minimal provider config
                self.provider_config = ProviderConfig(
                    provider_id="openai_test",
                    provider_type="openai",
                    description="Test OpenAI provider",
                    models={"gpt-3.5-turbo": {}},
                    connection_params={
                        "api_key": self.api_key,
                        "max_retries": 2,
                        "timeout_seconds": 30,
                    }
                )
                
                self.add_step(
                    step_name,
                    True,
                    "No OpenAI provider found in configuration, using minimal configuration",
                    {"config_path": config_path, "config_source": "generated"}
                )
                return
            
            # Get first OpenAI provider
            provider_id = openai_providers[0]
            provider_config_dict = providers[provider_id]
            
            # Override API key if provided
            if self.api_key:
                provider_config_dict["connection_params"] = provider_config_dict.get("connection_params", {})
                provider_config_dict["connection_params"]["api_key"] = self.api_key
            
            # Create provider config
            self.provider_config = ProviderConfig(
                provider_id=provider_id,
                provider_type="openai",
                connection_params=provider_config_dict.get("connection_params", {}),
                models=provider_config_dict.get("models", {"gpt-3.5-turbo": {}})
            )
            
            self.add_step(
                step_name,
                True,
                f"Configuration loaded from {config_path}",
                {
                    "config_path": config_path,
                    "provider_id": provider_id,
                    "models": list(provider_config_dict.get("models", {}).keys())
                }
            )
            
        except Exception as e:
            self.add_step(
                step_name,
                False,
                f"Configuration loading failed: {str(e)}",
                {"error": str(e)}
            )
            raise
    
    async def initialize_client(self):
        """Initialize the OpenAI client."""
        step_name = "Client Initialization"
        
        try:
            # Check if previous steps succeeded
            if not hasattr(self, 'gateway_config') or not hasattr(self, 'provider_config'):
                self.add_step(
                    step_name,
                    False,
                    "Skipped because previous steps failed"
                )
                return
            
            # Initialize client
            logger.info("Initializing OpenAI client...")
            self.client = OpenAIClient(self.provider_config, self.gateway_config)
            
            # Check if client is initialized
            if not self.client._client:
                self.add_step(
                    step_name,
                    False,
                    "OpenAI client not initialized",
                    {"error": "Client initialization failed"}
                )
                return
            
            self.add_step(
                step_name,
                True,
                "OpenAI client initialized successfully",
                {"client_type": type(self.client._client).__name__}
            )
            
        except Exception as e:
            self.add_step(
                step_name,
                False,
                f"Client initialization failed: {str(e)}",
                {"error": str(e)}
            )
            raise
    
    async def send_test_request(self):
        """Send a test request to the OpenAI API."""
        step_name = "Test Request"
        
        try:
            # Check if previous steps succeeded
            if not hasattr(self, 'client'):
                self.add_step(
                    step_name,
                    False,
                    "Skipped because previous steps failed"
                )
                return
            
            # Create request
            logger.info("Creating test request...")
            request_id = f"test_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            
            # Create intervention context
            context = InterventionContext(
                request_id=request_id,
                conversation_history=[],
                user_id="test_user",
                session_id="test_session",
                timestamp_start=datetime.now(timezone.utc),
                intervention_config=InterventionConfig(
                    enabled_pre_interventions=[],
                    enabled_post_interventions=[],
                    fail_open=True
                ),
                intervention_data={}
            )
            
            # Create LLM config
            llm_config = LLMConfig(
                model_identifier="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=100,
                system_prompt="You are a helpful assistant."
            )
            
            # Create the request
            request = LLMRequest(
                version="1.0",
                initial_context=context,
                config=llm_config,
                prompt_content="Hello, this is a test request. Please respond with a short greeting.",
                tools=[]
            )
            
            # Send request
            logger.info(f"Sending test request to OpenAI API with request ID: {request_id}")
            response = await self.client.generate(request)
            
            # Check response
            if response.error_details:
                self.add_step(
                    step_name,
                    False,
                    f"Error in response: {response.error_details.code} - {response.error_details.message}",
                    {
                        "error_code": response.error_details.code,
                        "error_message": response.error_details.message,
                        "provider_details": response.error_details.provider_error_details
                    }
                )
                return
            
            # Success
            self.add_step(
                step_name,
                True,
                "Test request successful",
                {
                    "generated_content": response.generated_content,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "latency_ms": response.performance_metrics.llm_latency_ms if response.performance_metrics else None
                }
            )
            
        except Exception as e:
            self.add_step(
                step_name,
                False,
                f"Test request failed: {str(e)}",
                {"error": str(e)}
            )
            raise

async def main():
    """Run the OpenAI connection test."""
    logger.info("Starting detailed OpenAI connection test")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Detailed OpenAI connection test")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--config", help="Path to the LLM Gateway configuration file")
    parser.add_argument("--output", help="Path to save the test results")
    args = parser.parse_args()
    
    # Run test
    tester = OpenAIConnectionTester(args.api_key, args.config)
    results = await tester.run_test()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to {args.output}")
    
    # Return success status
    return results["success"]

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
