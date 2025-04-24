#!/usr/bin/env python
# diagnostic.py

"""
Diagnostic tool for the LLM Gateway.
This script performs a series of tests to diagnose issues with the LLM Gateway.
"""

import asyncio
import logging
import os
import sys
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from asf.bo.backend.api.routers.llm.utils import load_config
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.factory import ProviderFactory
from asf.medical.llm_gateway.providers.openai_client import OpenAIClient

class LLMGatewayDiagnostic:
    """Diagnostic tool for the LLM Gateway."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the diagnostic tool."""
        self.config_path = config_path
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        }

    async def run_all_tests(self):
        """Run all diagnostic tests."""
        logger.info("Starting LLM Gateway diagnostic tests")

        # Test 1: Check configuration
        await self.test_configuration()

        # Test 2: Check provider factory
        await self.test_provider_factory()

        # Test 3: Check gateway client
        await self.test_gateway_client()

        # Test 4: Check OpenAI provider
        await self.test_openai_provider()

        # Test 5: Check environment variables
        await self.test_environment_variables()

        # Update summary
        self.update_summary()

        # Print summary
        self.print_summary()

        return self.results

    def update_summary(self):
        """Update the summary of test results."""
        self.results["summary"]["total"] = len(self.results["tests"])
        self.results["summary"]["passed"] = sum(1 for test in self.results["tests"] if test["status"] == "passed")
        self.results["summary"]["failed"] = sum(1 for test in self.results["tests"] if test["status"] == "failed")
        self.results["summary"]["skipped"] = sum(1 for test in self.results["tests"] if test["status"] == "skipped")

    def print_summary(self):
        """Print a summary of the test results."""
        summary = self.results["summary"]
        logger.info(f"Diagnostic tests completed: {summary['passed']}/{summary['total']} passed, "
                   f"{summary['failed']} failed, {summary['skipped']} skipped")

        # Print failed tests
        if summary["failed"] > 0:
            logger.error("Failed tests:")
            for test in self.results["tests"]:
                if test["status"] == "failed":
                    logger.error(f"  - {test['name']}: {test['message']}")

    def add_test_result(self, name: str, status: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Add a test result."""
        self.results["tests"].append({
            "name": name,
            "status": status,
            "message": message,
            "details": details or {}
        })

        if status == "passed":
            logger.info(f"✅ {name}: {message}")
        elif status == "failed":
            logger.error(f"❌ {name}: {message}")
        else:
            logger.warning(f"⚠️ {name}: {message}")

    async def test_configuration(self):
        """Test the configuration loading."""
        test_name = "Configuration Loading"

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
                    self.add_test_result(
                        test_name,
                        "failed",
                        "Configuration file not found",
                        {"searched_paths": possible_paths}
                    )
                    return

            # Load configuration using the utility function that handles local config
            config_dict = load_config(config_path)

            # Validate configuration
            try:
                gateway_config = GatewayConfig(**config_dict)
                self.gateway_config = gateway_config

                # Check for required fields
                if not gateway_config.gateway_id:
                    self.add_test_result(
                        test_name,
                        "failed",
                        "Configuration missing gateway_id",
                        {"config_path": config_path}
                    )
                    return

                if not gateway_config.allowed_providers:
                    self.add_test_result(
                        test_name,
                        "failed",
                        "Configuration missing allowed_providers",
                        {"config_path": config_path}
                    )
                    return

                # Check for providers
                providers = gateway_config.additional_config.get("providers", {})
                if not providers:
                    self.add_test_result(
                        test_name,
                        "failed",
                        "Configuration missing providers",
                        {"config_path": config_path}
                    )
                    return

                # Success
                self.add_test_result(
                    test_name,
                    "passed",
                    f"Configuration loaded successfully from {config_path}",
                    {
                        "config_path": config_path,
                        "gateway_id": gateway_config.gateway_id,
                        "allowed_providers": gateway_config.allowed_providers,
                        "provider_count": len(providers)
                    }
                )

            except Exception as e:
                self.add_test_result(
                    test_name,
                    "failed",
                    f"Configuration validation failed: {str(e)}",
                    {"config_path": config_path, "error": str(e)}
                )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                f"Configuration loading failed: {str(e)}",
                {"error": str(e)}
            )

    async def test_provider_factory(self):
        """Test the provider factory."""
        test_name = "Provider Factory"

        try:
            # Check if configuration test passed
            if not hasattr(self, 'gateway_config'):
                self.add_test_result(
                    test_name,
                    "skipped",
                    "Skipped because configuration test failed"
                )
                return

            # Create provider factory
            provider_factory = ProviderFactory()
            self.provider_factory = provider_factory

            # Initialize the provider registry
            if not hasattr(provider_factory, '_provider_registry'):
                provider_factory._provider_registry = {}
                provider_factory._provider_instances = {}
                provider_factory._instance_locks = {}
                provider_factory._allow_overwrite = False
                provider_factory._register_known_providers()

            # Check registered providers
            if not provider_factory._provider_registry:
                self.add_test_result(
                    test_name,
                    "failed",
                    "Provider factory has no registered providers"
                )
                return

            # Success
            self.add_test_result(
                test_name,
                "passed",
                "Provider factory initialized successfully",
                {"registered_providers": list(provider_factory._provider_registry.keys())}
            )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                f"Provider factory initialization failed: {str(e)}",
                {"error": str(e)}
            )

    async def test_gateway_client(self):
        """Test the gateway client."""
        test_name = "Gateway Client"

        try:
            # Check if previous tests passed
            if not hasattr(self, 'gateway_config') or not hasattr(self, 'provider_factory'):
                self.add_test_result(
                    test_name,
                    "skipped",
                    "Skipped because previous tests failed"
                )
                return

            # Create gateway client
            client = LLMGatewayClient(self.gateway_config, self.provider_factory)
            self.client = client

            # Success
            self.add_test_result(
                test_name,
                "passed",
                "Gateway client initialized successfully"
            )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                f"Gateway client initialization failed: {str(e)}",
                {"error": str(e)}
            )

    async def test_openai_provider(self):
        """Test the OpenAI provider."""
        test_name = "OpenAI Provider"

        try:
            # Check if configuration test passed
            if not hasattr(self, 'gateway_config'):
                self.add_test_result(
                    test_name,
                    "skipped",
                    "Skipped because configuration test failed"
                )
                return

            # Check if OpenAI provider is configured
            providers = self.gateway_config.additional_config.get("providers", {})
            openai_providers = [
                provider_id for provider_id, config in providers.items()
                if config.get("provider_type") == "openai"
            ]

            if not openai_providers:
                self.add_test_result(
                    test_name,
                    "skipped",
                    "No OpenAI providers configured"
                )
                return

            # Get first OpenAI provider
            provider_id = openai_providers[0]
            provider_config = providers[provider_id]

            # Check API key
            api_key = None
            api_key_env_var = provider_config.get("connection_params", {}).get("api_key_env_var")

            if api_key_env_var:
                api_key = os.environ.get(api_key_env_var)

            direct_api_key = provider_config.get("connection_params", {}).get("api_key")
            if direct_api_key:
                api_key = direct_api_key

            if not api_key:
                self.add_test_result(
                    test_name,
                    "failed",
                    f"OpenAI API key not found. Environment variable: {api_key_env_var}",
                    {"provider_id": provider_id, "api_key_env_var": api_key_env_var}
                )
                return

            # Create provider config
            provider_config_obj = ProviderConfig(
                provider_id=provider_id,
                provider_type="openai",
                connection_params=provider_config.get("connection_params", {}),
                models=provider_config.get("models", {})
            )

            # Create OpenAI client
            openai_client = OpenAIClient(provider_config_obj, self.gateway_config)

            # Check if client is initialized
            if not openai_client._client:
                self.add_test_result(
                    test_name,
                    "failed",
                    "OpenAI client not initialized",
                    {"provider_id": provider_id}
                )
                return

            # Success
            self.add_test_result(
                test_name,
                "passed",
                f"OpenAI provider '{provider_id}' initialized successfully",
                {
                    "provider_id": provider_id,
                    "api_key_masked": f"{api_key[:5]}...{api_key[-4:]}" if api_key else None,
                    "models": list(provider_config.get("models", {}).keys())
                }
            )

            # Store for later use
            self.openai_client = openai_client
            self.openai_provider_id = provider_id

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                f"OpenAI provider test failed: {str(e)}",
                {"error": str(e)}
            )

    async def test_environment_variables(self):
        """Test environment variables."""
        test_name = "Environment Variables"

        try:
            # Check if configuration test passed
            if not hasattr(self, 'gateway_config'):
                self.add_test_result(
                    test_name,
                    "skipped",
                    "Skipped because configuration test failed"
                )
                return

            # Get all providers
            providers = self.gateway_config.additional_config.get("providers", {})

            # Check environment variables for each provider
            env_vars = {}
            missing_env_vars = []

            for provider_id, config in providers.items():
                provider_type = config.get("provider_type")
                connection_params = config.get("connection_params", {})

                # Check for API key environment variables
                api_key_env_var = connection_params.get("api_key_env_var")
                if api_key_env_var:
                    api_key = os.environ.get(api_key_env_var)
                    if api_key:
                        env_vars[api_key_env_var] = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) > 9 else "***"
                    else:
                        missing_env_vars.append(api_key_env_var)

                # Check for other environment variables
                for param, value in connection_params.items():
                    if param.endswith("_env_var") and param != "api_key_env_var":
                        env_var = connection_params.get(param)
                        if env_var:
                            env_value = os.environ.get(env_var)
                            if env_value:
                                env_vars[env_var] = "***"  # Mask all values for security
                            else:
                                missing_env_vars.append(env_var)

            # Report results
            if missing_env_vars:
                self.add_test_result(
                    test_name,
                    "failed",
                    f"Missing environment variables: {', '.join(missing_env_vars)}",
                    {"missing": missing_env_vars, "found": list(env_vars.keys())}
                )
            else:
                self.add_test_result(
                    test_name,
                    "passed",
                    f"All required environment variables found",
                    {"found": list(env_vars.keys())}
                )

        except Exception as e:
            self.add_test_result(
                test_name,
                "failed",
                f"Environment variables test failed: {str(e)}",
                {"error": str(e)}
            )

async def main():
    """Run the diagnostic tool."""
    logger.info("Starting LLM Gateway diagnostic tool")

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="LLM Gateway diagnostic tool")
    parser.add_argument("--config", help="Path to the LLM Gateway configuration file")
    parser.add_argument("--output", help="Path to save the diagnostic results")
    args = parser.parse_args()

    # Run diagnostic
    diagnostic = LLMGatewayDiagnostic(args.config)
    results = await diagnostic.run_all_tests()

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Diagnostic results saved to {args.output}")

    # Return success if all tests passed
    return results["summary"]["failed"] == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
