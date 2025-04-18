"""
Medical Client Service

This module provides services for managing medical clients, their configurations,
status, and usage statistics.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, date, timedelta
import logging
import json
import httpx
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from ..models.medical_client import (
    MedicalClient, 
    MedicalClientConfig, 
    MedicalClientStatus, 
    MedicalClientStatusLog, 
    MedicalClientUsageStat,
    ClientStatus
)

# Import DSPy related models - these will need to be created
from ..models.dspy_client import (
    DSPyClient,
    DSPyClientConfig,
    DSPyClientStatus,
    DSPyClientStatusLog,
    DSPyClientUsageStat,
    DSPyModule,
    DSPyModuleMetric,
    DSPyAuditLog,
    DSPyCircuitBreakerStatus
)

logger = logging.getLogger(__name__)

class MedicalClientService:
    """Service for managing medical clients"""

    def __init__(self, db: Session):
        """Initialize the service with a database session"""
        self.db = db

    def get_all_clients(self) -> List[Dict[str, Any]]:
        """Get all medical clients with their status"""
        clients = self.db.query(MedicalClient).all()
        return [client.to_dict() for client in clients]

    def get_client(self, client_id: str) -> Dict[str, Any]:
        """Get a specific medical client by ID"""
        client = self.db.query(MedicalClient).filter(MedicalClient.client_id == client_id).first()
        if not client:
            return None
        
        # Get the client configuration
        config = self.db.query(MedicalClientConfig).filter(
            MedicalClientConfig.client_id == client_id
        ).first()
        
        # Combine client and config data
        client_data = client.to_dict()
        if config:
            client_data["config"] = config.to_dict()
        
        return client_data

    def update_client_config(self, client_id: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a medical client configuration"""
        # Check if client exists
        client = self.db.query(MedicalClient).filter(MedicalClient.client_id == client_id).first()
        if not client:
            return None
        
        # Get or create client config
        config = self.db.query(MedicalClientConfig).filter(
            MedicalClientConfig.client_id == client_id
        ).first()
        
        if not config:
            config = MedicalClientConfig(client_id=client_id)
            self.db.add(config)
        
        # Update config fields
        for key, value in config_data.items():
            if hasattr(config, key) and key not in ['config_id', 'client_id', 'created_at', 'updated_at']:
                setattr(config, key, value)
        
        # Handle additional_config as JSON
        if 'additional_config' in config_data and isinstance(config_data['additional_config'], dict):
            config.additional_config = config_data['additional_config']
        
        self.db.commit()
        return config.to_dict()

    async def test_client_connection(self, client_id: str) -> Dict[str, Any]:
        """Test connection to a medical client"""
        # Get client and config
        client = self.db.query(MedicalClient).filter(MedicalClient.client_id == client_id).first()
        if not client:
            return {"success": False, "message": f"Client not found: {client_id}"}
        
        config = self.db.query(MedicalClientConfig).filter(
            MedicalClientConfig.client_id == client_id
        ).first()
        
        # Get current status or create new one
        status = self.db.query(MedicalClientStatus).filter(
            MedicalClientStatus.client_id == client_id
        ).first()
        
        if not status:
            status = MedicalClientStatus(client_id=client_id)
            self.db.add(status)
        
        # Create status log entry
        status_log = MedicalClientStatusLog(client_id=client_id)
        self.db.add(status_log)
        
        # Test connection based on client type
        try:
            start_time = datetime.now()
            result = await self._test_specific_client(client_id, client, config)
            end_time = datetime.now()
            
            # Calculate response time
            response_time = (end_time - start_time).total_seconds()
            
            if result["success"]:
                # Update status to connected
                status.status = ClientStatus.CONNECTED
                status.response_time = response_time
                status.last_checked = datetime.now()
                status.error_message = None
                
                # Update status log
                status_log.status = ClientStatus.CONNECTED
                status_log.response_time = response_time
                status_log.error_message = None
            else:
                # Update status to error
                status.status = ClientStatus.ERROR
                status.response_time = response_time
                status.last_checked = datetime.now()
                status.error_message = result.get("message", "Unknown error")
                
                # Update status log
                status_log.status = ClientStatus.ERROR
                status_log.response_time = response_time
                status_log.error_message = result.get("message", "Unknown error")
            
            self.db.commit()
            
            # Add response time to result
            result["response_time"] = response_time
            return result
            
        except Exception as e:
            logger.error(f"Error testing client connection: {str(e)}")
            
            # Update status to error
            status.status = ClientStatus.ERROR
            status.last_checked = datetime.now()
            status.error_message = str(e)
            
            # Update status log
            status_log.status = ClientStatus.ERROR
            status_log.error_message = str(e)
            
            self.db.commit()
            
            return {"success": False, "message": f"Error: {str(e)}"}

    async def _test_specific_client(self, client_id: str, client: MedicalClient, config: MedicalClientConfig) -> Dict[str, Any]:
        """Test connection to a specific client type"""
        base_url = client.base_url
        timeout = config.timeout if config and config.timeout else 30
        
        # Default test endpoints for each client
        test_endpoints = {
            "ncbi": "/entrez/eutils/esearch.fcgi?db=pubmed&term=covid&retmax=1&format=json",
            "umls": "/content/current?apiKey=",  # Will append API key if available
            "clinical_trials": "/info/study_structure",
            "cochrane": "/v1/health",
            "crossref": "/works?query=medicine&rows=1",
            "snomed": "/branches"
        }
        
        if client_id not in test_endpoints:
            return {"success": False, "message": f"No test endpoint defined for client: {client_id}"}
        
        endpoint = test_endpoints[client_id]
        
        # Add API key for UMLS if available
        if client_id == "umls" and config and config.api_key:
            endpoint += config.api_key
        
        # Prepare headers
        headers = {}
        if config:
            # Add email for NCBI and Crossref if available
            if client_id in ["ncbi", "crossref"] and config.email:
                headers["User-Agent"] = f"MedicalResearchSynthesizer/1.0 ({config.email})"
            
            # Add API key header for certain clients
            if config.api_key and client_id in ["cochrane", "crossref"]:
                headers["Authorization"] = f"Bearer {config.api_key}"
        
        # Make the request
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url}{endpoint}", headers=headers)
                
                if response.status_code < 400:
                    return {
                        "success": True, 
                        "status_code": response.status_code,
                        "message": "Connection successful"
                    }
                else:
                    return {
                        "success": False,
                        "status_code": response.status_code,
                        "message": f"Error: {response.text}"
                    }
        except Exception as e:
            logger.error(f"Error connecting to {client_id}: {str(e)}")
            return {"success": False, "message": f"Connection error: {str(e)}"}

    def get_client_usage(self, client_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get usage statistics for a client"""
        # Check if client exists
        client = self.db.query(MedicalClient).filter(MedicalClient.client_id == client_id).first()
        if not client:
            return None
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get usage stats
        stats = self.db.query(MedicalClientUsageStat).filter(
            MedicalClientUsageStat.client_id == client_id,
            MedicalClientUsageStat.date >= start_date,
            MedicalClientUsageStat.date <= end_date
        ).order_by(MedicalClientUsageStat.date).all()
        
        return [stat.to_dict() for stat in stats]

    def record_client_usage(self, client_id: str, success: bool, cached: bool, response_time: float) -> None:
        """Record usage statistics for a client"""
        # Check if client exists
        client = self.db.query(MedicalClient).filter(MedicalClient.client_id == client_id).first()
        if not client:
            logger.error(f"Cannot record usage for non-existent client: {client_id}")
            return
        
        # Get or create today's stats
        today = date.today()
        stat = self.db.query(MedicalClientUsageStat).filter(
            MedicalClientUsageStat.client_id == client_id,
            MedicalClientUsageStat.date == today
        ).first()
        
        if not stat:
            stat = MedicalClientUsageStat(
                client_id=client_id,
                date=today,
                requests_count=0,
                successful_requests=0,
                failed_requests=0,
                cached_requests=0,
                total_response_time=0,
                average_response_time=0
            )
            self.db.add(stat)
        
        # Update stats
        stat.requests_count += 1
        if success:
            stat.successful_requests += 1
        else:
            stat.failed_requests += 1
        
        if cached:
            stat.cached_requests += 1
        
        stat.total_response_time += response_time
        stat.average_response_time = stat.total_response_time / stat.requests_count
        
        self.db.commit()

    def get_client_status_history(self, client_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get status history for a client"""
        # Check if client exists
        client = self.db.query(MedicalClient).filter(MedicalClient.client_id == client_id).first()
        if not client:
            return None
        
        # Get status logs
        logs = self.db.query(MedicalClientStatusLog).filter(
            MedicalClientStatusLog.client_id == client_id
        ).order_by(MedicalClientStatusLog.checked_at.desc()).limit(limit).all()
        
        return [log.to_dict() for log in logs]

class DSPyClientService:
    """Service for managing DSPy clients, modules, and related metrics"""

    def __init__(self, db: Session):
        """Initialize the service with a database session"""
        self.db = db

    def get_all_dspy_clients(self) -> List[Dict[str, Any]]:
        """Get all DSPy clients with their status"""
        clients = self.db.query(DSPyClient).all()
        return [client.to_dict() for client in clients]

    def get_dspy_client(self, client_id: str) -> Dict[str, Any]:
        """Get a specific DSPy client by ID"""
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            return None
        
        # Get the client configuration
        config = self.db.query(DSPyClientConfig).filter(
            DSPyClientConfig.client_id == client_id
        ).first()
        
        # Combine client and config data
        client_data = client.to_dict()
        if config:
            client_data["config"] = config.to_dict()
        
        return client_data

    def update_dspy_client_config(self, client_id: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a DSPy client configuration"""
        # Check if client exists
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            return None
        
        # Get or create client config
        config = self.db.query(DSPyClientConfig).filter(
            DSPyClientConfig.client_id == client_id
        ).first()
        
        if not config:
            config = DSPyClientConfig(client_id=client_id)
            self.db.add(config)
        
        # Update config fields
        for key, value in config_data.items():
            if hasattr(config, key) and key not in ['config_id', 'client_id', 'created_at', 'updated_at']:
                setattr(config, key, value)
        
        # Handle additional_config as JSON
        if 'additional_config' in config_data and isinstance(config_data['additional_config'], dict):
            config.additional_config = config_data['additional_config']
        
        self.db.commit()
        return config.to_dict()

    async def test_dspy_client_connection(self, client_id: str) -> Dict[str, Any]:
        """Test connection to a DSPy client"""
        # Get client and config
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            return {"success": False, "message": f"Client not found: {client_id}"}
        
        config = self.db.query(DSPyClientConfig).filter(
            DSPyClientConfig.client_id == client_id
        ).first()
        
        # Get current status or create new one
        status = self.db.query(DSPyClientStatus).filter(
            DSPyClientStatus.client_id == client_id
        ).first()
        
        if not status:
            status = DSPyClientStatus(client_id=client_id)
            self.db.add(status)
        
        # Create status log entry
        status_log = DSPyClientStatusLog(client_id=client_id)
        self.db.add(status_log)
        
        # Test connection to DSPy client
        try:
            start_time = datetime.now()
            
            # For DSPy, we'll test the health endpoint
            base_url = client.base_url
            timeout = config.timeout if config and config.timeout else 30
            
            # Make the request to the health endpoint
            async with httpx.AsyncClient(timeout=timeout) as http_client:
                response = await http_client.get(f"{base_url}/dspy/health")
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                if response.status_code == 200:
                    # Update status to connected
                    status.status = "CONNECTED"
                    status.response_time = response_time
                    status.last_checked = datetime.now()
                    status.error_message = None
                    
                    # Update status log
                    status_log.status = "CONNECTED"
                    status_log.response_time = response_time
                    status_log.error_message = None
                    
                    self.db.commit()
                    
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "message": "Connection successful",
                        "response_time": response_time
                    }
                else:
                    # Update status to error
                    status.status = "ERROR"
                    status.response_time = response_time
                    status.last_checked = datetime.now()
                    status.error_message = f"Error: {response.text}"
                    
                    # Update status log
                    status_log.status = "ERROR"
                    status_log.response_time = response_time
                    status_log.error_message = f"Error: {response.text}"
                    
                    self.db.commit()
                    
                    return {
                        "success": False,
                        "status_code": response.status_code,
                        "message": f"Error: {response.text}",
                        "response_time": response_time
                    }
                
        except Exception as e:
            logger.error(f"Error testing DSPy client connection: {str(e)}")
            
            # Update status to error
            status.status = "ERROR"
            status.last_checked = datetime.now()
            status.error_message = str(e)
            
            # Update status log
            status_log.status = "ERROR"
            status_log.error_message = str(e)
            
            self.db.commit()
            
            return {"success": False, "message": f"Error: {str(e)}"}

    def get_dspy_modules(self, client_id: str) -> List[Dict[str, Any]]:
        """Get all modules for a DSPy client"""
        # Check if client exists
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            return []
        
        # Get modules
        modules = self.db.query(DSPyModule).filter(
            DSPyModule.client_id == client_id
        ).all()
        
        return [module.to_dict() for module in modules]

    def get_dspy_module(self, module_id: str) -> Dict[str, Any]:
        """Get a specific DSPy module by ID"""
        module = self.db.query(DSPyModule).filter(
            DSPyModule.module_id == module_id
        ).first()
        
        if not module:
            return None
        
        # Get module metrics
        metrics = self.db.query(DSPyModuleMetric).filter(
            DSPyModuleMetric.module_id == module_id
        ).order_by(DSPyModuleMetric.timestamp.desc()).limit(100).all()
        
        module_data = module.to_dict()
        module_data["metrics"] = [metric.to_dict() for metric in metrics]
        
        return module_data

    def get_dspy_client_usage(self, client_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get usage statistics for a DSPy client"""
        # Check if client exists
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            return None
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get usage stats
        stats = self.db.query(DSPyClientUsageStat).filter(
            DSPyClientUsageStat.client_id == client_id,
            DSPyClientUsageStat.date >= start_date,
            DSPyClientUsageStat.date <= end_date
        ).order_by(DSPyClientUsageStat.date).all()
        
        return [stat.to_dict() for stat in stats]

    def record_dspy_client_usage(
        self, 
        client_id: str, 
        module_name: str, 
        success: bool, 
        cached: bool, 
        response_time: float,
        tokens_used: int = 0
    ) -> None:
        """Record usage statistics for a DSPy client"""
        # Check if client exists
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            logger.error(f"Cannot record usage for non-existent DSPy client: {client_id}")
            return
        
        # Get or create today's stats
        today = date.today()
        stat = self.db.query(DSPyClientUsageStat).filter(
            DSPyClientUsageStat.client_id == client_id,
            DSPyClientUsageStat.date == today
        ).first()
        
        if not stat:
            stat = DSPyClientUsageStat(
                client_id=client_id,
                date=today,
                requests_count=0,
                successful_requests=0,
                failed_requests=0,
                cached_requests=0,
                total_response_time=0,
                average_response_time=0,
                tokens_used=0
            )
            self.db.add(stat)
        
        # Update stats
        stat.requests_count += 1
        if success:
            stat.successful_requests += 1
        else:
            stat.failed_requests += 1
        
        if cached:
            stat.cached_requests += 1
        
        stat.total_response_time += response_time
        stat.average_response_time = stat.total_response_time / stat.requests_count
        stat.tokens_used += tokens_used
        
        self.db.commit()
        
        # Update module metrics if module_name is provided
        if module_name:
            module = self.db.query(DSPyModule).filter(
                DSPyModule.client_id == client_id,
                DSPyModule.name == module_name
            ).first()
            
            if module:
                # Create metric record
                metric = DSPyModuleMetric(
                    module_id=module.module_id,
                    timestamp=datetime.now(),
                    response_time=response_time,
                    success=success,
                    cached=cached,
                    tokens_used=tokens_used
                )
                self.db.add(metric)
                self.db.commit()

    def get_dspy_client_status_history(self, client_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get status history for a DSPy client"""
        # Check if client exists
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            return None
        
        # Get status logs
        logs = self.db.query(DSPyClientStatusLog).filter(
            DSPyClientStatusLog.client_id == client_id
        ).order_by(DSPyClientStatusLog.checked_at.desc()).limit(limit).all()
        
        return [log.to_dict() for log in logs]

    def get_dspy_audit_logs(
        self, 
        client_id: str = None, 
        module_id: str = None, 
        event_type: str = None,
        start_date: date = None,
        end_date: date = None,
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get audit logs with flexible filtering options"""
        query = self.db.query(DSPyAuditLog)
        
        # Apply filters
        if client_id:
            query = query.filter(DSPyAuditLog.client_id == client_id)
        
        if module_id:
            query = query.filter(DSPyAuditLog.module_id == module_id)
        
        if event_type:
            query = query.filter(DSPyAuditLog.event_type == event_type)
        
        if start_date:
            query = query.filter(DSPyAuditLog.timestamp >= start_date)
        
        if end_date:
            query = query.filter(DSPyAuditLog.timestamp <= end_date)
        
        # Apply pagination
        logs = query.order_by(DSPyAuditLog.timestamp.desc()).offset(offset).limit(limit).all()
        
        return [log.to_dict() for log in logs]

    def get_circuit_breaker_status(self, client_id: str) -> List[Dict[str, Any]]:
        """Get circuit breaker status for a DSPy client"""
        # Check if client exists
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            return []
        
        # Get circuit breaker status
        statuses = self.db.query(DSPyCircuitBreakerStatus).filter(
            DSPyCircuitBreakerStatus.client_id == client_id
        ).all()
        
        return [status.to_dict() for status in statuses]

    def sync_dspy_modules(self, client_id: str) -> Dict[str, Any]:
        """Synchronize modules from a DSPy client"""
        # Check if client exists
        client = self.db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            return {"success": False, "message": f"Client not found: {client_id}"}
        
        try:
            # In a real implementation, we would call the DSPy client API to get the modules
            # For now, we'll simulate this with a mock response
            async def fetch_modules():
                async with httpx.AsyncClient(timeout=30) as http_client:
                    response = await http_client.get(f"{client.base_url}/dspy/modules")
                    if response.status_code == 200:
                        return response.json()
                    else:
                        raise Exception(f"Failed to fetch modules: {response.text}")
            
            # Run the async function in the event loop
            modules = asyncio.run(fetch_modules())
            
            # Update database with modules
            for module_data in modules:
                module = self.db.query(DSPyModule).filter(
                    DSPyModule.client_id == client_id,
                    DSPyModule.name == module_data["name"]
                ).first()
                
                if not module:
                    module = DSPyModule(
                        client_id=client_id,
                        name=module_data["name"],
                        description=module_data.get("description", ""),
                        module_type=module_data.get("module_type", ""),
                        class_name=module_data.get("class_name", ""),
                        registered_at=datetime.now()
                    )
                    self.db.add(module)
                else:
                    module.description = module_data.get("description", module.description)
                    module.module_type = module_data.get("module_type", module.module_type)
                    module.class_name = module_data.get("class_name", module.class_name)
                    module.updated_at = datetime.now()
            
            self.db.commit()
            
            return {
                "success": True, 
                "message": f"Successfully synced {len(modules)} modules",
                "modules_count": len(modules)
            }
            
        except Exception as e:
            logger.error(f"Error syncing DSPy modules: {str(e)}")
            return {"success": False, "message": f"Error syncing modules: {str(e)}"}
