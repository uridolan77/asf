"""
Knowledge base updater for the Medical Research Synthesizer.

This module provides a service for updating knowledge bases on a schedule.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import time
import threading
from datetime import datetime, timedelta

from asf.medical.orchestration.task_scheduler import TaskScheduler
from asf.medical.storage.repositories import KnowledgeBaseRepository
from asf.medical.storage.database import get_db
from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class KnowledgeBaseUpdater:
    """
    Service for updating knowledge bases on a schedule.
    
    This service provides methods for scheduling and executing knowledge base updates.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Create a singleton instance of the knowledge base updater.
        
        Returns:
            KnowledgeBaseUpdater: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(KnowledgeBaseUpdater, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the knowledge base updater."""
        self.task_scheduler = TaskScheduler()
        self.kb_repository = KnowledgeBaseRepository()
        self.running = False
        self.scheduler_thread = None
        self.update_interval = 60 * 60  # 1 hour
        
        logger.info("Knowledge base updater initialized")
    
    def start(self) -> None:
        """Start the knowledge base updater."""
        if self.running:
            logger.info("Knowledge base updater already running")
            return
        
        logger.info("Starting knowledge base updater")
        
        # Start task scheduler
        self.task_scheduler.start()
        
        # Start scheduler thread
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Knowledge base updater started")
    
    def stop(self) -> None:
        """Stop the knowledge base updater."""
        if not self.running:
            logger.info("Knowledge base updater not running")
            return
        
        logger.info("Stopping knowledge base updater")
        
        # Stop scheduler thread
        self.running = False
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # Stop task scheduler
        self.task_scheduler.stop()
        
        logger.info("Knowledge base updater stopped")
    
    def _scheduler_loop(self) -> None:
        """Scheduler loop for checking and scheduling knowledge base updates."""
        logger.info("Scheduler loop started")
        
        while self.running:
            try:
                # Check for knowledge bases that need updating
                self._check_for_updates()
                
                # Sleep until next check
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(60)  # Sleep for 1 minute before retrying
        
        logger.info("Scheduler loop stopped")
    
    def _check_for_updates(self) -> None:
        """Check for knowledge bases that need updating."""
        logger.info("Checking for knowledge base updates")
        
        try:
            # Get knowledge bases that are due for an update
            with get_db() as db:
                knowledge_bases = self.kb_repository.get_due_for_update(db)
            
            logger.info(f"Found {len(knowledge_bases)} knowledge bases due for update")
            
            # Schedule updates
            for kb in knowledge_bases:
                self._schedule_update(kb)
        except Exception as e:
            logger.error(f"Error checking for updates: {str(e)}")
    
    def _schedule_update(self, kb) -> None:
        """
        Schedule an update for a knowledge base.
        
        Args:
            kb: Knowledge base to update
        """
        logger.info(f"Scheduling update for knowledge base: {kb.name}")
        
        # Schedule task
        task_id = self.task_scheduler.schedule_task(
            func=self._update_knowledge_base,
            args=(kb.id, kb.kb_id, kb.name, kb.query, kb.file_path),
            task_id=f"kb_update_{kb.kb_id}_{datetime.now().isoformat()}",
            priority=0,
            timeout=3600,  # 1 hour
            retry_count=3,
            retry_delay=300  # 5 minutes
        )
        
        logger.info(f"Update scheduled for knowledge base: {kb.name}, task_id: {task_id}")
    
    def _update_knowledge_base(
        self,
        kb_id: int,
        kb_uuid: str,
        name: str,
        query: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Update a knowledge base.
        
        Args:
            kb_id: Knowledge base ID
            kb_uuid: Knowledge base UUID
            name: Knowledge base name
            query: Search query
            file_path: Path to the knowledge base file
            
        Returns:
            Update result
        """
        logger.info(f"Updating knowledge base: {name}")
        
        try:
            # Import here to avoid circular imports
            from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
            
            # Create synthesizer
            synthesizer = EnhancedMedicalResearchSynthesizer(
                email=settings.NCBI_EMAIL,
                api_key=settings.NCBI_API_KEY.get_secret_value() if settings.NCBI_API_KEY else None,
                impact_factor_source=settings.IMPACT_FACTOR_SOURCE
            )
            
            # Update knowledge base
            result = synthesizer.incremental_client.search_and_update_knowledge_base(
                query,
                file_path,
                max_results=100
            )
            
            # Update knowledge base in database
            with get_db() as db:
                # Update last updated timestamp
                self.kb_repository.update_last_updated(db, kb_id)
                
                # Add update record
                self.kb_repository.add_update_record(
                    db,
                    kb_id,
                    result["new_count"],
                    result["total_count"],
                    "success"
                )
            
            logger.info(f"Knowledge base updated: {name}, new_count: {result['new_count']}, total_count: {result['total_count']}")
            
            return result
        except Exception as e:
            logger.error(f"Error updating knowledge base: {name}, error: {str(e)}")
            
            # Add update record with error
            try:
                with get_db() as db:
                    self.kb_repository.add_update_record(
                        db,
                        kb_id,
                        0,
                        0,
                        "failure",
                        str(e)
                    )
            except Exception as db_error:
                logger.error(f"Error adding update record: {str(db_error)}")
            
            raise
