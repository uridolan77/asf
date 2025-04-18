"""
Knowledge base updater for the Medical Research Synthesizer.
This module provides a service for updating knowledge bases on a schedule.
"""
import logging
import time
import threading
from asf.medical.orchestration.task_scheduler import TaskScheduler
from asf.medical.storage.repositories import KnowledgeBaseRepository
from asf.medical.storage.database import get_db
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
        """
        Initialize the knowledge base updater.
        
        Sets up task scheduler, repository and default update interval.
        """
        self.task_scheduler = TaskScheduler()
        self.kb_repository = KnowledgeBaseRepository()
        self.running = False
        self.scheduler_thread = None
        self.update_interval = 60 * 60  # 1 hour
        logger.info("Knowledge base updater initialized")
    def start(self) -> None:
        """
        Start the knowledge base updater.
        
        Starts the task scheduler and scheduler loop thread.
        """
        if self.running:
            logger.info("Knowledge base updater already running")
            return
        logger.info("Starting knowledge base updater")
        self.task_scheduler.start()
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        logger.info("Knowledge base updater started")
    def stop(self) -> None:
        """
        Stop the knowledge base updater.
        
        Stops the task scheduler and scheduler loop thread.
        """
        if not self.running:
            logger.info("Knowledge base updater not running")
            return
        logger.info("Stopping knowledge base updater")
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.task_scheduler.stop()
        logger.info("Knowledge base updater stopped")
    def _scheduler_loop(self) -> None:
        """
        Scheduler loop for checking and scheduling knowledge base updates.
        
        Periodically checks for updates and schedules them.
        """
        logger.info("Scheduler loop started")
        while self.running:
            try:
                self._check_for_updates()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(60)  # Sleep for 1 minute before retrying
        logger.info("Scheduler loop stopped")
    def _check_for_updates(self) -> None:
        """
        Check for knowledge bases that need updating.
        
        Queries the repository for knowledge bases due for update and schedules them.
        """
        logger.info("Checking for knowledge base updates")
        try:
            with get_db() as db:
                knowledge_bases = self.kb_repository.get_due_for_update(db)
            logger.info(f"Found {len(knowledge_bases)} knowledge bases due for update")
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
        pass