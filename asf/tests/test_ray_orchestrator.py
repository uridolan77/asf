"""
Test Ray Orchestrator

This module provides tests for the Ray-based orchestration framework
in the ASF framework.
"""

import os
import sys
import unittest
import logging
import time
import asyncio
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asf.orchestration.ray_orchestrator import (
    RayConfig, RayOrchestrator, RayTaskScheduler, RayWorker, Task, TaskStatus
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test-ray-orchestrator")

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

async def async_add(a, b):
    await asyncio.sleep(0.1)
    return a + b

def failing_function():
    raise ValueError("Test error")

class TestRayOrchestrator(unittest.TestCase):
    """Test cases for RayOrchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RayConfig(use_ray=False)
        
        self.orchestrator = RayOrchestrator(config=self.config)
        
        self.orchestrator.register_function(add, "add")
        self.orchestrator.register_function(multiply, "multiply")
        self.orchestrator.register_function(failing_function, "failing_function")
    
    def test_create_task(self):
        """Test task creation."""
        # Create task
        task_id = self.orchestrator.create_task(
            name="test_task",
            function_name="add",
            args=[1, 2],
            kwargs={},
            timeout=10,
            max_retries=2,
            priority=1
        )
        
        # Check task exists
        self.assertIn(task_id, self.orchestrator.tasks)
        
        # Check task properties
        task = self.orchestrator.tasks[task_id]
        self.assertEqual(task.name, "test_task")
        self.assertEqual(task.function_name, "add")
        self.assertEqual(task.args, [1, 2])
        self.assertEqual(task.kwargs, {})
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.timeout, 10)
        self.assertEqual(task.max_retries, 2)
        self.assertEqual(task.priority, 1)
    
    def test_execute_task(self):
        """Test task execution."""
        task_id = self.orchestrator.create_task(
            name="add_task",
            function_name="add",
            args=[1, 2]
        )
        
        result = self.orchestrator.execute_task(task_id)
        
        self.assertEqual(result, 3)
        
        task = self.orchestrator.tasks[task_id]
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.result, 3)
        self.assertIsNotNone(task.start_time)
        self.assertIsNotNone(task.end_time)
    
    def test_execute_task_with_dependencies(self):
        """Test task execution with dependencies."""
        # Create first task
        task1_id = self.orchestrator.create_task(
            name="add_task",
            function_name="add",
            args=[1, 2]
        )
        
        # Create second task with dependency on first task
        task2_id = self.orchestrator.create_task(
            name="multiply_task",
            function_name="multiply",
            args=[3, 4],
            dependencies=[task1_id]
        )
        
        # Execute first task
        result1 = self.orchestrator.execute_task(task1_id)
        
        # Check result
        self.assertEqual(result1, 3)
        
        # Execute second task
        result2 = self.orchestrator.execute_task(task2_id)
        
        # Check result
        self.assertEqual(result2, 12)
        
        # Check task statuses
        task1 = self.orchestrator.tasks[task1_id]
        task2 = self.orchestrator.tasks[task2_id]
        self.assertEqual(task1.status, TaskStatus.COMPLETED)
        self.assertEqual(task2.status, TaskStatus.COMPLETED)
    
    def test_execute_task_with_failure(self):
        """Test task execution with failure."""
        task_id = self.orchestrator.create_task(
            name="failing_task",
            function_name="failing_function",
            max_retries=0
        )
        
        with self.assertRaises(ValueError):
            self.orchestrator.execute_task(task_id)
        
        task = self.orchestrator.tasks[task_id]
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIsNotNone(task.error)
        self.assertIn("Test error", task.error)
    
    def test_execute_task_with_retry(self):
        """Test task execution with retry."""
        # Create task with retry
        task_id = self.orchestrator.create_task(
            name="failing_task",
            function_name="failing_function",
            max_retries=2
        )
        
        # Execute task (should raise exception after retries)
        with self.assertRaises(ValueError):
            self.orchestrator.execute_task(task_id)
        
        # Check task status
        task = self.orchestrator.tasks[task_id]
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.retry_count, 2)
    
    def test_execute_workflow(self):
        """Test workflow execution."""
        task1_id = self.orchestrator.create_task(
            name="add_task",
            function_name="add",
            args=[1, 2]
        )
        
        task2_id = self.orchestrator.create_task(
            name="multiply_task",
            function_name="multiply",
            args=[3, 4],
            dependencies=[task1_id]
        )
        
        results = self.orchestrator.execute_workflow([task1_id, task2_id])
        
        self.assertEqual(results[task1_id], 3)
        self.assertEqual(results[task2_id], 12)
        
        task1 = self.orchestrator.tasks[task1_id]
        task2 = self.orchestrator.tasks[task2_id]
        self.assertEqual(task1.status, TaskStatus.COMPLETED)
        self.assertEqual(task2.status, TaskStatus.COMPLETED)
    
    def test_get_task(self):
        """Test getting a task."""
        # Create task
        task_id = self.orchestrator.create_task(
            name="test_task",
            function_name="add",
            args=[1, 2]
        )
        
        # Get task
        task = self.orchestrator.get_task(task_id)
        
        # Check task properties
        self.assertEqual(task.name, "test_task")
        self.assertEqual(task.function_name, "add")
        self.assertEqual(task.args, [1, 2])
    
    def test_get_task_result(self):
        """Test getting task result."""
        task_id = self.orchestrator.create_task(
            name="add_task",
            function_name="add",
            args=[1, 2]
        )
        self.orchestrator.execute_task(task_id)
        
        result = self.orchestrator.get_task_result(task_id)
        
        self.assertEqual(result, 3)
    
    def test_cancel_task(self):
        """Test canceling a task."""
        # Create task
        task_id = self.orchestrator.create_task(
            name="test_task",
            function_name="add",
            args=[1, 2]
        )
        
        # Cancel task
        result = self.orchestrator.cancel_task(task_id)
        
        # Check result
        self.assertTrue(result)
        
        # Check task status
        task = self.orchestrator.tasks[task_id]
        self.assertEqual(task.status, TaskStatus.CANCELED)

class TestRayTaskScheduler(unittest.TestCase):
    """Test cases for RayTaskScheduler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create config with Ray disabled
        self.config = RayConfig(use_ray=False)
        
        # Create orchestrator
        self.orchestrator = RayOrchestrator(config=self.config)
        
        # Register functions
        self.orchestrator.register_function(add, "add")
        self.orchestrator.register_function(multiply, "multiply")
        
        # Create scheduler
        self.scheduler = RayTaskScheduler(orchestrator=self.orchestrator)
    
    def test_schedule_task(self):
        """Test task scheduling."""
        task_id = self.scheduler.schedule_task(
            name="test_task",
            function_name="add",
            args=[1, 2],
            kwargs={},
            timeout=10,
            max_retries=2,
            priority=1
        )
        
        self.assertIn(task_id, self.orchestrator.tasks)
        
        self.assertIn(task_id, self.scheduler.scheduled_tasks)
        
        task = self.orchestrator.tasks[task_id]
        self.assertEqual(task.name, "test_task")
        self.assertEqual(task.function_name, "add")
        self.assertEqual(task.args, [1, 2])
        self.assertEqual(task.kwargs, {})
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.timeout, 10)
        self.assertEqual(task.max_retries, 2)
        self.assertEqual(task.priority, 1)

class TestAsyncRayOrchestrator(unittest.IsolatedAsyncioTestCase):
    """Test cases for async methods of RayOrchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RayConfig(use_ray=False)
        
        self.orchestrator = RayOrchestrator(config=self.config)
        
        self.orchestrator.register_function(add, "add")
        self.orchestrator.register_function(multiply, "multiply")
        self.orchestrator.register_function(async_add, "async_add")
        self.orchestrator.register_function(failing_function, "failing_function")
    
    async def test_execute_task_async(self):
        task_id = self.orchestrator.create_task(
            name="async_add_task",
            function_name="async_add",
            args=[1, 2]
        )
        
        result = await self.orchestrator.execute_task_async(task_id)
        
        self.assertEqual(result, 3)
        
        task = self.orchestrator.tasks[task_id]
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.result, 3)
    
    async def test_execute_workflow_async(self):