"""
Test script for the progress tracking system in the LLM Gateway.

This script tests the progress tracking system, including the tracker,
registry, and decorators, to ensure they work correctly.
"""

import time
import asyncio
import unittest
from unittest.mock import MagicMock, patch

from .tracker import ProgressTracker, ProgressState
from .registry import ProgressRegistry, get_progress_registry
from .decorators import track_progress, track_llm_progress, get_progress_tracker
from .models import OperationType


class TestProgressTracker(unittest.TestCase):
    """Test the ProgressTracker class."""
    
    def test_initialization(self):
        """Test initialization of the progress tracker."""
        tracker = ProgressTracker(
            operation_id="test-init",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        self.assertEqual(tracker.operation_id, "test-init")
        self.assertEqual(tracker.operation_type, OperationType.GENERAL)
        self.assertEqual(tracker.total_steps, 5)
        self.assertEqual(tracker.current_step, 0)
        self.assertEqual(tracker.status, ProgressState.PENDING)
    
    def test_update(self):
        """Test updating the progress tracker."""
        tracker = ProgressTracker(
            operation_id="test-update",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        # Update to step 2
        tracker.update(2, "Step 2 completed")
        
        self.assertEqual(tracker.current_step, 2)
        self.assertEqual(tracker.message, "Step 2 completed")
        self.assertEqual(tracker.status, ProgressState.RUNNING)
        self.assertEqual(len(tracker.steps), 2)  # Initial step + update
    
    def test_complete(self):
        """Test completing the progress tracker."""
        tracker = ProgressTracker(
            operation_id="test-complete",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        # Update to step 3
        tracker.update(3, "Step 3 completed")
        
        # Complete
        tracker.complete("All steps completed")
        
        self.assertEqual(tracker.current_step, 5)
        self.assertEqual(tracker.message, "All steps completed")
        self.assertEqual(tracker.status, ProgressState.COMPLETED)
        self.assertIsNotNone(tracker.end_time)
    
    def test_fail(self):
        """Test failing the progress tracker."""
        tracker = ProgressTracker(
            operation_id="test-fail",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        # Update to step 2
        tracker.update(2, "Step 2 completed")
        
        # Fail
        tracker.fail("Operation failed")
        
        self.assertEqual(tracker.current_step, 2)
        self.assertEqual(tracker.message, "Operation failed")
        self.assertEqual(tracker.status, ProgressState.FAILED)
        self.assertIsNotNone(tracker.end_time)
    
    def test_get_progress_details(self):
        """Test getting progress details."""
        tracker = ProgressTracker(
            operation_id="test-details",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        # Update to step 3
        tracker.update(3, "Step 3 completed")
        
        # Get details
        details = tracker.get_progress_details()
        
        self.assertEqual(details.operation_id, "test-details")
        self.assertEqual(details.operation_type, OperationType.GENERAL)
        self.assertEqual(details.total_steps, 5)
        self.assertEqual(details.current_step, 3)
        self.assertEqual(details.status, ProgressState.RUNNING)
        self.assertEqual(details.message, "Step 3 completed")
        self.assertEqual(details.percent_complete, 60.0)
        self.assertEqual(len(details.steps), 2)  # Initial step + update
    
    def test_get_percent_complete(self):
        """Test calculating percent complete."""
        tracker = ProgressTracker(
            operation_id="test-percent",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        # Test different steps
        self.assertEqual(tracker.get_percent_complete(), 0.0)
        
        tracker.update(1, "Step 1")
        self.assertEqual(tracker.get_percent_complete(), 20.0)
        
        tracker.update(2, "Step 2")
        self.assertEqual(tracker.get_percent_complete(), 40.0)
        
        tracker.update(5, "Step 5")
        self.assertEqual(tracker.get_percent_complete(), 100.0)


class TestProgressRegistry(unittest.TestCase):
    """Test the ProgressRegistry class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a fresh registry for each test
        self.registry = ProgressRegistry()
        
        # Clear existing trackers
        self.registry.trackers.clear()
    
    def test_singleton(self):
        """Test that the registry is a singleton."""
        registry1 = ProgressRegistry()
        registry2 = ProgressRegistry()
        
        self.assertIs(registry1, registry2)
    
    def test_create_tracker(self):
        """Test creating a tracker through the registry."""
        tracker = self.registry.create_tracker(
            operation_id="test-create",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        self.assertEqual(tracker.operation_id, "test-create")
        self.assertEqual(len(self.registry.trackers), 1)
        self.assertIn("test-create", self.registry.trackers)
    
    def test_get_tracker(self):
        """Test getting a tracker from the registry."""
        # Create a tracker
        self.registry.create_tracker(
            operation_id="test-get",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        # Get the tracker
        tracker = self.registry.get_tracker("test-get")
        
        self.assertIsNotNone(tracker)
        self.assertEqual(tracker.operation_id, "test-get")
    
    def test_unregister(self):
        """Test unregistering a tracker."""
        # Create a tracker
        self.registry.create_tracker(
            operation_id="test-unregister",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        # Unregister the tracker
        self.registry.unregister("test-unregister")
        
        self.assertEqual(len(self.registry.trackers), 0)
        self.assertNotIn("test-unregister", self.registry.trackers)
    
    def test_get_all_trackers(self):
        """Test getting all trackers."""
        # Create some trackers
        self.registry.create_tracker(
            operation_id="test-all-1",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        self.registry.create_tracker(
            operation_id="test-all-2",
            operation_type=OperationType.LLM_REQUEST,
            total_steps=3
        )
        
        # Get all trackers
        trackers = self.registry.get_all_trackers()
        
        self.assertEqual(len(trackers), 2)
    
    def test_get_active_trackers(self):
        """Test getting active trackers."""
        # Create some trackers
        tracker1 = self.registry.create_tracker(
            operation_id="test-active-1",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        tracker2 = self.registry.create_tracker(
            operation_id="test-active-2",
            operation_type=OperationType.LLM_REQUEST,
            total_steps=3
        )
        
        # Complete one tracker
        tracker1.complete("Completed")
        
        # Get active trackers
        active_trackers = self.registry.get_active_trackers()
        
        self.assertEqual(len(active_trackers), 1)
        self.assertEqual(active_trackers[0].operation_id, "test-active-2")
    
    def test_cleanup(self):
        """Test cleaning up old trackers."""
        # Create some trackers
        tracker1 = self.registry.create_tracker(
            operation_id="test-cleanup-1",
            operation_type=OperationType.GENERAL,
            total_steps=5
        )
        
        tracker2 = self.registry.create_tracker(
            operation_id="test-cleanup-2",
            operation_type=OperationType.LLM_REQUEST,
            total_steps=3
        )
        
        # Complete one tracker and set end_time to a long time ago
        tracker1.complete("Completed")
        tracker1.end_time = time.time() - 7200  # 2 hours ago
        
        # Clean up trackers older than 1 hour
        removed = self.registry.cleanup(max_age_seconds=3600)
        
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.registry.trackers), 1)
        self.assertNotIn("test-cleanup-1", self.registry.trackers)
        self.assertIn("test-cleanup-2", self.registry.trackers)


class TestDecorators(unittest.TestCase):
    """Test the progress tracking decorators."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a fresh registry for each test
        self.registry = get_progress_registry()
        
        # Clear existing trackers
        self.registry.trackers.clear()
    
    def test_track_progress_sync(self):
        """Test the track_progress decorator with a synchronous function."""
        @track_progress(operation_type=OperationType.GENERAL, total_steps=3)
        def test_func(arg1, arg2=None):
            # Get the current tracker
            tracker = get_progress_tracker()
            
            # Update progress
            tracker.update(1, "Step 1")
            tracker.update(2, "Step 2")
            
            return arg1 + (arg2 or 0)
        
        # Call the function
        result = test_func(1, 2)
        
        # Check the result
        self.assertEqual(result, 3)
        
        # Check that a tracker was created
        self.assertEqual(len(self.registry.trackers), 1)
        
        # Get the tracker
        tracker = list(self.registry.trackers.values())[0]
        
        # Check tracker state
        self.assertEqual(tracker.status, ProgressState.COMPLETED)
        self.assertEqual(tracker.current_step, 3)
    
    @patch('asyncio.create_task')
    async def test_track_progress_async(self, mock_create_task):
        """Test the track_progress decorator with an asynchronous function."""
        @track_progress(operation_type=OperationType.GENERAL, total_steps=3)
        async def test_func(arg1, arg2=None):
            # Get the current tracker
            tracker = get_progress_tracker()
            
            # Update progress
            tracker.update(1, "Step 1")
            await asyncio.sleep(0.1)
            
            tracker.update(2, "Step 2")
            await asyncio.sleep(0.1)
            
            return arg1 + (arg2 or 0)
        
        # Call the function
        result = await test_func(1, 2)
        
        # Check the result
        self.assertEqual(result, 3)
        
        # Check that a tracker was created
        self.assertEqual(len(self.registry.trackers), 1)
        
        # Get the tracker
        tracker = list(self.registry.trackers.values())[0]
        
        # Check tracker state
        self.assertEqual(tracker.status, ProgressState.COMPLETED)
        self.assertEqual(tracker.current_step, 3)
    
    @patch('asyncio.create_task')
    async def test_track_llm_progress(self, mock_create_task):
        """Test the track_llm_progress decorator."""
        @track_llm_progress(operation_type=OperationType.LLM_REQUEST, total_steps=3)
        async def test_llm_func(prompt, model="gpt-4", temperature=0.7):
            # Get the current tracker
            tracker = get_progress_tracker()
            
            # Update progress
            tracker.update(1, "Preparing request")
            await asyncio.sleep(0.1)
            
            tracker.update(2, "Sending request")
            await asyncio.sleep(0.1)
            
            return {"response": f"Response to: {prompt}"}
        
        # Call the function
        result = await test_llm_func("Hello", model="gpt-3.5-turbo")
        
        # Check the result
        self.assertEqual(result["response"], "Response to: Hello")
        
        # Check that a tracker was created
        self.assertEqual(len(self.registry.trackers), 1)
        
        # Get the tracker
        tracker = list(self.registry.trackers.values())[0]
        
        # Check tracker state
        self.assertEqual(tracker.status, ProgressState.COMPLETED)
        self.assertEqual(tracker.current_step, 3)
        
        # Check that metadata was extracted
        self.assertIn("model", tracker.metadata)
        self.assertEqual(tracker.metadata["model"], "gpt-3.5-turbo")
        self.assertIn("prompt", tracker.metadata)
        self.assertEqual(tracker.metadata["prompt"], "Hello")
    
    def test_error_handling(self):
        """Test error handling in tracked functions."""
        @track_progress(operation_type=OperationType.GENERAL, total_steps=3)
        def test_error_func():
            # Get the current tracker
            tracker = get_progress_tracker()
            
            # Update progress
            tracker.update(1, "Step 1")
            
            # Raise an error
            raise ValueError("Test error")
        
        # Call the function and expect an error
        with self.assertRaises(ValueError):
            test_error_func()
        
        # Check that a tracker was created
        self.assertEqual(len(self.registry.trackers), 1)
        
        # Get the tracker
        tracker = list(self.registry.trackers.values())[0]
        
        # Check tracker state
        self.assertEqual(tracker.status, ProgressState.FAILED)
        self.assertEqual(tracker.current_step, 1)
        self.assertIn("Test error", tracker.message)


if __name__ == "__main__":
    unittest.main()
