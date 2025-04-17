"""
Tests for the performance enhancements in the Medical Research Synthesizer.

This module tests the parallel processing, batch processing, caching, and online learning
capabilities of the enhanced Medical Research Synthesizer.
"""

import os
import time
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from asf.medical.ml.document_processing import MedicalResearchSynthesizer
from asf.medical.ml.document_processing.document_structure import DocumentStructure, Entity

# Sample text for testing
SAMPLE_TEXT = """
# Test Medical Paper

## Abstract
This is a test abstract for the medical paper.

## Introduction
This is the introduction section.

## Methods
These are the methods used in the study.

## Results
These are the results of the study.

## Discussion
This is the discussion of the results.

## Conclusion
This is the conclusion of the study.
"""


class TestPerformanceEnhancements(unittest.TestCase):
    """Test case for the performance enhancements in the Medical Research Synthesizer."""

    def setUp(self):
        """Set up the test case."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create mock components
        self.mock_document_processor = MagicMock()
        self.mock_entity_extractor = MagicMock()
        self.mock_relation_extractor = MagicMock()
        self.mock_summarizer = MagicMock()
        
        # Create a sample document structure
        self.doc_structure = DocumentStructure(
            title="Test Medical Paper",
            abstract="This is a test abstract for the medical paper.",
            sections=[],
            entities=[
                Entity(text="test", label="TEST", start=0, end=4)
            ],
            relations=[{"head": "test", "tail": "paper", "relation": "TEST"}],
            summary={"abstract": "Test summary"}
        )
        
        # Configure mocks
        self.mock_document_processor.process_document.return_value = self.doc_structure
        self.mock_entity_extractor.process_document.return_value = self.doc_structure
        self.mock_relation_extractor.process_document.return_value = self.doc_structure
        self.mock_summarizer.process_document.return_value = self.doc_structure
        
        # Create patch objects
        self.document_processor_patch = patch(
            'asf.medical.ml.document_processing.medical_research_synthesizer.BiomedicalDocumentProcessor',
            return_value=self.mock_document_processor
        )
        self.entity_extractor_patch = patch(
            'asf.medical.ml.document_processing.medical_research_synthesizer.GLiNERBiomedExtractor',
            return_value=self.mock_entity_extractor
        )
        self.relation_extractor_patch = patch(
            'asf.medical.ml.document_processing.medical_research_synthesizer.HGTRelationExtractor',
            return_value=self.mock_relation_extractor
        )
        self.summarizer_patch = patch(
            'asf.medical.ml.document_processing.medical_research_synthesizer.EnhancedResearchSummarizer',
            return_value=self.mock_summarizer
        )
        
        # Start patches
        self.document_processor_patch.start()
        self.entity_extractor_patch.start()
        self.relation_extractor_patch.start()
        self.summarizer_patch.start()
        
        # Create synthesizer
        self.synthesizer = MedicalResearchSynthesizer(
            use_cache=True,
            cache_dir=self.cache_dir,
            cache_size_mb=10
        )

    def tearDown(self):
        """Tear down the test case."""
        # Stop patches
        self.document_processor_patch.stop()
        self.entity_extractor_patch.stop()
        self.relation_extractor_patch.stop()
        self.summarizer_patch.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_caching(self):
        """Test that caching works correctly."""
        # First call should not use cache
        result1, metrics1 = self.synthesizer.process(SAMPLE_TEXT)
        
        # Second call should use cache
        result2, metrics2 = self.synthesizer.process(SAMPLE_TEXT)
        
        # Check that cache was used
        self.assertTrue(metrics2.get('cache_hit', False))
        self.assertLess(metrics2['total_processing_time'], metrics1['total_processing_time'])
        
        # Check that results are the same
        self.assertEqual(result1.title, result2.title)
        self.assertEqual(len(result1.entities), len(result2.entities))

    def test_parallel_processing(self):
        """Test that parallel processing works correctly."""
        # Process with standard method
        start_time = time.time()
        result1, metrics1 = self.synthesizer.process(SAMPLE_TEXT)
        standard_time = time.time() - start_time
        
        # Process with parallel method
        start_time = time.time()
        result2, metrics2 = self.synthesizer.process_parallel(SAMPLE_TEXT)
        parallel_time = time.time() - start_time
        
        # Check that results are the same
        self.assertEqual(result1.title, result2.title)
        self.assertEqual(len(result1.entities), len(result2.entities))
        
        # Check that parallel processing was called
        self.mock_entity_extractor.process_document.assert_called()
        self.mock_relation_extractor.process_document.assert_called()

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        # Create test files
        file_paths = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"test_{i}.txt")
            with open(file_path, "w") as f:
                f.write(SAMPLE_TEXT)
            file_paths.append(file_path)
        
        # Process batch
        batch_metrics = self.synthesizer.process_batch(
            file_list=file_paths,
            output_dir=self.output_dir,
            batch_size=2,
            all_pdfs=False
        )
        
        # Check batch metrics
        self.assertEqual(batch_metrics['total_documents'], 3)
        self.assertEqual(batch_metrics['successful'], 3)
        self.assertEqual(batch_metrics['failed'], 0)
        
        # Check that output files were created
        for i in range(3):
            output_dir = os.path.join(self.output_dir, f"test_{i}")
            self.assertTrue(os.path.exists(output_dir))

    def test_online_learning(self):
        """Test that online learning works correctly."""
        # Create labeled data
        labeled_data = {
            "entities": [
                {"text": "test", "label": "TEST", "start": 0, "end": 4}
            ],
            "relations": [
                {"head": "test", "tail": "paper", "relation": "TEST"}
            ]
        }
        
        # Configure mocks for update_model
        self.mock_entity_extractor.update_model.return_value = {"loss": 0.1}
        self.mock_relation_extractor.update_model.return_value = {"loss": 0.2}
        
        # Update models
        update_metrics = self.synthesizer.update_models(
            labeled_data=labeled_data,
            learning_rate=1e-5,
            batch_size=2,
            epochs=1
        )
        
        # Check that update_model was called
        self.mock_entity_extractor.update_model.assert_called_once()
        self.mock_relation_extractor.update_model.assert_called_once()
        
        # Check update metrics
        self.assertIn("entity_extractor", update_metrics)
        self.assertIn("relation_extractor", update_metrics)
        self.assertEqual(update_metrics["entity_extractor"]["loss"], 0.1)
        self.assertEqual(update_metrics["relation_extractor"]["loss"], 0.2)


if __name__ == '__main__':
    unittest.main()
