# === FILE: asf/tests/test_prediction_framework.py ===

import asyncio
import time
import unittest
import logging
from asf.environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestPredictionFramework(unittest.TestCase):
    
    def setUp(self):
        self.modeler = PredictiveEnvironmentalModeler()
        
    def test_prediction_generation(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_prediction_generation())
        
    async def _async_test_prediction_generation(self):
        # Test entity
        entity_id = "test_entity_1"
        
        # Generate a prediction
        context = {"last_interaction_type": "query"}
        prediction = await self.modeler.predict_interaction(entity_id, context)
        
        # Validate prediction
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.environmental_entity_id, entity_id)
        self.assertGreaterEqual(prediction.confidence, 0.3)
        self.assertLessEqual(prediction.confidence, 0.9)
        
        # Check prediction structure
        self.assertIn('predicted_interaction_type', prediction.predicted_data)
        self.assertIn('predicted_content', prediction.predicted_data)
        self.assertIn('predicted_timing', prediction.predicted_data)
        
        # Validate it's stored properly
        self.assertIn(prediction.id, self.modeler.predictions)
        self.assertIn(prediction.id, self.modeler.entity_predictions[entity_id])
        
    def test_prediction_evaluation(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_prediction_evaluation())
        
    async def _async_test_prediction_evaluation(self):
        # Test entity
        entity_id = "test_entity_2"
        
        # Generate a prediction
        prediction = await self.modeler.predict_interaction(entity_id)
        
        # Create some actual data
        actual_data = {
            "interaction_type": prediction.predicted_data['predicted_interaction_type'],
            "content_type": "text",
            "timestamp": time.time(),
            "value": "test data"
        }
        
        # Evaluate prediction
        evaluation = await self.modeler.evaluate_prediction(prediction.id, actual_data)
        
        # Validate evaluation
        self.assertIsNotNone(evaluation)
        self.assertEqual(evaluation['prediction_id'], prediction.id)
        self.assertIn('error', evaluation)
        self.assertIn('precision', evaluation)
        
        # Check prediction was updated
        updated_prediction = self.modeler.predictions[prediction.id]
        self.assertIsNotNone(updated_prediction.verification_time)
        self.assertIsNotNone(updated_prediction.prediction_error)
        
        # Check precision updating
        self.assertIn(entity_id, self.modeler.precision)
        
    def test_precision_calculation(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_precision_calculation())
        
    async def _async_test_precision_calculation(self):
        # Test entity
        entity_id = "test_entity_3"
        
        # Generate multiple predictions with varying accuracy
        for i in range(10):
            prediction = await self.modeler.predict_interaction(entity_id)
            
            # Create actual data with controlled error
            error_level = 0.2 if i < 5 else 0.8  # First 5 accurate, next 5 inaccurate
            
            # Interaction type matches prediction half the time
            type_match = i % 2 == 0
            
            actual_data = {
                "interaction_type": prediction.predicted_data['predicted_interaction_type'] if type_match else "unexpected_type",
                "content_type": "text",
                "timestamp": prediction.predicted_data['predicted_timing'] + (5 if error_level < 0.5 else 30),
                "value": f"test data {i}"
            }
            
            # Evaluate prediction
            evaluation = await self.modeler.evaluate_prediction(prediction.id, actual_data)
            
            # Print current precision for debugging
            print(f"Iteration {i}: Error={evaluation['error']:.3f}, Precision={evaluation['precision']:.3f}")
            
        # Final precision should reflect the pattern of errors
        final_precision = self.modeler.precision.get(entity_id, 0)
        print(f"Final precision: {final_precision:.4f}")
        
        # Precision should be meaningful (not too high or too low)
        self.assertGreater(final_precision, 0.1)
        self.assertLess(final_precision, 10.0)

if __name__ == "__main__":
    unittest.main()
