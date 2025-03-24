
import asyncio
import time
import unittest
import logging
import numpy as np
from asf.layer4_environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler
from asf.layer4_environmental_coupling.components.enhanced_bayesian_updater import EnhancedBayesianUpdater

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestPrecisionWeighting(unittest.TestCase):
    
    def setUp(self):
        self.modeler = PredictiveEnvironmentalModeler()
        self.bayesian_updater = EnhancedBayesianUpdater()
        
    def test_precision_calculation(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_precision_calculation())
        
    async def _async_test_precision_calculation(self):
        # Test entity and coupling
        entity_id = "test_entity_precision"
        coupling_id = "test_coupling_precision"
        
        # Initialize Bayesian model
        await self.bayesian_updater.initialize_coupling_model(coupling_id)
        
        # Generate several predictions with varying accuracy
        initial_precision = self.modeler.precision.get(entity_id, 1.0)
        print(f"Initial precision: {initial_precision}")
        
        # First set of accurate predictions
        for i in range(5):
            prediction = await self.modeler.predict_interaction(entity_id)
            
            # Create actual data with low error (high accuracy)
            actual_data = {
                "interaction_type": prediction.predicted_data['predicted_interaction_type'],
                "content_type": "text",
                "timestamp": prediction.predicted_data['predicted_timing'] + 5,  # Only 5 seconds off
                "value": f"test data {i}"
            }
            
            # Evaluate prediction
            evaluation = await self.modeler.evaluate_prediction(prediction.id, actual_data)
            
            # Update Bayesian model with precision
            await self.bayesian_updater.update_precision(coupling_id, evaluation['error'])
            
            print(f"Iteration {i} (accurate): Error={evaluation['error']:.3f}, Precision={evaluation['precision']:.3f}")
            
        # Get precision after accurate predictions
        mid_precision = self.modeler.precision.get(entity_id, 1.0)
        print(f"Precision after accurate predictions: {mid_precision}")
        self.assertGreater(mid_precision, initial_precision, "Precision should increase after accurate predictions")
        
        # Second set of inaccurate predictions
        for i in range(5):
            prediction = await self.modeler.predict_interaction(entity_id)
            
            # Create actual data with high error (low accuracy)
            actual_data = {
                "interaction_type": "unexpected_type",  # Wrong type
                "content_type": "binary",  # Wrong content
                "timestamp": prediction.predicted_data['predicted_timing'] + 60,  # 60 seconds off
                "value": f"unexpected data {i}"
            }
            
            # Evaluate prediction
            evaluation = await self.modeler.evaluate_prediction(prediction.id, actual_data)
            
            # Update Bayesian model with precision
            await self.bayesian_updater.update_precision(coupling_id, evaluation['error'])
            
            print(f"Iteration {i} (inaccurate): Error={evaluation['error']:.3f}, Precision={evaluation['precision']:.3f}")
            
        # Get final precision after inaccurate predictions
        final_precision = self.modeler.precision.get(entity_id, 1.0)
        print(f"Final precision: {final_precision}")
        self.assertLess(final_precision, mid_precision, "Precision should decrease after inaccurate predictions")
        
    def test_precision_weighted_bayesian_updating(self):
        # Run async test in event loop
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._async_test_precision_weighted_bayesian_updating())
        
    async def _async_test_precision_weighted_bayesian_updating(self):
        # Test coupling
        coupling_id = "test_coupling_bayes"
        
        # Initialize Bayesian model
        await self.bayesian_updater.initialize_coupling_model(coupling_id)
        
        # Test multiple interactions with varying precision
        precisions = [0.5, 1.0, 2.0, 5.0]
        
        for precision in precisions:
            # Manually set precision
            self.bayesian_updater.precision_values[coupling_id] = precision
            
            # Create interaction data
            interaction_data = {
                "interaction_type": "query",
                "content": "test content",
                "timestamp": time.time()
            }
            
            # Standard confidence
            confidence = 0.7
            
            # Update Bayesian model
            result = await self.bayesian_updater.update_from_interaction(
                coupling_id, interaction_data, "query", confidence
            )
            
            # Print results
            print(f"Precision {precision}:")
            print(f"  Prior: {result['prior_confidence']:.4f}")
            print(f"  Weighted confidence: {result['weighted_confidence']:.4f}")
            print(f"  Posterior: {result['new_confidence']:.4f}")
            print(f"  Strength delta: {result['strength_delta']:.4f}")
            
            # Verify that higher precision leads to larger confidence changes
            if precision > 1.0:
                self.assertGreater(
                    abs(result['strength_delta']), 
                    0.05, 
                    "Higher precision should lead to larger confidence changes"
                )

if __name__ == "__main__":
    unittest.main()
