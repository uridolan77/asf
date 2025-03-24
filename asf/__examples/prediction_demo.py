# === FILE: asf/examples/prediction_demo.py ===

import asyncio
import time
import logging
import random
from asf.layer4_environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PredictionDemo")

async def simulate_interactions(entity_id, count=10, modeler=None):
    """Simulate a sequence of interactions with an entity."""
    if modeler is None:
        modeler = PredictiveEnvironmentalModeler()
        
    interaction_types = ['query', 'update', 'notification', 'request']
    
    logger.info(f"Starting simulation with entity {entity_id} for {count} interactions")
    
    for i in range(count):
        # Generate a prediction
        context = {
            "last_interaction_type": interaction_types[i % len(interaction_types)],
            "iteration": i
        }
        prediction = await modeler.predict_interaction(entity_id, context)
        
        logger.info(f"Generated prediction {prediction.id} with confidence {prediction.confidence:.2f}")
        logger.info(f"Predicted interaction: {prediction.predicted_data['predicted_interaction_type']}")
        logger.info(f"Predicted timing: {time.ctime(prediction.predicted_data['predicted_timing'])}")
        
        # Simulate delay before actual interaction
        wait_time = random.uniform(0.5, 2.0)
        logger.info(f"Waiting {wait_time:.2f} seconds for actual interaction...")
        await asyncio.sleep(wait_time)
        
        # Generate actual interaction data (sometimes matching, sometimes not)
        prediction_correct = random.random() > 0.3  # 70% chance of correct prediction
        
        actual_data = {
            "interaction_type": prediction.predicted_data['predicted_interaction_type'] if prediction_correct else random.choice(interaction_types),
            "content_type": prediction.predicted_data['predicted_content'].get('type', 'text'),
            "timestamp": time.time(),
            "value": f"test data {i}"
        }
        
        # Evaluate prediction
        evaluation = await modeler.evaluate_prediction(prediction.id, actual_data)
        
        logger.info(f"Evaluated prediction - Error: {evaluation['error']:.4f}, Precision: {evaluation['precision']:.4f}")
        logger.info("-" * 60)
        
    # Show final statistics
    logger.info(f"Simulation completed for entity {entity_id}")
    logger.info(f"Final precision: {modeler.precision.get(entity_id, 0):.4f}")
    logger.info(f"Total predictions: {len(modeler.entity_predictions.get(entity_id, []))}")
    
async def main():
    """Main demo function."""
    # Create the modeler
    modeler = PredictiveEnvironmentalModeler()
    
    # Simulate interactions for a few entities
    await simulate_interactions("entity_001", count=5, modeler=modeler)
    await simulate_interactions("entity_002", count=10, modeler=modeler)
    
    # Verify we can retrieve predictions for specific entities
    entity_predictions = await modeler.get_predictions_for_entity("entity_001", limit=10)
    logger.info(f"Retrieved {len(entity_predictions)} predictions for entity_001")
    
    entity_predictions = await modeler.get_predictions_for_entity("entity_002", limit=10)
    logger.info(f"Retrieved {len(entity_predictions)} predictions for entity_002")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
