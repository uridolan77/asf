
import asyncio
import time
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from asf.layer4_environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler
from asf.layer4_environmental_coupling.components.enhanced_bayesian_updater import EnhancedBayesianUpdater

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PrecisionWeightedDemo")

async def simulate_precision_weighted_interactions(entity_id, coupling_id, count=15):
    """Simulate a sequence of interactions with precision-weighted updates."""
    # Initialize components
    modeler = PredictiveEnvironmentalModeler()
    updater = EnhancedBayesianUpdater()
    
    # Initialize Bayesian model
    await updater.initialize_coupling_model(coupling_id)
    
    logger.info(f"Starting precision-weighted simulation for {count} interactions")
    logger.info(f"Entity ID: {entity_id}, Coupling ID: {coupling_id}")
    
    # Track metrics
    precisions = []
    confidences = []
    errors = []
    
    # Phases of the simulation
    # Phase 1: Predictable entity (low errors)
    # Phase 2: Unpredictable entity (high errors)
    # Phase 3: Return to predictable (low errors)
    
    phase_lengths = [5, 5, 5]  # 5 interactions per phase
    current_phase = 0
    phase_interaction = 0
    
    for i in range(count):
        # Determine current phase
        if phase_interaction >= phase_lengths[current_phase]:
            current_phase = (current_phase + 1) % len(phase_lengths)
            phase_interaction = 0
            logger.info(f"Switching to phase {current_phase + 1}")
            
        phase_interaction += 1
        
        # Generate prediction
        context = {"phase": current_phase, "iteration": i}
        prediction = await modeler.predict_interaction(entity_id, context)
        
        logger.info(f"Step {i+1}: Generated prediction with confidence {prediction.confidence:.2f}")
        
        # Wait a bit
        await asyncio.sleep(0.5)
        
        # Generate actual data with accuracy dependent on phase
        if current_phase == 0 or current_phase == 2:  # Predictable phases
            # Low error - type matches, timing close
            error_level = 0.2
            actual_data = {
                "interaction_type": prediction.predicted_data['predicted_interaction_type'],
                "content_type": "text",
                "timestamp": prediction.predicted_data['predicted_timing'] + random.uniform(0, 10),
                "value": f"accurate data {i}"
            }
        else:  # Unpredictable phase
            # High error - different type, timing way off
            error_level = 0.8
            actual_data = {
                "interaction_type": "unexpected_type",
                "content_type": "binary",
                "timestamp": prediction.predicted_data['predicted_timing'] + random.uniform(30, 60),
                "value": f"unpredictable data {i}"
            }
            
        # Evaluate prediction
        evaluation = await modeler.evaluate_prediction(prediction.id, actual_data)
        logger.info(f"Evaluated prediction - Error: {evaluation['error']:.4f}")
        
        # Update precision
        await updater.update_precision(coupling_id, evaluation['error'])
        current_precision = updater.precision_values.get(coupling_id, 1.0)
        logger.info(f"Updated precision: {current_precision:.4f}")
        
        # Update Bayesian model with precision-weighted confidence
        update_result = await updater.update_from_interaction(
            coupling_id, 
            actual_data, 
            actual_data["interaction_type"], 
            0.7  # Base confidence
        )
        
        logger.info(f"Bayesian update - Prior: {update_result['prior_confidence']:.4f}, " +
                     f"Posterior: {update_result['new_confidence']:.4f}, " +
                     f"Weighted confidence: {update_result['weighted_confidence']:.4f}")
        
        # Track metrics
        precisions.append(current_precision)
        confidences.append(update_result['new_confidence'])
        errors.append(evaluation['error'])
        
        logger.info("-" * 60)
    
    # Show final results
    logger.info(f"Simulation completed - Final metrics:")
    logger.info(f"Final precision: {precisions[-1]:.4f}")
    logger.info(f"Final confidence: {confidences[-1]:.4f}")
    logger.info(f"Average error: {np.mean(errors):.4f}")
    
    # Visualize results if matplotlib is available
    try:
       
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(precisions, 'g-', label='Precision')
        plt.title('Precision Evolution')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(confidences, 'b-', label='Bayesian Confidence')
        plt.title('Confidence Evolution')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(errors, 'r-', label='Prediction Errors')
        plt.title('Prediction Errors')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('precision_weighted_simulation.png')
        logger.info("Saved visualization to precision_weighted_simulation.png")
    except ImportError:
        logger.info("Matplotlib not available for visualization")
    
    return {
        "precisions": precisions,
        "confidences": confidences,
        "errors": errors
    }

async def main():
    """Main demo function."""
    entity_id = "demo_entity_001"
    coupling_id = "demo_coupling_001"
    
    # Run simulation
    results = await simulate_precision_weighted_interactions(entity_id, coupling_id, count=15)
    
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
