
import asyncio
import time
import logging
import random
import uuid
import json
from dataclasses import asdict

from asf.layer4_environmental_coupling.components.counterfactual_simulator import CounterfactualSimulator
from asf.layer4_environmental_coupling.models import EnvironmentalCoupling
from asf.layer4_environmental_coupling.enums import CouplingType, CouplingStrength, CouplingState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CounterfactualDemo")

async def run_counterfactual_demonstration():
    """Run a demonstration of counterfactual simulation capabilities."""
    logger.info("Starting Counterfactual Simulation Demonstration")
    
    # Create simulator
    simulator = CounterfactualSimulator()
    
    # Create different types of couplings to test
    coupling_types = [
        ('informational', CouplingType.INFORMATIONAL),
        ('operational', CouplingType.OPERATIONAL),
        ('adaptive', CouplingType.ADAPTIVE),
        ('predictive', CouplingType.PREDICTIVE)
    ]
    
    for coupling_name, coupling_type in coupling_types:
        logger.info(f"\n{'=' * 40}\nTesting {coupling_name.upper()} coupling\n{'=' * 40}")
        
        # Create test coupling
        coupling = EnvironmentalCoupling(
            id=f"coupling_{coupling_name}_{int(time.time())}",
            internal_entity_id=f"internal_{coupling_name}",
            environmental_entity_id=f"external_{coupling_name}",
            coupling_type=coupling_type,
            coupling_strength=0.6,
            coupling_state=CouplingState.ACTIVE,
            bayesian_confidence=0.5,
            interaction_count=random.randint(5, 20)
        )
        
        # Add properties based on coupling type
        if coupling_type == CouplingType.INFORMATIONAL:
            coupling.properties = {
                'reliability': 0.7,
                'refresh_interval': 60
            }
        elif coupling_type == CouplingType.OPERATIONAL:
            coupling.properties = {
                'response_time': 1.2,
                'throughput': 50
            }
        elif coupling_type == CouplingType.ADAPTIVE:
            coupling.properties = {
                'learning_rate': 0.05,
                'adaptation_threshold': 0.3
            }
        elif coupling_type == CouplingType.PREDICTIVE:
            coupling.properties = {
                'prediction_horizon': 300,
                'min_confidence': 0.4
            }
        
        # Generate variations
        variations_count = 5
        logger.info(f"Generating {variations_count} counterfactual variations")
        
        variations = await simulator.generate_coupling_variations(coupling, variations_count)
        
        # Log variations
        for i, variation in enumerate(variations):
            logger.info(f"Variation {i+1}: {variation['variation_type']} - {variation['description']}")
            
            # Log key changes
            if 'coupling_type' in variation and variation['coupling_type'] != coupling.coupling_type:
                logger.info(f"  Changed type: {coupling.coupling_type} -> {variation['coupling_type']}")
            
            if 'coupling_strength' in variation and variation['coupling_strength'] != coupling.coupling_strength:
                logger.info(f"  Changed strength: {coupling.coupling_strength:.2f} -> {variation['coupling_strength']:.2f}")
            
            if 'properties' in variation:
                for key, value in variation['properties'].items():
                    if key in coupling.properties and coupling.properties[key] != value:
                        logger.info(f"  Changed property {key}: {coupling.properties[key]} -> {value}")
                    elif key not in coupling.properties:
                        logger.info(f"  Added property {key}: {value}")
        
        # Simulate outcomes
        logger.info("\nSimulating outcomes for variations")
        simulation_results = await simulator.simulate_outcomes(variations)
        
        # Log simulation results
        for i, result in enumerate(simulation_results):
            score = simulator._calculate_simulation_score(result['outcome'])
            logger.info(f"Variation {i+1} score: {score:.3f}")
            
            # Log key metrics
            outcome = result['outcome']
            logger.info(f"  Success rate: {outcome['success_rate']:.2f}")
            logger.info(f"  Efficiency: {outcome['efficiency']:.2f}")
            logger.info(f"  Response time: {outcome['response_time']:.2f}s")
            logger.info(f"  Prediction precision: {outcome['prediction_precision']:.2f}")
        
        # Identify optimal configuration
        logger.info("\nIdentifying optimal configuration")
        optimal = await simulator.identify_optimal_configuration(simulation_results)
        
        # Log optimal configuration
        opt_config = optimal['optimal_configuration']
        logger.info(f"Optimal configuration: {opt_config['variation_type']} - {opt_config['description']}")
        logger.info(f"Improvement over alternatives: {optimal['improvement']:.2f}x")
        
        # Simulate applying the configuration
        logger.info("\nSimulating application of optimal configuration")
        
        # Create a copy of the original coupling for comparison
        original_coupling_dict = {
            'type': coupling.coupling_type.name,
            'strength': coupling.coupling_strength,
            'properties': dict(coupling.properties)
        }
        
        # Apply changes from optimal configuration to the coupling
        if 'coupling_type' in opt_config:
            coupling.coupling_type = opt_config['coupling_type']
        
        if 'coupling_strength' in opt_config:
            coupling.coupling_strength = opt_config['coupling_strength']
        
        if 'properties' in opt_config:
            for key, value in opt_config['properties'].items():
                coupling.properties[key] = value
        
        # Log the changes
        logger.info("Changes applied to coupling:")
        if original_coupling_dict['type'] != coupling.coupling_type.name:
            logger.info(f"  Type: {original_coupling_dict['type']} -> {coupling.coupling_type.name}")
        
        if original_coupling_dict['strength'] != coupling.coupling_strength:
            logger.info(f"  Strength: {original_coupling_dict['strength']:.2f} -> {coupling.coupling_strength:.2f}")
        
        for key, value in coupling.properties.items():
            if key in original_coupling_dict['properties']:
                if original_coupling_dict['properties'][key] != value:
                    logger.info(f"  Property {key}: {original_coupling_dict['properties'][key]} -> {value}")
            else:
                logger.info(f"  Added property {key}: {value}")
        
        # Simulate recording actual outcome
        logger.info("\nSimulating actual outcome after applying configuration")
        
        # Create simulated actual outcome (similar to predicted but with some differences)
        predicted_outcome = optimal['predicted_outcome']
        
        # Apply some noise to represent real-world variance
        actual_outcome = {}
        for key, value in predicted_outcome.items():
            if key not in ['simulation_time', 'simulation_id', 'model_confidence']:
                # Add random noise ±15%
                noise_factor = 1.0 + random.uniform(-0.15, 0.15)
                actual_outcome[key] = value * noise_factor
        
        # Ensure values are in valid ranges
        for key in ['success_rate', 'efficiency', 'reliability', 'adaptability']:
            if key in actual_outcome:
                actual_outcome[key] = min(1.0, max(0.0, actual_outcome[key]))
        
        if 'response_time' in actual_outcome:
            actual_outcome['response_time'] = max(0.1, actual_outcome['response_time'])
        
        if 'prediction_precision' in actual_outcome:
            actual_outcome['prediction_precision'] = max(0.1, actual_outcome['prediction_precision'])
        
        # Record actual outcome
        await simulator.record_actual_outcome(
            coupling.id,
            {'predicted_outcome': predicted_outcome},
            actual_outcome
        )
        
        # Log actual vs predicted
        logger.info("Actual vs Predicted Outcome:")
        for key in ['success_rate', 'efficiency', 'response_time', 'prediction_precision', 'reliability']:
            if key in actual_outcome and key in predicted_outcome:
                logger.info(f"  {key}: {predicted_outcome[key]:.2f} (predicted) vs {actual_outcome[key]:.2f} (actual)")
        
        # Check model accuracy
        if coupling.id in simulator.simulation_models:
            model = simulator.simulation_models[coupling.id]
            logger.info(f"\nSimulation model accuracy: {model['accuracy']:.3f}")
            logger.info(f"Recorded outcomes: {len(model['outcomes'])}")
        
        logger.info(f"\nCompleted counterfactual simulation for {coupling_name.upper()} coupling")
        logger.info('-' * 80)

async def main():
    """Main demo function."""
    await run_counterfactual_demonstration()
    logger.info("Counterfactual demonstration completed")

if __name__ == "__main__":
    asyncio.run(main())
