import asyncio
import time
import uuid
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from collections import defaultdict

class ReinforcementLearningOptimizer:
    """
    Enhanced reinforcement learning optimizer with active inference capabilities.
    Optimizes coupling parameters through reinforcement learning and counterfactual simulation.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.coupling_models = {}  # Maps coupling_id to RL models
        self.experience_buffer = defaultdict(list)  # Maps coupling_id to experiences
        self.neural_models = {}  # Maps coupling_id to neural networks
        self.model_parameters = {}  # Maps coupling_id to hyperparameters
        self.optimization_history = defaultdict(list)  # Maps coupling_id to optimization history
        
        # Seth's Data Paradox enhancements
        self.counterfactual_simulations = {}  # Maps coupling_id to simulated interactions
        self.active_inference_tests = {}  # Maps coupling_id to active inference test results
        self.predicted_outcomes = {}  # Maps coupling_id to predicted interaction outcomes
        
        self.logger = logging.getLogger("ASF.Layer4.ReinforcementLearningOptimizer")
        
    async def initialize(self, knowledge_substrate):
        """Initialize the RL optimizer."""
        self.knowledge_substrate = knowledge_substrate
        return True
        
    async def initialize_coupling_model(self, coupling_id, coupling):
        """Initialize RL model for a new coupling."""
        model_type = self._determine_model_type(coupling)
        
        self.coupling_models[coupling_id] = {
            'model_type': model_type,
            'state_space': self._define_state_space(coupling),
            'action_space': self._define_action_space(coupling),
            'q_values': {},  # Q-values for state-action pairs
            'last_updated': time.time(),
            'update_count': 0
        }
        
        # Set model parameters
        self.model_parameters[coupling_id] = {
            'learning_rate': self.learning_rate,
            'discount_factor': 0.9,
            'exploration_rate': 0.2,
            'batch_size': 32
        }
        
        # Initialize neural model for more complex couplings
        if model_type == 'neural':
            self._initialize_neural_model(coupling_id, coupling)
            
        # Initialize counterfactual simulation model
        self.counterfactual_simulations[coupling_id] = []
        
        return True
        
    async def update_from_interaction(self, coupling_id, interaction_data, interaction_type, bayesian_update):
        """
        Update RL model based on interaction outcome.
        """
        start_time = time.time()
        
        if coupling_id not in self.coupling_models:
            return {'status': 'model_not_found'}
            
        model = self.coupling_models[coupling_id]
        
        # Extract state and action from interaction
        state = self._extract_state(interaction_data)
        action = self._extract_action(interaction_data)
        
        # Calculate reward based on Bayesian update
        reward = self._calculate_reward(bayesian_update)
        
        # Determine new state
        new_state = self._extract_state({
            'interaction_type': interaction_type,
            'bayesian_confidence': bayesian_update.get('new_confidence', 0.5)
        })
        
        # Add to experience buffer
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'new_state': new_state,
            'timestamp': time.time()
        }
        self.experience_buffer[coupling_id].append(experience)
        
        # Limit buffer size
        if len(self.experience_buffer[coupling_id]) > 1000:
            self.experience_buffer[coupling_id] = self.experience_buffer[coupling_id][-1000:]
            
        # Update model
        if model['model_type'] == 'tabular':
            await self._update_tabular_model(coupling_id, experience)
        elif model['model_type'] == 'neural':
            await self._update_neural_model(coupling_id, experience)
            
        # Generate predicted outcomes for future interactions
        await self._generate_predictions(coupling_id, new_state)
        
        # Check if this was an active inference test
        active_inference_result = None
        if coupling_id in self.active_inference_tests:
            # Compare actual results to test expectations
            active_inference_result = await self._evaluate_active_inference_test(
                coupling_id, interaction_data, reward
            )
            
        # Record optimization step
        self.optimization_history[coupling_id].append({
            'timestamp': time.time(),
            'reward': reward,
            'bayesian_confidence': bayesian_update.get('new_confidence', 0.5),
            'exploration_rate': self.model_parameters[coupling_id]['exploration_rate']
        })
        
        # Adaptive exploration rate decay
        self.model_parameters[coupling_id]['exploration_rate'] *= 0.995
        
        return {
            'status': 'updated',
            'reward': reward,
            'model_type': model['model_type'],
            'experiences': len(self.experience_buffer[coupling_id]),
            'active_inference_result': active_inference_result,
            'elapsed_time': time.time() - start_time
        }
        
    async def get_optimal_parameters(self, entity, target_id, coupling):
        """
        Get optimal parameters for interacting with target.
        Uses RL model to determine best action in current state.
        """
        coupling_id = coupling.id
        
        if coupling_id not in self.coupling_models:
            return {}
            
        model = self.coupling_models[coupling_id]
        
        # Extract current state
        current_state = self._extract_state({
            'entity_type': getattr(entity, 'type', 'unknown'),
            'bayesian_confidence': coupling.bayesian_confidence
        })
        
        # Determine if we should explore or exploit
        if random.random() < self.model_parameters[coupling_id]['exploration_rate']:
            # Exploration: choose random action
            action = random.choice(list(model['action_space'].keys()))
        else:
            # Exploitation: choose best action
            action = await self._get_best_action(coupling_id, current_state)
            
        # Convert action to parameters
        parameters = self._action_to_parameters(action, coupling)
        
        # Add counterfactual thinking - consider alternative parameters
        counterfactual_params = await self._generate_counterfactual_parameters(
            coupling_id, current_state, parameters
        )
        
        # If counterfactual parameters are better, use them
        if counterfactual_params and counterfactual_params.get('predicted_reward', 0) > parameters.get('predicted_reward', 0):
            parameters = counterfactual_params
            
        return parameters
        
    async def _generate_counterfactual_parameters(self, coupling_id, state, base_parameters):
        """
        Generate counterfactual parameters to consider alternative actions.
        Implements Seth's counterfactual thinking principle.
        """
        if coupling_id not in self.coupling_models:
            return None
            
        model = self.coupling_models[coupling_id]
        
        # Get all possible actions except the one chosen
        all_actions = list(model['action_space'].keys())
        base_action = base_parameters.get('action_type', 'default')
        alternative_actions = [a for a in all_actions if a != base_action]
        
        if not alternative_actions:
            return None
            
        # Pick an alternative action
        alt_action = random.choice(alternative_actions)
        
        # Convert to parameters
        alt_parameters = model['action_space'][alt_action].copy()
        alt_parameters['action_type'] = alt_action
        
        # Predict reward for this alternative
        if model['model_type'] == 'tabular':
            state_key = self._state_to_key(state)
            alt_reward = model['q_values'].get((state_key, alt_action), 0)
        elif model['model_type'] == 'neural' and coupling_id in self.neural_models:
            nn_model = self.neural_models[coupling_id]
            state_tensor = torch.tensor(list(state.values()), dtype=torch.float32)
            with torch.no_grad():
                q_values = nn_model(state_tensor)
                action_idx = all_actions.index(alt_action)
                alt_reward = q_values[action_idx].item()
        else:
            alt_reward = 0
            
        alt_parameters['predicted_reward'] = alt_reward
        
        # Record counterfactual simulation
        self.counterfactual_simulations[coupling_id].append({
            'timestamp': time.time(),
            'state': state,
            'base_action': base_action,
            'alt_action': alt_action,
            'base_reward': base_parameters.get('predicted_reward', 0),
            'alt_reward': alt_reward
        })
        
        # Limit history
        if len(self.counterfactual_simulations[coupling_id]) > 100:
            self.counterfactual_simulations[coupling_id] = self.counterfactual_simulations[coupling_id][-100:]
            
        return alt_parameters
        
    async def setup_active_inference_test(self, coupling_id, test_parameters):
        """
        Set up an active inference test for a coupling.
        Implements Seth's active inference principle to test predictions.
        """
        if coupling_id not in self.coupling_models:
            return {'status': 'model_not_found'}
            
        # Record test setup
        test_id = str(uuid.uuid4())
        self.active_inference_tests[coupling_id] = {
            'test_id': test_id,
            'parameters': test_parameters,
            'setup_time': time.time(),
            'expected_reward': test_parameters.get('expected_reward', 0),
            'completed': False
        }
        
        return {
            'status': 'test_setup',
            'test_id': test_id,
            'coupling_id': coupling_id
        }
        
    async def _evaluate_active_inference_test(self, coupling_id, interaction_data, actual_reward):
        """
        Evaluate the results of an active inference test.
        """
        if coupling_id not in self.active_inference_tests:
            return None
            
        test = self.active_inference_tests[coupling_id]
        if test['completed']:
            return None
            
        # Calculate prediction error
        expected_reward = test['expected_reward']
        prediction_error = abs(actual_reward - expected_reward)
        
        # Mark test as completed
        test['completed'] = True
        test['completion_time'] = time.time()
        test['actual_reward'] = actual_reward
        test['prediction_error'] = prediction_error
        
        # Calculate information gain
        information_gain = 1.0 / (1.0 + prediction_error)
        test['information_gain'] = information_gain
        
        return {
            'test_id': test['test_id'],
            'expected_reward': expected_reward,
            'actual_reward': actual_reward,
            'prediction_error': prediction_error,
            'information_gain': information_gain
        }
        
    async def _generate_predictions(self, coupling_id, current_state):
        """
        Generate predictions about future interaction outcomes.
        Implements Seth's controlled hallucination principle.
        """
        if coupling_id not in self.coupling_models:
            return
            
        model = self.coupling_models[coupling_id]
        
        # Predict outcomes for all possible actions
        predictions = {}
        for action in model['action_space'].keys():
            # Predict reward for this action
            if model['model_type'] == 'tabular':
                state_key = self._state_to_key(current_state)
                reward = model['q_values'].get((state_key, action), 0)
            elif model['model_type'] == 'neural' and coupling_id in self.neural_models:
                nn_model = self.neural_models[coupling_id]
                state_tensor = torch.tensor(list(current_state.values()), dtype=torch.float32)
                with torch.no_grad():
                    q_values = nn_model(state_tensor)
                    action_idx = list(model['action_space'].keys()).index(action)
                    reward = q_values[action_idx].item()
            else:
                reward = 0
                
            predictions[action] = {
                'predicted_reward': reward,
                'confidence': min(1.0, model['update_count'] / 100),  # Confidence grows with experience
                'generation_time': time.time()
            }
            
        # Store predictions
        self.predicted_outcomes[coupling_id] = predictions
        
    async def _get_best_action(self, coupling_id, state):
        """Get best action for a given state."""
        model = self.coupling_models[coupling_id]
        
        if model['model_type'] == 'tabular':
            state_key = self._state_to_key(state)
            # Find action with highest Q-value
            best_action = None
            best_value = float('-inf')
            
            for (s, a), value in model['q_values'].items():
                if s == state_key and value > best_value:
                    best_action = a
                    best_value = value
                    
            # If no action found, choose random
            if best_action is None:
                best_action = random.choice(list(model['action_space'].keys()))
                
            return best_action
            
        elif model['model_type'] == 'neural' and coupling_id in self.neural_models:
            nn_model = self.neural_models[coupling_id]
            state_tensor = torch.tensor(list(state.values()), dtype=torch.float32)
            
            with torch.no_grad():
                q_values = nn_model(state_tensor)
                best_idx = torch.argmax(q_values).item()
                
            # Convert index to action
            actions = list(model['action_space'].keys())
            return actions[best_idx]
            
        # Fallback to random action
        return random.choice(list(model['action_space'].keys()))
        
    async def _update_tabular_model(self, coupling_id, experience):
        """Update tabular Q-learning model."""
        model = self.coupling_models[coupling_id]
        params = self.model_parameters[coupling_id]
        
        # Convert states to keys
        state_key = self._state_to_key(experience['state'])
        new_state_key = self._state_to_key(experience['new_state'])
        action = experience['action']
        
        # Get current Q-value
        current_q = model['q_values'].get((state_key, action), 0)
        
        # Find max Q-value for new state
        max_q_next = 0
        for (s, a), value in model['q_values'].items():
            if s == new_state_key and value > max_q_next:
                max_q_next = value
                
        # Q-learning update rule
        new_q = current_q + params['learning_rate'] * (
            experience['reward'] + params['discount_factor'] * max_q_next - current_q
        )
        
        # Update Q-value
        model['q_values'][(state_key, action)] = new_q
        model['update_count'] += 1
        model['last_updated'] = time.time()
        
    async def _update_neural_model(self, coupling_id, experience):
        """Update neural network model."""
        if coupling_id not in self.neural_models:
            return
            
        model = self.coupling_models[coupling_id]
        params = self.model_parameters[coupling_id]
        nn_model = self.neural_models[coupling_id]
        
        # Add to experience buffer (already done in main update method)
        
        # If we have enough experiences, perform a batch update
        if len(self.experience_buffer[coupling_id]) >= params['batch_size']:
            # Sample batch
            batch = random.sample(self.experience_buffer[coupling_id], params['batch_size'])
            
            # Prepare tensors
            states = torch.tensor([list(exp['state'].values()) for exp in batch], dtype=torch.float32)
            actions = [exp['action'] for exp in batch]
            rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32)
            next_states = torch.tensor([list(exp['new_state'].values()) for exp in batch], dtype=torch.float32)
            
            # Convert actions to indices
            action_space = list(model['action_space'].keys())
            action_indices = torch.tensor([action_space.index(a) for a in actions], dtype=torch.long)
            
            # Get current Q-values
            current_q = nn_model(states)
            
            # Get max Q-values for next states
            with torch.no_grad():
                next_q = nn_model(next_states)
                max_next_q = next_q.max(1)[0]
                
            # Calculate target Q-values
            target_q = current_q.clone()
            for i in range(params['batch_size']):
                target_q[i, action_indices[i]] = rewards[i] + params['discount_factor'] * max_next_q[i]
                
            # Optimize model
            nn_model.optimizer.zero_grad()
            loss = F.smooth_l1_loss(current_q, target_q)
            loss.backward()
            nn_model.optimizer.step()
            
            # Update model metadata
            model['update_count'] += 1
            model['last_updated'] = time.time()
            
    def _initialize_neural_model(self, coupling_id, coupling):
        """Initialize neural network for a coupling."""
        state_size = len(self._define_state_space(coupling))
        action_size = len(self._define_action_space(coupling))
        
        # Simple neural network
        model = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Store model and optimizer
        model.optimizer = optimizer
        self.neural_models[coupling_id] = model
        
    def _determine_model_type(self, coupling):
        """Determine appropriate model type for coupling."""
        # Simple heuristic: use neural for more complex couplings
        if coupling.coupling_type.name in ['ADAPTIVE', 'PREDICTIVE']:
            return 'neural'
        return 'tabular'
        
    def _define_state_space(self, coupling):
        """Define state space for a coupling."""
        # Default state features
        state_space = {
            'bayesian_confidence': 0.5,
            'coupling_strength': coupling.coupling_strength,
            'interaction_count': 0,
            'time_since_last': 0
        }
        return state_space
        
    def _define_action_space(self, coupling):
        """Define action space for a coupling."""
        # Basic action types with parameters
        action_space = {
            'standard': {
                'intensity': 0.5,
                'timing_offset': 0
            },
            'intensive': {
                'intensity': 0.8,
                'timing_offset': 0
            },
            'minimal': {
                'intensity': 0.2,
                'timing_offset': 0
            }
        }
        return action_space
        
    def _extract_state(self, data):
        """Extract state representation from data."""
        # Simple state extraction
        state = {
            'interaction_type': hash(data.get('interaction_type', 'unknown')) % 100 / 100,
            'bayesian_confidence': data.get('bayesian_confidence', 0.5)
        }
        return state
        
    def _extract_action(self, data):
        """Extract action from interaction data."""
        # Default to standard action
        return data.get('action_type', 'standard')
        
    def _calculate_reward(self, bayesian_update):
        """Calculate reward from Bayesian update."""
        # Reward based on confidence change
        confidence_before = bayesian_update.get('prior_confidence', 0.5)
        confidence_after = bayesian_update.get('new_confidence', 0.5)
        
        # Positive reward for confidence increase
        if confidence_after > confidence_before:
            return 2 * (confidence_after - confidence_before)
        # Smaller positive reward for maintaining high confidence
        elif confidence_after > 0.7:
            return 0.1
        # Small negative reward for confidence decrease
        else:
            return -0.5 * (confidence_before - confidence_after)
            
    def _state_to_key(self, state):
        """Convert state dict to hashable key."""
        # Round values to reduce state space
        return tuple((k, round(v, 2)) for k, v in sorted(state.items()))
        
    def _action_to_parameters(self, action, coupling):
        """Convert action to distribution parameters."""
        model = self.coupling_models.get(coupling.id, {})
        action_space = model.get('action_space', {})
        
        # Get parameters for this action
        parameters = action_space.get(action, {}).copy()
        parameters['action_type'] = action
        
        # Add predicted reward if available
        if coupling.id in self.predicted_outcomes and action in self.predicted_outcomes[coupling.id]:
            parameters['predicted_reward'] = self.predicted_outcomes[coupling.id][action]['predicted_reward']
            
        return parameters
        
    async def cleanup_coupling_model(self, coupling_id):
        """Clean up resources for a terminated coupling."""
        if coupling_id in self.coupling_models:
            del self.coupling_models[coupling_id]
        if coupling_id in self.experience_buffer:
            del self.experience_buffer[coupling_id]
        if coupling_id in self.neural_models:
            del self.neural_models[coupling_id]
        if coupling_id in self.model_parameters:
            del self.model_parameters[coupling_id]
        if coupling_id in self.optimization_history:
            del self.optimization_history[coupling_id]
        if coupling_id in self.counterfactual_simulations:
            del self.counterfactual_simulations[coupling_id]
        if coupling_id in self.active_inference_tests:
            del self.active_inference_tests[coupling_id]
        if coupling_id in self.predicted_outcomes:
            del self.predicted_outcomes[coupling_id]
        return True
        
    async def perform_maintenance(self):
        """Perform periodic maintenance on RL models."""
        start_time = time.time()
        
        # Count models
        model_count = len(self.coupling_models)
        
        # Clean up old experiences
        total_experiences = 0
        cleaned_experiences = 0
        
        for coupling_id, experiences in self.experience_buffer.items():
            total_experiences += len(experiences)
            if len(experiences) > 1000:
                self.experience_buffer[coupling_id] = experiences[-1000:]
                cleaned_experiences += len(experiences) - 1000
                
        # Free GPU memory for neural models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            'model_count': model_count,
            'total_experiences': total_experiences,
            'cleaned_experiences': cleaned_experiences,
            'neural_models': len(self.neural_models),
            'active_inference_tests': len(self.active_inference_tests),
            'elapsed_time': time.time() - start_time
        }
        
    async def get_metrics(self):
        """Get metrics about the RL optimizer."""
        return {
            'model_count': len(self.coupling_models),
            'neural_model_count': len(self.neural_models),
            'total_experiences': sum(len(exp) for exp in self.experience_buffer.values()),
            'avg_update_count': np.mean([model['update_count'] for model in self.coupling_models.values()]) if self.coupling_models else 0,
            'counterfactual_simulations': sum(len(sims) for sims in self.counterfactual_simulations.values()),
            'active_inference_tests': len(self.active_inference_tests),
            'using_gpu': torch.cuda.is_available()
        }
