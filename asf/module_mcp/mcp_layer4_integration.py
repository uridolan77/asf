# === FILE: layer4_environmental_coupling/mcp_integration.py ===

import asyncio
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable

# Import MCP SDK
from mcp_sdk import (
    MCPClient,
    MCPConfig,
    Message,
    Context,
    Prediction,
    MessageType,
    ContextLevel,
    ConfidenceLevel,
    InteractionStage,
    ModelState,
    MCPError
)

class MCPIntegration:
    """
    Integrates Layer 4 with the Model Context Protocol using the MCP SDK.
    """
    def __init__(self, env_coupling_layer):
        """
        Initialize the MCP integration.
        
        Args:
            env_coupling_layer: Reference to the EnvironmentalCouplingLayer
        """
        self.env_coupling_layer = env_coupling_layer
        self.logger = logging.getLogger("ASF.Layer4.MCPIntegration")
        
        # Initialize MCP client
        config = MCPConfig(
            entity_id="asf.layer4",
            default_timeout=10.0,
            log_level="INFO"
        )
        self.client = MCPClient(config)
        
        # Conversation tracking
        self.active_conversations = {}
        
        # Metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'queries_processed': 0,
            'predictions_sent': 0,
            'tests_performed': 0
        }
    
    async def initialize(self):
        """Initialize the MCP integration."""
        self.logger.info("Initializing MCP Integration with SDK")
        
        # Register message handlers
        await self.client.register_handler(MessageType.QUERY, self._handle_query)
        await self.client.register_handler(MessageType.RESPONSE, self._handle_response)
        await self.client.register_handler(MessageType.PREDICTION, self._handle_prediction)
        await self.client.register_handler(MessageType.OBSERVATION, self._handle_observation)
        await self.client.register_handler(MessageType.ACTION, self._handle_action)
        await self.client.register_handler(MessageType.TEST, self._handle_test)
        await self.client.register_handler(MessageType.COUNTERFACTUAL, self._handle_counterfactual)
        await self.client.register_handler(MessageType.FEEDBACK, self._handle_feedback)
        await self.client.register_handler(MessageType.SYSTEM, self._handle_system)
        
        # Start the client
        await self.client.start()
        
        self.logger.info("MCP Integration initialized")
        return {'status': 'initialized'}
    
    async def process_message(self, message_data: Any) -> Dict[str, Any]:
        """
        Process an incoming MCP message.
        
        Args:
            message_data: The message data (string, dict, or Message)
            
        Returns:
            Result of message processing
        """
        start_time = time.time()
        
        try:
            # Process the message
            response = await self.client.process_message(message_data)
            
            self.metrics['messages_received'] += 1
            
            result = {
                'success': True,
                'message_processed': True,
                'has_response': response is not None,
                'elapsed_time': time.time() - start_time
            }
            
            # Include response if available
            if response:
                result['response'] = response.to_dict()
                
            return result
            
        except MCPError as e:
            self.logger.error(f"MCP error processing message: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def send_interaction(self, 
                           entity_id: str,
                           target_id: str,
                           interaction_data: Dict[str, Any],
                           interaction_type: str = 'query',
                           prediction_data: Optional[Dict[str, Any]] = None,
                           confidence: float = 0.5,
                           wait_for_response: bool = False,
                           timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Send an interaction using MCP.
        
        Args:
            entity_id: ID of the sending entity
            target_id: ID of the target environmental entity
            interaction_data: Data for the interaction
            interaction_type: Type of interaction
            prediction_data: Optional prediction data
            confidence: Confidence in prediction (0.0 to 1.0)
            wait_for_response: Whether to wait for a response
            timeout: Timeout in seconds
            
        Returns:
            Result of the interaction
        """
        start_time = time.time()
        
        try:
            # Map interaction type to message type
            message_type = self._map_interaction_to_message_type(interaction_type)
            
            # Create the message
            message = await self.client.create_message(
                message_type=message_type,
                content=interaction_data,
                recipient_id=target_id,
                prediction_data=prediction_data,
                confidence=confidence
            )
            
            # Override sender_id if different from client's entity_id
            if entity_id != self.client.entity_id:
                message.sender_id = entity_id
                message.context.entity_id = entity_id
            
            # Send the message
            if wait_for_response:
                response = await self.client.send_message(
                    message, 
                    wait_for_response=True,
                    timeout=timeout
                )
                
                self.metrics['messages_sent'] += 1
                
                return {
                    'success': True,
                    'message_id': message.id,
                    'message_type': message_type.name,
                    'has_response': True,
                    'response': response.to_dict() if response else None,
                    'elapsed_time': time.time() - start_time
                }
            else:
                await self.client.send_message(message)
                
                self.metrics['messages_sent'] += 1
                
                return {
                    'success': True,
                    'message_id': message.id,
                    'message_type': message_type.name,
                    'has_response': False,
                    'elapsed_time': time.time() - start_time
                }
        
        except MCPError as e:
            self.logger.error(f"MCP error sending interaction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
        except Exception as e:
            self.logger.error(f"Error sending interaction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def send_prediction(self, 
                          entity_id: str,
                          target_id: str,
                          prediction_data: Dict[str, Any],
                          confidence: float = 0.7,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a prediction using MCP.
        
        Args:
            entity_id: ID of the entity making the prediction
            target_id: ID of the target entity
            prediction_data: The prediction data
            confidence: Confidence in the prediction (0.0 to 1.0)
            metadata: Optional metadata
            
        Returns:
            Result of sending the prediction
        """
        start_time = time.time()
        
        try:
            result = await self.client.send_prediction(
                recipient_id=target_id,
                prediction_data=prediction_data,
                confidence=confidence,
                metadata=metadata
            )
            
            self.metrics['predictions_sent'] += 1
            
            result['elapsed_time'] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Error sending prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def send_observation(self,
                           environmental_id: str,
                           entity_id: str,
                           observation_data: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send an observation using MCP.
        
        Args:
            environmental_id: ID of the environmental entity
            entity_id: ID of the target entity
            observation_data: The observation data
            metadata: Optional metadata
            
        Returns:
            Result of sending the observation
        """
        start_time = time.time()
        
        try:
            # Use client's entity_id as the source of observation
            temp_entity_id = self.client.entity_id
            
            # Override the client's entity_id temporarily if needed
            if environmental_id != self.client.entity_id:
                # In this case, we're sending on behalf of the environmental entity
                self.client.entity_id = environmental_id
            
            result = await self.client.send_observation(
                recipient_id=entity_id,
                observation_data=observation_data,
                metadata=metadata
            )
            
            # Restore original entity_id
            self.client.entity_id = temp_entity_id
            
            self.metrics['messages_sent'] += 1
            
            result['elapsed_time'] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Error sending observation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def perform_active_inference_test(self,
                                       entity_id: str,
                                       target_id: str,
                                       test_parameters: Dict[str, Any],
                                       expected_results: Dict[str, Any],
                                       confidence: float = 0.7,
                                       wait_for_result: bool = True,
                                       timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform an active inference test using MCP.
        
        Args:
            entity_id: ID of the entity performing the test
            target_id: ID of the target entity
            test_parameters: The test parameters
            expected_results: Expected results of the test
            confidence: Confidence in expected results (0.0 to 1.0)
            wait_for_result: Whether to wait for result
            timeout: Timeout in seconds
            
        Returns:
            Result of the test
        """
        start_time = time.time()
        
        try:
            # Override entity_id if needed
            temp_entity_id = self.client.entity_id
            if entity_id != self.client.entity_id:
                self.client.entity_id = entity_id
            
            # Send the test
            response = await self.client.send_test(
                recipient_id=target_id,
                test_parameters=test_parameters,
                expected_results=expected_results,
                test_type='active_inference',
                confidence=confidence,
                wait_for_result=wait_for_result,
                timeout=timeout
            )
            
            # Restore original entity_id
            self.client.entity_id = temp_entity_id
            
            self.metrics['tests_performed'] += 1
            
            result = {
                'success': True,
                'has_response': response is not None,
                'elapsed_time': time.time() - start_time
            }
            
            # Include response if available
            if response:
                result['response'] = response.to_dict()
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing active inference test: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def simulate_counterfactual(self,
                                  entity_id: str,
                                  target_id: str,
                                  variation_parameters: Dict[str, Any],
                                  expected_outcomes: Dict[str, Any],
                                  wait_for_result: bool = True,
                                  timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Simulate a counterfactual using MCP.
        
        Args:
            entity_id: ID of the entity performing the simulation
            target_id: ID of the target entity
            variation_parameters: The variation parameters
            expected_outcomes: Expected outcomes of the simulation
            wait_for_result: Whether to wait for result
            timeout: Timeout in seconds
            
        Returns:
            Result of the simulation
        """
        start_time = time.time()
        
        try:
            # Override entity_id if needed
            temp_entity_id = self.client.entity_id
            if entity_id != self.client.entity_id:
                self.client.entity_id = entity_id
            
            # Send the counterfactual
            response = await self.client.send_counterfactual(
                recipient_id=target_id,
                variation_parameters=variation_parameters,
                expected_outcomes=expected_outcomes,
                wait_for_result=wait_for_result,
                timeout=timeout
            )
            
            # Restore original entity_id
            self.client.entity_id = temp_entity_id
            
            self.metrics['messages_sent'] += 1
            
            result = {
                'success': True,
                'has_response': response is not None,
                'elapsed_time': time.time() - start_time
            }
            
            # Include response if available
            if response:
                result['response'] = response.to_dict()
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error simulating counterfactual: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def send_feedback(self,
                        entity_id: str,
                        target_id: str,
                        prediction_id: str,
                        feedback_value: float,
                        feedback_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send feedback on a prediction using MCP.
        
        Args:
            entity_id: ID of the entity sending feedback
            target_id: ID of the target entity
            prediction_id: ID of the prediction to provide feedback on
            feedback_value: Feedback value (-1.0 to 1.0, where positive is good)
            feedback_data: Optional additional feedback data
            
        Returns:
            Result of sending the feedback
        """
        start_time = time.time()
        
        try:
            # Override entity_id if needed
            temp_entity_id = self.client.entity_id
            if entity_id != self.client.entity_id:
                self.client.entity_id = entity_id
            
            # Send the feedback
            result = await self.client.send_feedback(
                recipient_id=target_id,
                prediction_id=prediction_id,
                feedback_value=feedback_value,
                feedback_data=feedback_data
            )
            
            # Restore original entity_id
            self.client.entity_id = temp_entity_id
            
            self.metrics['messages_sent'] += 1
            
            result['elapsed_time'] = time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.error(f"Error sending feedback: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    async def shutdown(self):
        """Shutdown the MCP client."""
        self.logger.info("Shutting down MCP client")
        await self.client.stop()
        return {'status': 'shutdown'}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the MCP integration."""
        return {
            'active_conversations': len(self.active_conversations),
            'messages_sent': self.metrics['messages_sent'],
            'messages_received': self.metrics['messages_received'],
            'queries_processed': self.metrics['queries_processed'],
            'predictions_sent': self.metrics['predictions_sent'],
            'tests_performed': self.metrics['tests_performed']
        }
    
    # === Message Handlers ===
    
    async def _handle_query(self, message: Message) -> Optional[Message]:
        """Handle a query message."""
        self.logger.debug(f"Handling query message {message.id}")
        self.metrics['queries_processed'] += 1
        
        # Process as environmental interaction
        interaction_result = await self.env_coupling_layer.process_environmental_interaction(
            interaction_data=message.content,
            source_id=message.sender_id,
            interaction_type='query',
            context=message.context.to_dict()
        )
        
        # Create response based on interaction result
        return message.create_reply(
            content={
                'status': 'success' if interaction_result.get('success', False) else 'error',
                'result': interaction_result
            },
            message_type=MessageType.RESPONSE
        )
    
    async def _handle_response(self, message: Message) -> Optional[Message]:
        """Handle a response message."""
        self.logger.debug(f"Handling response message {message.id}")
        
        # No need to create a response to a response
        return None
    
    async def _handle_prediction(self, message: Message) -> Optional[Message]:
        """Handle a prediction message."""
        self.logger.debug(f"Handling prediction message {message.id}")
        
        if not message.prediction:
            return message.create_reply(
                content={'status': 'error', 'error': 'Missing prediction data'},
                message_type=MessageType.ERROR
            )
            
        # Process as prediction data
        if hasattr(self.env_coupling_layer, 'predictive_modeler'):
            modeler = self.env_coupling_layer.predictive_modeler
            
            # Convert MCP prediction to environmental prediction
            env_prediction = await self._convert_mcp_to_env_prediction(
                message.prediction,
                message.sender_id
            )
            
            # Store in the predictions dictionary
            modeler.predictions[env_prediction.id] = env_prediction
            
            # Add to entity predictions
            entity_id = env_prediction.environmental_entity_id
            modeler.entity_predictions[entity_id].append(env_prediction.id)
            
            return message.create_reply(
                content={'status': 'success', 'prediction_id': env_prediction.id},
                message_type=MessageType.SYSTEM
            )
        else:
            return message.create_reply(
                content={'status': 'error', 'error': 'Predictive modeler not available'},
                message_type=MessageType.ERROR
            )
    
    async def _handle_observation(self, message: Message) -> Optional[Message]:
        """Handle an observation message."""
        self.logger.debug(f"Handling observation message {message.id}")
        
        # Process as environmental interaction
        interaction_result = await self.env_coupling_layer.process_environmental_interaction(
            interaction_data=message.content,
            source_id=message.sender_id,
            interaction_type='observation',
            context=message.context.to_dict()
        )
        
        # Create acknowledgment response
        return message.create_reply(
            content={
                'status': 'received',
                'processing_status': 'success' if interaction_result.get('success', False) else 'error'
            },
            message_type=MessageType.SYSTEM
        )
    
    async def _handle_action(self, message: Message) -> Optional[Message]:
        """Handle an action message."""
        self.logger.debug(f"Handling action message {message.id}")
        
        # Process as environmental interaction with action type
        interaction_result = await self.env_coupling_layer.process_environmental_interaction(
            interaction_data=message.content,
            source_id=message.sender_id,
            interaction_type='action',
            context=message.context.to_dict()
        )
        
        # Create response based on action result
        return message.create_reply(
            content={
                'status': 'action_processed',
                'result': interaction_result,
                'success': interaction_result.get('success', False)
            },
            message_type=MessageType.RESPONSE
        )
    
    async def _handle_test(self, message: Message) -> Optional[Message]:
        """Handle a test message."""
        self.logger.debug(f"Handling test message {message.id}")
        
        # Check if this is an active inference test
        is_active_inference = (message.context.interaction_stage == InteractionStage.ACTIVE_INFERENCE or
                              message.context.metadata.get('test_type') == 'active_inference')
        
        if is_active_inference and hasattr(self.env_coupling_layer, 'active_inference'):
            # Process as active inference test
            test_id = message.context.metadata.get('test_id')
            
            if not test_id:
                test_id = str(uuid.uuid4())
                message.context.metadata['test_id'] = test_id
                
            # Process through active inference controller
            entity_id = message.context.entity_id
            target_id = message.context.environmental_id
            
            # Find coupling_id for these entities
            coupling_id = await self._find_coupling_id(entity_id, target_id)
            
            if coupling_id:
                # Create test parameters from message
                test_parameters = {
                    'test_id': test_id,
                    'test_content': message.content,
                    'prediction': message.prediction.predicted_data if message.prediction else {},
                    'expected_reward': message.content.get('expected_reward', 0.5),
                    'timestamp': time.time()
                }
                
                # Set up the test
                test_result = await self.env_coupling_layer.active_inference.setup_active_inference_test(
                    coupling_id,
                    test_parameters
                )
                
                # Process the test interaction
                interaction_result = await self.env_coupling_layer.process_environmental_interaction(
                    interaction_data=message.content,
                    source_id=message.sender_id,
                    interaction_type='test',
                    context=message.context.to_dict()
                )
                
                # Create response with test setup information
                return message.create_reply(
                    content={
                        'status': 'test_processed',
                        'test_id': test_id,
                        'test_setup': test_result,
                        'interaction_result': interaction_result
                    },
                    message_type=MessageType.RESPONSE
                )
            else:
                return message.create_reply(
                    content={'status': 'error', 'error': 'No coupling found for test'},
                    message_type=MessageType.ERROR
                )
        else:
            # Process as a regular test interaction
            interaction_result = await self.env_coupling_layer.process_environmental_interaction(
                interaction_data=message.content,
                source_id=message.sender_id,
                interaction_type='test',
                context=message.context.to_dict()
            )
            
            # Create response based on test result
            return message.create_reply(
                content={
                    'status': 'test_processed',
                    'result': interaction_result,
                    'success': interaction_result.get('success', False)
                },
                message_type=MessageType.RESPONSE
            )
    
    async def _handle_counterfactual(self, message: Message) -> Optional[Message]:
        """Handle a counterfactual message."""
        self.logger.debug(f"Handling counterfactual message {message.id}")
        
        # Check if counterfactual processing is available
        if hasattr(self.env_coupling_layer, 'counterfactual_simulator'):
            # Extract information from message
            variation_type = message.context.metadata.get('variation_type', 'unknown')
            
            # Find coupling_id for the entities
            entity_id = message.context.entity_id
            target_id = message.context.environmental_id
            coupling_id = await self._find_coupling_id(entity_id, target_id)
            
            if coupling_id:
                # Create counterfactual parameters
                variation = {
                    'id': message.id,
                    'base_coupling_id': coupling_id,
                    'variation_type': variation_type,
                    'description': message.content.get('description', f'Counterfactual {variation_type}'),
                    'parameters': message.content
                }
                
                # Simulate the counterfactual
                simulation_result = await self.env_coupling_layer.counterfactual_simulator.simulate_outcomes([variation])
                
                # Create response with simulation results
                return message.create_reply(
                    content={
                        'status': 'counterfactual_processed',
                        'coupling_id': coupling_id,
                        'variation_type': variation_type,
                        'simulation_result': simulation_result[0] if simulation_result else None
                    },
                    message_type=MessageType.RESPONSE
                )
            else:
                return message.create_reply(
                    content={'status': 'error', 'error': 'No coupling found for counterfactual'},
                    message_type=MessageType.ERROR
                )
        else:
            return message.create_reply(
                content={'status': 'error', 'error': 'Counterfactual simulator not available'},
                message_type=MessageType.ERROR
            )
    
    async def _handle_feedback(self, message: Message) -> Optional[Message]:
        """Handle a feedback message."""
        self.logger.debug(f"Handling feedback message {message.id}")
        
        # Extract feedback information
        feedback_data = message.content
        prediction_id = feedback_data.get('prediction_id')
        feedback_value = feedback_data.get('feedback_value')
        
        if not prediction_id:
            return message.create_reply(
                content={'status': 'error', 'error': 'Missing prediction_id in feedback'},
                message_type=MessageType.ERROR
            )
            
        # Process feedback for the prediction
        if hasattr(self.env_coupling_layer, 'predictive_modeler'):
            modeler = self.env_coupling_layer.predictive_modeler
            
            # Find the prediction
            if prediction_id in modeler.predictions:
                prediction = modeler.predictions[prediction_id]
                
                # Create mock actual data based on feedback
                actual_data = {
                    'feedback': feedback_value,
                    'timestamp': time.time()
                }
                
                # Evaluate the prediction against feedback
                evaluation = await modeler.evaluate_prediction(prediction_id, actual_data)
                
                return message.create_reply(
                    content={
                        'status': 'feedback_processed',
                        'prediction_id': prediction_id,
                        'evaluation': evaluation
                    },
                    message_type=MessageType.SYSTEM
                )
            else:
                return message.create_reply(
                    content={'status': 'error', 'error': f'Prediction {prediction_id} not found'},
                    message_type=MessageType.ERROR
                )
        else:
            return message.create_reply(
                content={'status': 'error', 'error': 'Predictive modeler not available'},
                message_type=MessageType.ERROR
            )
    
    async def _handle_system(self, message: Message) -> Optional[Message]:
        """Handle a system message."""
        self.logger.debug(f"Handling system message {message.id}")
        
        # Process system message
        system_command = message.content.get('command')
        
        if system_command == 'ping':
            return message.create_reply(
                content={'status': 'pong', 'timestamp': time.time()},
                message_type=MessageType.SYSTEM
            )
        elif system_command == 'status':
            # Collect status information
            if hasattr(self.env_coupling_layer, 'get_metrics'):
                metrics = await self.env_coupling_layer.get_metrics()
            else:
                metrics = {'status': 'available'}
                
            return message.create_reply(
                content={'status': 'active', 'metrics': metrics},
                message_type=MessageType.SYSTEM
            )
        else:
            # Generic acknowledgment
            return message.create_reply(
                content={'status': 'acknowledged', 'timestamp': time.time()},
                message_type=MessageType.SYSTEM
            )
    
    # === Utility Methods ===
    
    def _map_interaction_to_message_type(self, interaction_type: str) -> MessageType:
        """Map interaction type to MCP message type."""
        mapping = {
            'query': MessageType.QUERY,
            'response': MessageType.RESPONSE,
            'prediction': MessageType.PREDICTION,
            'observation': MessageType.OBSERVATION,
            'update': MessageType.UPDATE,
            'action': MessageType.ACTION,
            'feedback': MessageType.FEEDBACK,
            'test': MessageType.TEST,
            'counterfactual': MessageType.COUNTERFACTUAL,
            'metadata': MessageType.METADATA,
            'system': MessageType.SYSTEM,
            'error': MessageType.ERROR
        }
        
        return mapping.get(interaction_type.lower(), MessageType.SYSTEM)
    
    async def _find_coupling_id(self, entity_id: str, environmental_id: str) -> Optional[str]:
        """Find coupling ID for an entity pair."""
        if not hasattr(self.env_coupling_layer, 'coupling_registry'):
            return None
            
        # Get couplings for the internal entity
        couplings = await self.env_coupling_layer.coupling_registry.get_couplings_by_internal_entity(entity_id)
        
        # Find coupling with matching environmental entity
        for coupling in couplings:
            if coupling.environmental_entity_id == environmental_id:
                return coupling.id
                
        return None
    
    async def _convert_mcp_to_env_prediction(self, mcp_prediction: Prediction, 
                                       environmental_entity_id: str) -> Any:
        """Convert MCP prediction to environmental prediction."""
        from asf.layer4_environmental_coupling.models import EnvironmentalPrediction
        
        return EnvironmentalPrediction(
            id=mcp_prediction.prediction_id,
            environmental_entity_id=environmental_entity_id,
            predicted_data=mcp_prediction.predicted_data,
            confidence=mcp_prediction.confidence,
            precision=mcp_prediction.precision,
            prediction_time=mcp_prediction.creation_time,
            context=mcp_prediction.metadata.get('context', {})
        )
    
    async def _convert_env_to_mcp_prediction(self, env_prediction: Any) -> Prediction:
        """Convert environmental prediction to MCP prediction."""
        # Convert confidence to confidence level
        confidence_level = self._confidence_to_level(env_prediction.confidence)
        
        # Create MCP Prediction
        return Prediction(
            prediction_id=env_prediction.id,
            predicted_data=env_prediction.predicted_data,
            confidence=env_prediction.confidence,
            confidence_level=confidence_level,
            precision=env_prediction.precision,
            creation_time=env_prediction.prediction_time,
            expiration_time=env_prediction.prediction_time + 3600,  # 1 hour expiration
            metadata={
                'original_type': 'EnvironmentalPrediction',
                'context': env_prediction.context,
                'verification_time': env_prediction.verification_time,
                'prediction_error': env_prediction.prediction_error
            }
        )
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence value to confidence level."""
        if confidence < 0.3:
            return ConfidenceLevel.SPECULATIVE
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MODERATE
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.CERTAIN


# === FILE: layer4_environmental_coupling/environmental_coupling_layer.py (Updated for MCP SDK) ===

import asyncio
import time
import uuid
import logging
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from asf.layer4_environmental_coupling.enums import (
    CouplingType, CouplingStrength, CouplingState, EventPriority, PredictionState
)
from asf.layer4_environmental_coupling.models import (
    CouplingEvent, EnvironmentalCoupling, EnvironmentalPrediction, ActiveInferenceTest
)

# Original components
from asf.layer4_environmental_coupling.components.coupling_registry import SparseCouplingRegistry
from asf.layer4_environmental_coupling.components.event_processor import EventDrivenProcessor, AsyncEventQueue
from asf.layer4_environmental_coupling.components.enhanced_bayesian_updater import EnhancedBayesianUpdater
from asf.layer4_environmental_coupling.components.rl_optimizer import ReinforcementLearningOptimizer
from asf.layer4_environmental_coupling.components.coherence_boundary import CoherenceBoundaryController
from asf.layer4_environmental_coupling.components.gpu_accelerator import GPUAccelerationManager
from asf.layer4_environmental_coupling.components.context_tracker import AdaptiveContextTracker
from asf.layer4_environmental_coupling.components.distributed_cache import DistributedCouplingCache
from asf.layer4_environmental_coupling.components.metrics_collector import PerformanceMetricsCollector

# Seth's Data Paradox components
from asf.layer4_environmental_coupling.components.predictive_modeler import PredictiveEnvironmentalModeler
from asf.layer4_environmental_coupling.components.active_inference_controller import ActiveInferenceController
from asf.layer4_environmental_coupling.components.counterfactual_simulator import CounterfactualSimulator

# NEW: MCP Protocol using the SDK
from asf.layer4_environmental_coupling.mcp_integration import MCPIntegration

class EnvironmentalCouplingLayer:
    """
    Enhanced Layer 4 with complete integration of Seth's predictive processing principles
    and Model Context Protocol (MCP) for standardized communication using the MCP SDK.
    
    Orchestrates controlled hallucination, precision-weighted prediction errors, active inference,
    and counterfactual simulation as a unified predictive system, with MCP as the core communication protocol.
    """
    def __init__(self, knowledge_substrate, async_queue=None, config=None):
        # Original components with enhancements
        self.knowledge_substrate = knowledge_substrate
        self.async_queue = async_queue or AsyncEventQueue()
        self.config = config or {}
        
        # Core components with enhanced versions
        self.coupling_registry = SparseCouplingRegistry(
            initial_capacity=self.config.get('initial_capacity', 10000)
        )
        self.event_processor = EventDrivenProcessor(
            max_concurrency=self.config.get('max_concurrency', 16)
        )
        self.bayesian_updater = EnhancedBayesianUpdater()  # Seth-enhanced version
        self.rl_optimizer = ReinforcementLearningOptimizer(
            learning_rate=self.config.get('learning_rate', 0.01)
        )
        
        # Seth's Data Paradox components
        self.predictive_modeler = PredictiveEnvironmentalModeler()
        self.active_inference = ActiveInferenceController(knowledge_substrate)
        self.counterfactual_simulator = CounterfactualSimulator()
        
        # Original components
        self.coherence_boundary = CoherenceBoundaryController(knowledge_substrate)
        self.gpu_accelerator = GPUAccelerationManager(
            enabled=self.config.get('use_gpu', True)
        )
        self.context_tracker = AdaptiveContextTracker()
        self.distributed_cache = DistributedCouplingCache(
            self.config.get('cache_config', {})
        )
        
        # NEW: MCP Protocol integration using the SDK
        self.mcp_integration = MCPIntegration(self)
        self.use_mcp = self.config.get('use_mcp', True)
        
        # Seth's principle configuration
        self.prediction_enabled = self.config.get('prediction_enabled', True)
        self.active_inference_enabled = self.config.get('active_inference_enabled', True)
        self.counterfactual_enabled = self.config.get('counterfactual_enabled', True)
        
        # Adjacent layers
        self.layer5 = None
        self.layer6 = None
        
        # Performance metrics
        self.metrics_collector = PerformanceMetricsCollector()
        self.logger = logging.getLogger("ASF.Layer4.EnvironmentalCouplingLayer")
        
    async def initialize(self, layer5=None, layer6=None):
        """Initialize the layer and connect to adjacent layers."""
        self.layer5 = layer5  # AutopoieticMaintenanceLayer
        self.layer6 = layer6  # EnvironmentalDistributionLayer
        
        # Initialize components
        await self.coupling_registry.initialize()
        await self.event_processor.initialize(self.process_coupling_event)
        await self.rl_optimizer.initialize(self.knowledge_substrate)
        await self.gpu_accelerator.initialize()
        
        # NEW: Initialize MCP integration with SDK
        if self.use_mcp:
            await self.mcp_integration.initialize()
            self.logger.info("MCP Protocol integration enabled using SDK")
        
        # Set up component connections
        self.active_inference.set_coupling_registry(self.coupling_registry)
        
        # Start background task for async event processing
        asyncio.create_task(self.event_processor.run_processing_loop())
        
        self.logger.info(f"Layer 4 (Environmental Coupling) initialized with Seth's Data Paradox principles and MCP SDK integration")
        self.logger.info(f"Prediction enabled: {self.prediction_enabled}")
        self.logger.info(f"Active inference enabled: {self.active_inference_enabled}")
        self.logger.info(f"Counterfactual simulation enabled: {self.counterfactual_enabled}")
        self.logger.info(f"MCP enabled: {self.use_mcp}")
        
        return {'status': 'initialized'}
    
    # === MCP Protocol Methods using SDK ===
    
    async def process_mcp_message(self, message_data: Any) -> Dict[str, Any]:
        """
        Process an incoming MCP message using the SDK.
        
        Args:
            message_data: MCP message data (string, dict, or Message)
            
        Returns:
            Result of message processing
        """
        if not self.use_mcp:
            return {'success': False, 'error': 'MCP protocol is disabled'}
            
        return await self.mcp_integration.process_message(message_data)
    
    async def send_mcp_interaction(self, 
                               entity_id: str,
                               target_id: str,
                               interaction_data: Dict[str, Any],
                               interaction_type: str = 'query',
                               prediction_data: Optional[Dict[str, Any]] = None,
                               wait_for_response: bool = False) -> Dict[str, Any]:
        """
        Send an interaction using MCP SDK.
        
        Args:
            entity_id: ID of the internal entity
            target_id: ID of the target environmental entity
            interaction_data: Data for the interaction
            interaction_type: Type of interaction
            prediction_data: Optional prediction data
            wait_for_response: Whether to wait for a response
            
        Returns:
            Result of the interaction
        """
        if not self.use_mcp:
            return {'success': False, 'error': 'MCP protocol is disabled'}
            
        return await self.mcp_integration.send_interaction(
            entity_id=entity_id,
            target_id=target_id,
            interaction_data=interaction_data,
            interaction_type=interaction_type,
            prediction_data=prediction_data,
            wait_for_response=wait_for_response
        )
    
    async def distribute_entity_with_mcp(self,
                                     entity_id: str,
                                     target_id: str,
                                     interaction_data: Dict[str, Any],
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Distribute an entity to an environmental target using MCP SDK.
        This is a bridge to Layer 6 (Environmental Distribution Layer) with MCP.
        
        Args:
            entity_id: ID of the entity to distribute
            target_id: Target environmental entity ID
            interaction_data: Data for the interaction
            context: Optional context information
            
        Returns:
            Result of distribution
        """
        # Determine interaction type from context
        interaction_type = 'query'
        wait_for_response = False
        
        if context:
            if context.get('test_interaction'):
                interaction_type = 'test'
                wait_for_response = True
            elif context.get('active_inference'):
                interaction_type = 'test'
                wait_for_response = True
                
        # Create prediction data if available
        prediction_data = None
        if self.prediction_enabled and hasattr(self, 'predictive_modeler'):
            # Generate a prediction for this interaction
            prediction = await self.predictive_modeler.predict_interaction(
                target_id,
                context=context
            )
            
            if prediction:
                # Extract prediction data
                prediction_data = prediction.predicted_data
        
        # Use MCP SDK to send interaction
        return await self.send_mcp_interaction(
            entity_id=entity_id,
            target_id=target_id,
            interaction_data=interaction_data,
            interaction_type=interaction_type,
            prediction_data=prediction_data,
            wait_for_response=wait_for_response
        )
    
    # The rest of the Environmental Coupling Layer methods would remain largely the same,
    # but would use the MCP SDK integration for communication.
    
    # For example, in the perform_active_inference method:
    
    async def perform_active_inference(self, entity_id):
        """
        Actively test and optimize couplings using controlled interactions.
        Implements Seth's active inference principle with MCP SDK integration.
        """
        if not self.active_inference_enabled:
            return {'success': False, 'error': 'Active inference is disabled'}
        
        start_time = time.time()
        
        # Get all current couplings
        couplings = await self.coupling_registry.get_couplings_by_internal_entity(entity_id)
        if not couplings:
            return {
                'success': False,
                'error': 'No couplings found',
                'elapsed_time': time.time() - start_time
            }
        
        inference_results = []
        
        # For each coupling, generate test interactions to minimize uncertainty
        for coupling in couplings:
            # Only test active couplings
            if not hasattr(coupling, 'coupling_state') or coupling.coupling_state != CouplingState.ACTIVE:
                continue
                
            # Generate optimal test interaction based on current uncertainty
            test = await self.active_inference.generate_test_interaction(
                coupling.id,
                uncertainty_focus=True
            )
            
            if test:
                # Execute test interaction to gather information
                if self.layer6:  # EnvironmentalDistributionLayer
                    test_interaction_result = None
                    
                    # NEW: Use MCP SDK for test interaction if enabled
                    if self.use_mcp:
                        test_interaction_result = await self.mcp_integration.perform_active_inference_test(
                            entity_id=entity_id,
                            target_id=coupling.environmental_entity_id,
                            test_parameters=test.test_parameters,
                            expected_results={'expected_response': True},
                            confidence=0.7,
                            wait_for_result=True
                        )
                    else:
                        # Traditional approach if MCP not enabled
                        test_interaction_result = await self.layer6.distribute_entity(
                            entity_id,
                            target_id=coupling.environmental_entity_id,
                            context={
                                'test_interaction': True,
                                'active_inference': True,
                                'test_id': test.id
                            },
                            interaction_data=test.test_parameters
                        )
                    
                    inference_results.append({
                        'coupling_id': coupling.id,
                        'test_id': test.id,
                        'test_parameters': test.test_parameters,
                        'status': 'submitted',
                        'interaction_result': test_interaction_result
                    })
                else:
                    # No distribution layer available
                    inference_results.append({
                        'coupling_id': coupling.id,
                        'test_id': test.id,
                        'test_parameters': test.test_parameters,
                        'status': 'distribution_layer_unavailable'
                    })
        
        return {
            'success': True,
            'entity_id': entity_id,
            'couplings_tested': len(inference_results),
            'inference_results': inference_results,
            'elapsed_time': time.time() - start_time
        }
    
    async def get_metrics(self):
        """Get metrics about the layer and its components."""
        metrics = {
            'start_time': time.time(),
            'couplings': {},
            'predictions': {},
            'events': {},
            'mcp': {}
        }
        
        # Coupling metrics
        if self.coupling_registry:
            metrics['couplings'] = await self.coupling_registry.get_statistics()
            
        # Prediction metrics
        if hasattr(self, 'predictive_modeler'):
            prediction_count = len(self.predictive_modeler.predictions)
            entity_count = len(self.predictive_modeler.entity_predictions)
            
            metrics['predictions'] = {
                'total_predictions': prediction_count,
                'entity_count': entity_count,
                'precision_entities': len(self.predictive_modeler.precision)
            }
            
        # Event metrics
        if hasattr(self, 'event_processor'):
            event_metrics = await self.event_processor.get_metrics()
            metrics['events'] = event_metrics
            
        # MCP metrics
        if self.use_mcp and hasattr(self, 'mcp_integration'):
            mcp_metrics = await self.mcp_integration.get_metrics()
            metrics['mcp'] = mcp_metrics
            
        # Calculate total processing time
        metrics['elapsed_time'] = time.time() - metrics['start_time']
        
        return metrics