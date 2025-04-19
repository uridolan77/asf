import logging
from typing import Dict, Any, Optional, List, Tuple
import asyncio

from agentor.agents.base import Agent, AgentInput, AgentOutput
from agentor.llm_gateway.agents.router import SemanticRouter

logger = logging.getLogger(__name__)


class RouteResult:
    """Result of routing a query to an agent."""
    
    def __init__(self, agent: Agent, confidence: float):
        """Initialize the route result.
        
        Args:
            agent: The agent to route to
            confidence: The confidence in the routing decision
        """
        self.agent = agent
        self.confidence = confidence


class RuleBasedRouter:
    """A router that uses rules to route queries to agents."""
    
    def __init__(self):
        """Initialize the rule-based router."""
        self.rules: List[Tuple[callable, Agent]] = []
    
    def add_rule(self, condition: callable, agent: Agent):
        """Add a rule to the router.
        
        Args:
            condition: A function that takes a query and returns a boolean
            agent: The agent to route to if the condition is met
        """
        self.rules.append((condition, agent))
    
    async def route(self, input_data: AgentInput) -> Optional[RouteResult]:
        """Route a query to an agent.
        
        Args:
            input_data: The input data to route
            
        Returns:
            The route result, or None if no rule matched
        """
        for condition, agent in self.rules:
            if condition(input_data.query):
                return RouteResult(agent=agent, confidence=0.8)
        
        return None


class HierarchicalRouter:
    """A router that uses multiple routing strategies with fallback."""
    
    def __init__(
        self,
        semantic_router: SemanticRouter,
        rule_router: RuleBasedRouter,
        fallback_agent: Agent
    ):
        """Initialize the hierarchical router.
        
        Args:
            semantic_router: The semantic router to use
            rule_router: The rule-based router to use
            fallback_agent: The fallback agent to use
        """
        self.semantic_router = semantic_router
        self.rule_router = rule_router
        self.fallback_agent = fallback_agent
    
    async def route(self, input_data: AgentInput) -> AgentOutput:
        """Route a query to an agent and run it.
        
        Args:
            input_data: The input data to route
            
        Returns:
            The agent's output
        """
        # Try routing with semantic understanding first
        try:
            intent = await self.semantic_router.route(input_data.query)
            agent = self._get_agent_for_intent(intent)
            confidence = 0.9  # This would be provided by the semantic router in a real implementation
            
            # If high confidence, use semantic route
            if confidence > 0.8:
                logger.info(f"Routing to agent {agent.name} with high confidence: {confidence:.2f}")
                return await agent.run(input_data.query, input_data.context)
            
            # Medium confidence, try rule-based routing
            elif confidence > 0.4:
                logger.info(f"Semantic confidence too low ({confidence:.2f}), trying rule-based routing")
                rule_result = await self.rule_router.route(input_data)
                
                if rule_result:
                    logger.info(f"Routing to agent {rule_result.agent.name} based on rules")
                    return await rule_result.agent.run(input_data.query, input_data.context)
                
                # No rule matched, use the semantic route anyway
                logger.info(f"No rule matched, using semantic route with agent {agent.name}")
                return await agent.run(input_data.query, input_data.context)
            
            # Low confidence, use fallback
            else:
                logger.info(f"Confidence too low ({confidence:.2f}), using fallback agent")
                return await self.fallback_agent.run(input_data.query, input_data.context)
        
        except Exception as e:
            logger.error(f"Semantic routing failed: {str(e)}")
            
            # Fall back to rule-based routing
            try:
                rule_result = await self.rule_router.route(input_data)
                
                if rule_result:
                    logger.info(f"Routing to agent {rule_result.agent.name} based on rules")
                    return await rule_result.agent.run(input_data.query, input_data.context)
                
                # No rule matched, use fallback
                logger.info("No rule matched, using fallback agent")
                return await self.fallback_agent.run(input_data.query, input_data.context)
            
            except Exception as e2:
                logger.error(f"Rule-based routing failed: {str(e2)}")
                
                # Ultimate fallback
                logger.info("Using fallback agent after all routing failed")
                return await self.fallback_agent.run(input_data.query, input_data.context)
    
    def _get_agent_for_intent(self, intent: str) -> Agent:
        """Get the agent for an intent.
        
        In a real implementation, this would look up the agent in a registry.
        For now, we'll just return the fallback agent.
        
        Args:
            intent: The intent
            
        Returns:
            The agent for the intent
        """
        # This is a placeholder. In a real implementation, you would
        # have a registry of agents and look up the agent for the intent.
        return self.fallback_agent
