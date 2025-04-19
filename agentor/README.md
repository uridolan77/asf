# Agentor

A flexible agent framework with asynchronous support, enhanced error handling, caching, security, and monitoring.

## Features

### Asynchronous Support

- Async preprocess and postprocess methods for IO-bound operations
- Fully async API for LLM interactions
- Batch processing with concurrency control
- Parallel tool execution

### Enhanced Error Handling & Resilience

- Automatic retries for transient errors
- Circuit breakers for LLM calls
- Comprehensive error logging and monitoring
- Graceful degradation under load

### Performance Optimization

- LLM response caching with semantic deduplication
- Connection pooling for HTTP requests
- Batch processing for multiple LLM requests
- Cost-aware model selection

### Security

- API key authentication
- Role-based access control
- Input and output sanitization
- Prompt injection detection
- Sensitive data filtering

### Advanced Agent Routing

- Semantic routing based on embeddings
- Intent classification for query routing
- Hierarchical routing with fallback
- Multi-stage routing pipeline

### Monitoring & Observability

- Prometheus metrics for LLM requests, agent executions, and more
- Latency tracking and reporting
- Token usage and cost monitoring
- OpenTelemetry integration for distributed tracing

### Cost Management

- Token counting and cost tracking
- Budget enforcement
- Cost-aware model selection
- Usage reporting

## Installation

```bash
pip install -e .
```

## Usage

### Basic Agent

```python
import asyncio
from agentor.agents.base import Agent, AgentInput, AgentOutput

class MyAgent(Agent):
    def __init__(self, name=None):
        super().__init__(name)

    def decide(self):
        # Implement your decision logic here
        return "my_action"

    async def preprocess(self, input_data: AgentInput) -> AgentInput:
        # Preprocess the input data (e.g., fetch additional context)
        return input_data

    async def postprocess(self, output_data: AgentOutput) -> AgentOutput:
        # Postprocess the output data (e.g., log the response)
        return output_data

# Usage
async def main():
    agent = MyAgent(name="MyAgent")
    agent.register_action("my_action", lambda a: "Hello, world!")
    result = await agent.run("What should I do?", {"context": "value"})
    print(result.response)

asyncio.run(main())
```

### LLM Integration with Circuit Breaker and Cost Tracking

```python
from agentor.llm_gateway.llm.providers.openai import OpenAILLM
from agentor.llm_gateway.llm.base import LLMRequest
from agentor.llm_gateway.llm.semantic_cache import SemanticCache, SemanticCachedLLM
from agentor.llm_gateway.utils.circuit_breaker import LLMCircuitBreaker
from agentor.llm_gateway.utils.cost import CostTracker, TokenCounter, CostAwareLLM
from agentor.llm_gateway.utils.degradation import DegradationManager, DegradationAwareLLM

async def main():
    # Create an LLM client
    llm = OpenAILLM(api_key="your-api-key")

    # Add circuit breaker
    circuit_breaker = LLMCircuitBreaker()

    # Add semantic caching
    cache = SemanticCache(threshold=0.92)
    cached_llm = SemanticCachedLLM(llm=llm, cache=cache)

    # Add cost tracking
    cost_tracker = CostTracker()
    token_counter = TokenCounter()
    cost_aware_llm = CostAwareLLM(cached_llm, cost_tracker, token_counter)

    # Add degradation awareness
    degradation_manager = DegradationManager()
    degradation_aware_llm = DegradationAwareLLM(cost_aware_llm, degradation_manager)

    # Generate text
    request = LLMRequest(
        prompt="Hello, world!",
        model="gpt-3.5-turbo-instruct",
        temperature=0.7
    )

    response = await degradation_aware_llm.generate(request)
    print(response.text)
    print(f"Cost: ${response.metadata['cost']['total_cost']:.6f}")

asyncio.run(main())
```

### Hierarchical Routing with Fallback

```python
from agentor.llm_gateway.agents.router import SemanticRouter
from agentor.llm_gateway.agents.hierarchical_router import HierarchicalRouter, RuleBasedRouter
from agentor.agents.base import Agent, AgentInput

async def main():
    # Create a semantic router
    semantic_router = SemanticRouter(api_key="your-api-key")

    # Add routes
    await semantic_router.add_route("weather", "Get weather information for a location")
    await semantic_router.add_route("news", "Get the latest news headlines")

    # Create a rule-based router
    rule_router = RuleBasedRouter()

    # Create some agents
    weather_agent = Agent(name="WeatherAgent")
    news_agent = Agent(name="NewsAgent")
    fallback_agent = Agent(name="FallbackAgent")

    # Add rules
    rule_router.add_rule(
        lambda query: "weather" in query.lower(),
        weather_agent
    )
    rule_router.add_rule(
        lambda query: "news" in query.lower(),
        news_agent
    )

    # Create a hierarchical router
    router = HierarchicalRouter(
        semantic_router=semantic_router,
        rule_router=rule_router,
        fallback_agent=fallback_agent
    )

    # Route a query
    input_data = AgentInput(query="What's the weather like in New York?", context={})
    result = await router.route(input_data)
    print(f"Response: {result.response}")

asyncio.run(main())
```

### Parallel Tool Execution

```python
from agentor.agents.base import Agent
from agentor.agents.tools import BaseTool, ToolResult

class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__(name="weather", description="Get weather information")

    async def run(self, location: str) -> ToolResult:
        # In a real implementation, this would call a weather API
        return ToolResult(
            success=True,
            data={"temperature": 72, "conditions": "sunny", "location": location}
        )

class NewsTool(BaseTool):
    def __init__(self):
        super().__init__(name="news", description="Get news headlines")

    async def run(self, topic: str) -> ToolResult:
        # In a real implementation, this would call a news API
        return ToolResult(
            success=True,
            data={"headlines": [f"Latest {topic} news", f"Another {topic} story"]}
        )

async def main():
    # Create an agent
    agent = Agent(name="ToolAgent")

    # Register tools
    weather_tool = WeatherTool()
    news_tool = NewsTool()
    agent.register_tool(weather_tool)
    agent.register_tool(news_tool)

    # Execute tools in parallel
    tools_to_run = [
        (weather_tool, {"location": "New York"}),
        (news_tool, {"topic": "technology"})
    ]

    results = await agent.execute_tools(tools_to_run)

    for result in results:
        print(result)

asyncio.run(main())
```

## Testing

```bash
pytest
```

## License

MIT
