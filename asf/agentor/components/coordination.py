class AgentCoordinator:
    """Coordinates multiple agents."""
    
    def __init__(self):
        self.agents = {}
        self.message_queue = []
    
    def register_agent(self, agent):
        """
        Register an agent with the coordinator.
        
        Args:
            agent: Agent to register
        """
        self.agents[agent.agent_id] = agent
    
    def unregister_agent(self, agent_id):
        """
        Unregister an agent from the coordinator.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def send_message(self, from_agent_id, to_agent_id, message):
        """
        Send a message from one agent to another.
        
        Args:
            from_agent_id: ID of the sending agent
            to_agent_id: ID of the receiving agent
            message: Message to send
        """
        self.message_queue.append({
            'from': from_agent_id,
            'to': to_agent_id,
            'message': message,
            'timestamp': time.time(),
            'status': 'pending'
        })
    
    def broadcast_message(self, from_agent_id, message):
        """
        Broadcast a message to all other agents.
        
        Args:
            from_agent_id: ID of the sending agent
            message: Message to broadcast
        """
        for agent_id in self.agents:
            if agent_id != from_agent_id:
                self.send_message(from_agent_id, agent_id, message)
    
    def deliver_messages(self):
        """Deliver all pending messages to their recipients."""
        for msg in self.message_queue:
            if msg['status'] == 'pending':
                if msg['to'] in self.agents:
                    # Add message to recipient's perception queue
                    agent = self.agents[msg['to']]
                    if hasattr(agent, 'receive_message'):
                        agent.receive_message(msg['from'], msg['message'])
                    msg['status'] = 'delivered'
                else:
                    msg['status'] = 'failed'  # Recipient not found
        
        # Remove delivered and failed messages
        self.message_queue = [msg for msg in self.message_queue 
                             if msg['status'] == 'pending']
    
    def run_all(self, iterations=1):
        """
        Run all registered agents for the given number of iterations.
        
        Args:
            iterations: Number of iterations to run each agent
        """
        for _ in range(iterations):
            for agent in self.agents.values():
                agent.run_once()
            self.deliver_messages()