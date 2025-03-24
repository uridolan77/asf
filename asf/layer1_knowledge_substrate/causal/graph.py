# asf/knowledge_substrate/causal/graph.py
import time
import networkx as nx

from asf.layer1_knowledge_substrate.causal.variable import CausalVariable

class CausalGraph:
    """
    Represents a causal graph with variables and their relationships.
    """
    def __init__(self):
        self.variables = {}  # Map from variable name to CausalVariable object
        self.interventions = []  # Track interventions for causal discovery
    
    def add_variable(self, name, value=None, feature_id=None):
        """Add a new variable to the causal graph"""
        if name not in self.variables:
            self.variables[name] = CausalVariable(name, value, feature_id)
        return self.variables[name]
    
    def add_causal_link(self, cause_name, effect_name, strength=0.5):
        """Add a causal link between two variables"""
        # Ensure both variables exist
        if cause_name not in self.variables:
            self.add_variable(cause_name)
        if effect_name not in self.variables:
            self.add_variable(effect_name)
        
        # Add the relationship in both directions
        self.variables[cause_name].add_child(effect_name, strength)
        self.variables[effect_name].add_parent(cause_name, strength)
    
    def update_causal_strength(self, cause_name, effect_name, new_strength):
        """Update the strength of a causal relationship"""
        if cause_name in self.variables and effect_name in self.variables:
            self.variables[cause_name].update_causal_strength(effect_name, new_strength, is_parent=False)
            self.variables[effect_name].update_causal_strength(cause_name, new_strength, is_parent=True)
    
    def perform_intervention(self, variable_name, new_value):
        """
        Perform an intervention by setting a variable to a specific value
        Returns the previous value and records the intervention
        """
        if variable_name not in self.variables:
            return None
        
        var = self.variables[variable_name]
        old_value = var.value
        var.value = new_value
        
        # Record intervention
        self.interventions.append({
            "variable": variable_name,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": time.time()
        })
        
        return old_value
    
    def get_causal_parents(self, variable_name):
        """Get all causal parents of a variable"""
        if variable_name in self.variables:
            return self.variables[variable_name].parents
        return {}
    
    def get_causal_children(self, variable_name):
        """Get all causal children of a variable"""
        if variable_name in self.variables:
            return self.variables[variable_name].children
        return {}
    
    def to_networkx(self):
        """Convert causal graph to NetworkX graph for visualization and analysis"""
        G = nx.DiGraph()
        
        # Add nodes
        for name, var in self.variables.items():
            G.add_node(name, value=var.value, feature_id=var.feature_id)
        
        # Add edges
        for name, var in self.variables.items():
            for child_name, strength in var.children.items():
                G.add_edge(name, child_name, weight=strength)
        
        return G
