import datetime
import uuid

class ComplianceModelRegistry:
    """
    Registry for managing specialized compliance models for different regulatory domains.
    Provides efficient access, lifecycle management, and metadata tracking.
    """
    def __init__(self):
        self.models = {}
        self.model_stats = {}
        self.load_history = {}
        
    def register_model(self, domain, model, metadata=None):
        """
        Register a model for a specific regulatory domain
        
        Args:
            domain: Regulatory domain identifier (e.g., "GDPR", "HIPAA")
            model: The model instance
            metadata: Optional metadata about the model
        """
        if metadata is None:
            metadata = {}
            
        # Add registration timestamp
        metadata["registered_at"] = datetime.datetime.now().isoformat()
        
        # Store model and metadata
        self.models[domain] = {
            "model": model,
            "metadata": metadata,
            "status": "active"
        }
        
        # Initialize stats
        self.model_stats[domain] = {
            "usage_count": 0,
            "avg_latency": 0,
            "compliance_rate": 1.0,
            "last_used": None
        }
        
        # Record load in history
        self.load_history[domain] = {
            "loaded_at": datetime.datetime.now().isoformat(),
            "version": metadata.get("version", "unknown")
        }
        
        return True
        
    def get_model(self, domain):
        """Get model for a specific domain"""
        if domain not in self.models:
            return None
            
        model_entry = self.models[domain]
        
        # Update stats
        self.model_stats[domain]["usage_count"] += 1
        self.model_stats[domain]["last_used"] = datetime.datetime.now().isoformat()
        
        return model_entry["model"]
        
    def unregister_model(self, domain):
        """Unregister a model"""
        if domain in self.models:
            # Update load history
            self.load_history[domain]["unloaded_at"] = datetime.datetime.now().isoformat()
            
            # Remove model
            del self.models[domain]
            return True
            
        return False
        
    def update_model_stats(self, domain, latency=None, compliance_success=None):
        """Update model usage statistics"""
        if domain not in self.model_stats:
            return False
            
        stats = self.model_stats[domain]
        
        # Update latency if provided
        if latency is not None:
            # Exponential moving average for latency
            if stats["usage_count"] > 1:
                stats["avg_latency"] = 0.9 * stats["avg_latency"] + 0.1 * latency
            else:
                stats["avg_latency"] = latency
                
        # Update compliance rate if provided
        if compliance_success is not None:
            success_value = 1.0 if compliance_success else 0.0
            if stats["usage_count"] > 1:
                stats["compliance_rate"] = 0.95 * stats["compliance_rate"] + 0.05 * success_value
            else:
                stats["compliance_rate"] = success_value
                
        return True
        
    def get_model_stats(self, domain=None):
        """Get model usage statistics"""
        if domain:
            return self.model_stats.get(domain, {})
        else:
            return self.model_stats
            
    def get_available_domains(self):
        """Get list of available regulatory domains"""
        return list(self.models.keys())
        
    def get_model_metadata(self, domain):
        """Get metadata for a specific model"""
        if domain in self.models:
            return self.models[domain]["metadata"]
        return None