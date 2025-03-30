
import copy
import datetime
import uuid
import torch
import contextlib

# Context manager for temporary quantization during sensitivity analysis
@contextlib.contextmanager
def tempquant(model, target_module, bits):
    """Temporarily quantize a specific module for evaluation"""
    # Store original parameters
    orig_state = {}
    for name, module in model.named_modules():
        if name == target_module:
            for param_name, param in module.named_parameters():
                orig_state[param_name] = param.data.clone()
                
                # Apply temporary quantization
                if param.dim() > 0:
                    max_val = float(param.abs().max())
                    scale = (2**(bits-1) - 1) / max_val if max_val > 0 else 1.0
                    quantized = torch.round(param.data * scale)
                    quantized = torch.clamp(quantized, -2**(bits-1), 2**(bits-1)-1)
                    param.data = quantized / scale
    
    try:
        yield
    finally:
        # Restore original parameters
        for name, module in model.named_modules():
            if name == target_module:
                for param_name, param in module.named_parameters():
                    if param_name in orig_state:
                        param.data = orig_state[param_name]

class ComplianceAwareMixedPrecisionQuantizer:
    """
    Mixed-precision quantizer that preserves compliance-critical components
    while aggressively quantizing non-critical components
    """
    
    def __init__(self, model, compliance_evaluator):
        self.model = model
        self.compliance_evaluator = compliance_evaluator
        self.precision_mapping = {}
        self.compliance_modules = self._identify_compliance_modules()
        
    def _identify_compliance_modules(self):
        """Identify compliance-critical modules in the model"""
        compliance_modules = []
        
        # Identify modules by name pattern
        for name, module in self.model.named_modules():
            if any(keyword in name.lower() for keyword in [
                'compliance', 'regulatory', 'rule', 'constraint', 
                'filter', 'verify', 'check', 'gate'
            ]):
                compliance_modules.append(name)
        
        return compliance_modules
  

    def analyze_sensitivity(self, calibration_data, bits_options=[8, 4, 2]):
        """
        Analyze quantization sensitivity of different model components
        
        Args:
            calibration_data: Dataset for calibration
            bits_options: Different bit-width options to test
            
        Returns:
            Sensitivity analysis for each module
        """
        sensitivity_results = {}
        
        # Get baseline compliance score
        baseline_score = self.compliance_evaluator.evaluate(self.model, calibration_data)
        print(f"Baseline compliance score: {baseline_score:.4f}")
        
        # Test each module with different precision levels
        for name, module in self.model.named_modules():
            if not list(module.parameters()):  # Skip modules without parameters
                continue
                
            module_sensitivity = {}
            
            for bits in bits_options:
                # Temporarily quantize this module
                with tempquant(self.model, target_module=name, bits=bits):
                    # Evaluate compliance
                    quant_score = self.compliance_evaluator.evaluate(
                        self.model, calibration_data
                    )
                    
                # Calculate sensitivity as relative compliance reduction
                sensitivity = (baseline_score - quant_score) / baseline_score
                module_sensitivity[bits] = {
                    'compliance_score': quant_score,
                    'sensitivity': sensitivity,
                }
                
            sensitivity_results[name] = module_sensitivity
            
        return sensitivity_results
    
    def _determine_optimal_precision(self, sensitivity_results, compliance_threshold=0.99):
        """
        Determine optimal precision for each module based on sensitivity
        
        Args:
            sensitivity_results: Results from sensitivity analysis
            compliance_threshold: Minimum acceptable compliance after quantization
            
        Returns:
            Mapping of module names to optimal bit precision
        """
        precision_mapping = {}
        baseline_score = self.compliance_evaluator.baseline_score
        
        for name, sensitivities in sensitivity_results.items():
            # Default to high precision (8-bit) for compliance modules
            if name in self.compliance_modules:
                precision_mapping[name] = 8
                continue
                
            # Find lowest acceptable precision for other modules
            for bits in sorted(sensitivities.keys()):  # Try lowest precision first
                sensitivity = sensitivities[bits]['sensitivity']
                score = sensitivities[bits]['compliance_score']
                
                # Check if this precision maintains acceptable compliance
                if score >= compliance_threshold * baseline_score:
                    precision_mapping[name] = bits
                    break
            else:
                # If no precision level is acceptable, use highest precision
                precision_mapping[name] = max(sensitivities.keys())
                
        return precision_mapping
    
    def quantize(self, calibration_data=None, compliance_threshold=0.99):
        """
        Quantize model with mixed precision based on compliance sensitivity
        
        Args:
            calibration_data: Dataset for calibration (if not already analyzed)
            compliance_threshold: Minimum acceptable compliance after quantization
            
        Returns:
            Quantized model with mixed precision
        """
        # Run sensitivity analysis if not already done
        if not hasattr(self, 'sensitivity_results') and calibration_data is not None:
            self.sensitivity_results = self.analyze_sensitivity(calibration_data)
            
        # Determine optimal precision for each module
        self.precision_mapping = self._determine_optimal_precision(
            self.sensitivity_results, compliance_threshold
        )
        
        # Apply mixed-precision quantization
        quantized_model = copy.deepcopy(self.model)
        
        for name, module in quantized_model.named_modules():
            if name in self.precision_mapping:
                bits = self.precision_mapping[name]
                self._quantize_module(module, bits)
                
        return quantized_model
    
    def _quantize_module(self, module, bits):
        """Apply quantization to a single module with specified bit-width"""
        # This is a simplified implementation - actual quantization would use
        # a framework like PyTorch quantization, TensorRT, or similar
        
        # For each parameter in the module
        for param_name, param in module.named_parameters():
            if param.dim() > 0:  # Skip scalar parameters
                # Calculate quantization scale
                max_val = float(param.abs().max())
                scale = (2**(bits-1) - 1) / max_val if max_val > 0 else 1.0
                
                # Quantize weights
                quantized = torch.round(param * scale)
                quantized = torch.clamp(quantized, -2**(bits-1), 2**(bits-1)-1)
                
                # Store quantized weights and scale
                param.data = quantized / scale
                
                # In a full implementation, we would also:
                # 1. Replace operations with quantized versions
                # 2. Implement fake quantization for training
                # 3. Handle activations quantization
                # 4. Add observer modules for calibration
    
    def generate_report(self):
        """Generate detailed report on quantization decisions"""
        if not hasattr(self, 'sensitivity_results'):
            return "No sensitivity analysis results available"
            
        report = {
            "compliance_modules": self.compliance_modules,
            "precision_distribution": self._get_precision_distribution(),
            "memory_savings": self._calculate_memory_savings(),
            "module_details": self._get_module_details(),
        }
        
        return report
    
    def _get_precision_distribution(self):
        """Get distribution of precision levels across the model"""
        distribution = {}
        for bits in set(self.precision_mapping.values()):
            count = sum(1 for b in self.precision_mapping.values() if b == bits)
            distribution[bits] = count
            
        return distribution
    
    def _calculate_memory_savings(self):
        """Calculate memory savings from quantization"""
        original_bits = 32  # Assuming FP32 original model
        
        # Count parameters by precision
        param_counts = {}
        for name, module in self.model.named_modules():
            if name in self.precision_mapping:
                bits = self.precision_mapping[name]
                params = sum(p.numel() for p in module.parameters())
                
                if bits not in param_counts:
                    param_counts[bits] = 0
                param_counts[bits] += params
        
        # Calculate original and new storage requirements
        original_size = sum(param_counts.values()) * (original_bits / 8)  # in bytes
        new_size = sum(count * (bits / 8) for bits, count in param_counts.items())
        
        savings = 1.0 - (new_size / original_size)
        
        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": new_size / (1024 * 1024),
            "savings_percentage": savings * 100,
            "parameter_distribution": param_counts
        }
    
    def _get_module_details(self):
        """Get detailed information about each module's quantization"""
        details = {}
        for name in self.precision_mapping:
            bits = self.precision_mapping[name]
            is_compliance = name in self.compliance_modules
            
            if hasattr(self, 'sensitivity_results') and name in self.sensitivity_results:
                sensitivity = self.sensitivity_results[name][bits]['sensitivity']
            else:
                sensitivity = "Unknown"
                
            details[name] = {
                "precision_bits": bits,
                "is_compliance_critical": is_compliance,
                "sensitivity": sensitivity
            }
            
        return details
