import sys
import os
import optimizer
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from src.compliance.compliance_balanced_sampler import ComplianceBalancedSampler
from src.compliance.compliance_dataset import load_general_compliance_dataset, load_framework_dataset, load_compliance_edge_cases, load_compliance_adversarial    
# Ensure the correct Python environment is being used
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Add the current directory to the Python path if necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


class ComplianceAwareDistillation:
    """
    Knowledge distillation framework that preserves compliance capabilities
    while creating smaller, efficient models for specific regulatory domains
    """
    
    def __init__(self, teacher_model, student_architecture, regulatory_framework):
        """
        Initialize the compliance-aware distillation framework
        
        Args:
            teacher_model: Large, fully-capable compliance model
            student_architecture: Architecture specification for student model
            regulatory_framework: Target regulatory framework (e.g., 'GDPR', 'HIPAA')
        """
        self.teacher = teacher_model
        self.student = self._initialize_student(student_architecture)
        self.regulatory_framework = regulatory_framework
        self.framework_datasets = self._load_framework_datasets()
        
        # Track compliance metrics during distillation
        self.compliance_metrics = {
            "baseline": self._evaluate_teacher_compliance(),
            "history": []
        }
        
    def _initialize_student(self, architecture):
        """Initialize student model based on architecture spec"""
        # Implementation would create appropriate student model
        # Could be from scratch or a pretrained smaller model
        # Define or import SmallerComplianceModel
        # Placeholder implementation for SmallerComplianceModel
        class SmallerComplianceModel(nn.Module):
            def __init__(self, architecture):
                super(SmallerComplianceModel, self).__init__()
                # Define the architecture of the smaller model here
                self.architecture = architecture
                self.dummy_layer = nn.Linear(10, 10)  # Example layer

            def forward(self, x):
                return self.dummy_layer(x)

        return SmallerComplianceModel(architecture)

    def _load_framework_datasets(self):
        """Load datasets specific to the target regulatory framework"""
        datasets = {
            "general": load_general_compliance_dataset(),
            "framework_specific": load_framework_dataset(self.regulatory_framework),
            "edge_cases": load_compliance_edge_cases(self.regulatory_framework),
            "adversarial": load_compliance_adversarial(self.regulatory_framework)
        }
        
        return datasets
    
    def _evaluate_teacher_compliance(self):
        """Evaluate teacher model compliance as baseline"""
        metrics = {}
        
        for dataset_name, dataset in self.framework_datasets.items():
            metrics[dataset_name] = self.evaluate_compliance(
                self.teacher, dataset, self.regulatory_framework
            )
            
        metrics["overall"] = sum(metrics.values()) / len(metrics)
        return metrics
        
    def distill(self, epochs=50, batch_size=32, temperature=2.0, alpha=0.5):
        """
        Perform compliance-aware knowledge distillation
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            temperature: Softmax temperature for distillation
            alpha: Weight balancing distillation and compliance losses
            
        Returns:
            Trained student model with preserved compliance capabilities
        """
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-4)
        
        # Setup learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=epochs
        )
        
        # Calculate effective training steps
        total_steps = sum(len(d) // batch_size for d in self.framework_datasets.values()) * epochs
        print(f"Starting distillation for {self.regulatory_framework} compliance")
        print(f"Training for {epochs} epochs, {total_steps} total steps")
        
        # Track best model
        best_model = None
        best_compliance = 0.0
        
        for epoch in range(epochs):
            # Train for one epoch
            train_metrics = self._train_epoch(
                epoch, batch_size, temperature, alpha
            )
            
            # Evaluate compliance on validation sets
            compliance_metrics = self._evaluate_student_compliance()
            
            # Track history
            self.compliance_metrics["history"].append({
                "epoch": epoch,
                "train": train_metrics,
                "compliance": compliance_metrics
            })
            
            # Update best model if improved
            if compliance_metrics["overall"] > best_compliance:
                best_compliance = compliance_metrics["overall"]
                best_model = copy.deepcopy(self.student)
                print(f"Epoch {epoch}: New best model with compliance score {best_compliance:.4f}")
            
            # Log progress
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train loss: {train_metrics['total_loss']:.4f}")
            print(f"  Distillation loss: {train_metrics['distillation_loss']:.4f}")
            print(f"  Compliance loss: {train_metrics['compliance_loss']:.4f}")
            print(f"  Overall compliance: {compliance_metrics['overall']:.4f}")
            print(f"  Teacher compliance: {self.compliance_metrics['baseline']['overall']:.4f}")
            print(f"  Compliance ratio: {compliance_metrics['overall'] / self.compliance_metrics['baseline']['overall']:.4f}")
            
            # Update learning rate
            scheduler.step()
            
        # Restore best model
        if best_model is not None:
            self.student = best_model
            
        # Final compliance evaluation
        final_compliance = self._evaluate_student_compliance()
        self.compliance_metrics["final"] = final_compliance
        
        return self.student
    
    def _train_epoch(self, epoch, batch_size, temperature, alpha):
        """Train student model for one epoch"""
        self.student.train()
        self.teacher.eval()
        
        # Initialize metrics
        metrics = {
            "total_loss": 0.0,
            "distillation_loss": 0.0,
            "compliance_loss": 0.0,
            "steps": 0
        }
        
        # Create regulatory dataset sampler that balances different compliance aspects
        sampler = ComplianceBalancedSampler(self.framework_datasets)
        
        for batch in sampler.get_batches(batch_size):
            # Forward pass through teacher model
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    batch["input_ids"], 
                    batch["attention_mask"]
                )
                
            # Forward pass through student model
            student_outputs = self.student(
                batch["input_ids"],
                batch["attention_mask"]
            )
            
            # Compute distillation loss
            distillation_loss = self._compute_distillation_loss(
                student_outputs, teacher_outputs, temperature
            )
            
            # Compute compliance loss
            compliance_loss = self._compute_compliance_loss(
                student_outputs, batch
            )
            
            # Combined loss
            loss = alpha * distillation_loss + (1 - alpha) * compliance_loss
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update metrics
            metrics["total_loss"] += loss.item()
            metrics["distillation_loss"] += distillation_loss.item()
            metrics["compliance_loss"] += compliance_loss.item()
            metrics["steps"] += 1
            
            # Dynamic alpha adjustment based on compliance gap
            if metrics["steps"] % 100 == 0:
                # Periodically evaluate compliance
                compliance = self._evaluate_student_compliance(sample=True)
                teacher_compliance = self.compliance_metrics["baseline"]["overall"]
                
                # Adjust alpha based on compliance gap
                compliance_ratio = compliance["overall"] / teacher_compliance
                if compliance_ratio < 0.9:
                    # Increase focus on compliance if falling behind
                    alpha = max(0.1, alpha - 0.05)
                elif compliance_ratio > 0.99:
                    # Increase focus on distillation if compliance is good
                    alpha = min(0.9, alpha + 0.05)
        
        # Average metrics
        for key in ["total_loss", "distillation_loss", "compliance_loss"]:
            metrics[key] /= metrics["steps"]
            
        return metrics
    
    def _compute_distillation_loss(self, student_outputs, teacher_outputs, temperature):
        """
        Compute knowledge distillation loss
        
        Uses temperature-scaled softmax to create soft targets from teacher
        """
        # Get logits from outputs
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # Apply temperature scaling
        scaled_student = student_logits / temperature
        scaled_teacher = teacher_logits / temperature
        
        # Compute KL divergence loss
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        loss = loss_fn(
            F.log_softmax(scaled_student, dim=-1),
            F.softmax(scaled_teacher, dim=-1)
        )
        
        # Scale by temperature²
        return loss * (temperature ** 2)
    
    def _compute_compliance_loss(self, student_outputs, batch):
        """
        Compute specialized compliance loss
        
        Enforces regulatory constraints directly during training
        """
        # Standard task loss (e.g., token prediction)
        task_loss = F.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            batch["labels"].view(-1)
        )
        
        # Regulatory constraint loss
        regulatory_loss = self._regulatory_constraint_loss(
            student_outputs, batch
        )
        
        # Combine losses with higher weight on regulatory constraints
        combined_loss = 0.3 * task_loss + 0.7 * regulatory_loss
        
        return combined_loss
    
    def _regulatory_constraint_loss(self, outputs, batch):
        """
        Compute loss based on regulatory constraints
        
        Implements specific loss terms for the target regulatory framework
        """
        # Extract relevant constraint information from batch
        constraints = batch.get("regulatory_constraints", {})
        
        # Initialize constraint losses
        constraint_losses = []
        
        # Example: GDPR-specific constraints
        if self.regulatory_framework == "GDPR":
            # Data minimization constraint
            if "data_minimization" in constraints:
                min_loss = self._compute_data_minimization_loss(
                    outputs, constraints["data_minimization"]
                )
                constraint_losses.append(min_loss)
                
            # Purpose limitation constraint
            if "purpose_limitation" in constraints:
                purpose_loss = self._compute_purpose_limitation_loss(
                    outputs, constraints["purpose_limitation"]
                )
                constraint_losses.append(purpose_loss)
            
            # Additional GDPR-specific constraints...
            
        # Example: HIPAA-specific constraints
        elif self.regulatory_framework == "HIPAA":
            # PHI protection constraint
            if "phi_protection" in constraints:
                phi_loss = self._compute_phi_protection_loss(
                    outputs, constraints["phi_protection"]
                )
                constraint_losses.append(phi_loss)
                
            # Additional HIPAA-specific constraints...
            
        # Default to generic regulatory constraints if none specific
        if not constraint_losses:
            constraint_losses.append(self._compute_generic_regulatory_loss(outputs, batch))
            
        # Combine all constraint losses
        if constraint_losses:
            return sum(constraint_losses) / len(constraint_losses)
        else:
            # Return zero tensor if no constraints
            return torch.tensor(0.0, device=outputs.logits.device)
    
    def _compute_data_minimization_loss(self, outputs, constraint):
        """Compute loss enforcing GDPR data minimization principle"""
        # Implementation would penalize outputs that include excessive data
        return torch.tensor(0.1, device=outputs.logits.device)  # Placeholder
    
    def _compute_purpose_limitation_loss(self, outputs, constraint):
        """Compute loss enforcing GDPR purpose limitation principle"""
        # Implementation would encourage outputs that adhere to stated purpose
        return torch.tensor(0.1, device=outputs.logits.device)  # Placeholder
    
    def _compute_phi_protection_loss(self, outputs, constraint):
        """Compute loss enforcing HIPAA PHI protection requirements"""
        # Implementation would penalize disclosure of protected health information
        return torch.tensor(0.1, device=outputs.logits.device)  # Placeholder
    
    def _compute_generic_regulatory_loss(self, outputs, batch):
        """Compute generic regulatory compliance loss"""
        # Implementation would apply general compliance principles
        return torch.tensor(0.1, device=outputs.logits.device)  # Placeholder

  
    @staticmethod
    def evaluate_compliance(model, dataset, framework):
        """Evaluate model compliance on dataset for specific framework"""
        # Placeholder implementation for compliance evaluation
        # Replace with actual logic to evaluate compliance
        return 0.95  # Example compliance score

    def count_parameters(model):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def sample_dataset(self, dataset, max_samples=100):
        """Create smaller sample of dataset for quick evaluation"""
        # Implementation would sample subset of dataset
        return dataset  # Placeholder

    # Helper estimation functions
    def count_parameters(model):
        """Count trainable parameters in model"""
        if hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 0  # Default for models without parameter attribute

    def estimate_memory_footprint(model):
        """Estimate memory footprint of model in MB"""
        if hasattr(model, 'parameters'):
            params = sum(p.numel() for p in model.parameters())
            # Assume 4 bytes per parameter for FP32
            return (params * 4) / (1024 * 1024)
        return 0  # Default for models without parameter attribute

    def _evaluate_student_compliance(self, sample=False):
        """
        Evaluate student model compliance with regulatory framework
        
        Args:
            sample: If True, use smaller sample for efficiency during training
            
        Returns:
            Compliance metrics dictionary
        """
        self.student.eval()
        metrics = {}
        
        # Use either full datasets or sampled subsets
        datasets = {}
        if sample:
            # Use small samples during training
            for name, dataset in self.framework_datasets.items():
                datasets[name] = self.sample_dataset(dataset, max_samples=100)
        else:
            datasets = self.framework_datasets
        
        # Evaluate on each dataset
        for dataset_name, dataset in datasets.items():
            metrics[dataset_name] = ComplianceAwareDistillation.evaluate_compliance(
                self.student, dataset, self.regulatory_framework
            )
            
        # Calculate overall compliance
        metrics["overall"] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    def generate_compliance_report(self):
        """
        Generate detailed compliance comparison report between teacher and student
        
        Returns:
            Dictionary with detailed compliance metrics and visualizations
        """
        # Verify we have completed distillation
        if "final" not in self.compliance_metrics:
            return {"error": "Distillation not completed yet"}
            
        # Compile comprehensive report
        report = {
            "model_comparison": {
                "teacher_parameters": ComplianceAwareDistillation.count_parameters(self.teacher),
                "student_parameters": ComplianceAwareDistillation.count_parameters(self.student),
                "compression_ratio": ComplianceAwareDistillation.count_parameters(self.teacher) / ComplianceAwareDistillation.count_parameters(self.student),
            },
            "compliance_comparison": {
                "teacher": self.compliance_metrics["baseline"],
                "student": self.compliance_metrics["final"],
                "ratio": {
                    k: self.compliance_metrics["final"][k] / self.compliance_metrics["baseline"][k]
                    for k in self.compliance_metrics["baseline"]
                }
            },
            "training_history": self.compliance_metrics["history"],
            "framework_specific": self._generate_framework_specific_report(),
            "conclusion": self._generate_conclusion()
        }
        
        return report
    
    def _generate_framework_specific_report(self):
        """Generate regulatory framework-specific detailed analysis"""
        # Implementation would provide detailed framework-specific metrics
        return {"framework": self.regulatory_framework}  # Placeholder
    
    def _generate_conclusion(self):
        """Generate conclusion and recommendations"""
        # Implementation would analyze results and provide recommendations
        student_overall = self.compliance_metrics["final"]["overall"]
        teacher_overall = self.compliance_metrics["baseline"]["overall"]
        ratio = student_overall / teacher_overall
        
        if ratio > 0.95:
            status = "Excellent"
            message = "Student model successfully preserves compliance capabilities"
        elif ratio > 0.9:
            status = "Good"
            message = "Student model retains most compliance capabilities"
        elif ratio > 0.8:
            status = "Acceptable"
            message = "Student model meets minimum compliance requirements but could be improved"
        else:
            status = "Insufficient"
            message = "Student model fails to adequately preserve compliance capabilities"
            
        return {
            "status": status,
            "message": message,
            "compliance_preservation": f"{ratio:.2%}",
            "recommendations": self._generate_recommendations(ratio)
        }
    
    def _generate_recommendations(self, compliance_ratio):
        """Generate specific recommendations based on results"""
        # Implementation would provide targeted recommendations
        recommendations = []
        
        if compliance_ratio < 0.9:
            recommendations.append(
                "Increase model capacity in compliance-critical components"
            )
            
        if compliance_ratio < 0.95:
            recommendations.append(
                "Fine-tune with specialized compliance datasets"
            )
            
        return recommendations or ["No specific recommendations required"]
  
