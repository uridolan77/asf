from src.core.processors.compliant_language_processor import CompliantLanguageProcessor
from src.core.parallel_compliance_prefilter import ParallelCompliancePrefilter
# Create configuration
config = {
    "input_filtering": {
        "max_workers": 4,
        "patterns": [r"\b(password|credit_card)\b"],
        "filter_priority": {
            "sensitive_data": 1,
            "content_compliance": 2,
            "pattern_matching": 3,
            "topic_analysis": 4
        }
    },
    "semantic_monitoring": {
        "semantic_state_dim": 64,
        "sliding_window": 200,
        "concept_sensitivity": {
            "personal_data": 0.9,
            "phi": 0.9,
            "financial_data": 0.8
        }
    },
    "monitoring": {
        "interval_seconds": 60,
        "auto_optimize": True,
        "thresholds": {
            "latency_ms": 1000,
            "memory_usage_mb": 2000,
            "cache_hit_rate": 0.6
        }
    },
    "max_workers": 8
}

# Initialize processor
processor = CompliantLanguageProcessor(config)

# Generate compliant text
result = processor.generate_compliant_text(
    prompt="Please help me understand how to properly handle patient medical records.",
    context={"domain": "healthcare", "user_role": "healthcare_provider"},
    compliance_mode="strict",
    max_tokens=500,
    rag_enabled=True
)

# Process the result
if result['is_compliant']:
    print("Compliant response generated:")
    print(result['text'])
    print(f"Compliance score: {result['compliance_score']}")
else:
    print("Compliance error:", result['compliance_error'])




import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.compliance.compliance_dataset import load_gdpr_dataset
from src.compliance.compliance_aware_distillation import ComplianceAwareDistillation

def main():
    """
    Example implementation of compliance-aware distillation for GDPR
    """
    # Load teacher model
    teacher_model_name = "company/compliance-model-large"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    
    # Configure student architecture
    student_architecture = {
        "vocab_size": teacher_tokenizer.vocab_size,
        "hidden_size": 768,          # Smaller than teacher
        "num_hidden_layers": 6,      # Fewer layers than teacher
        "num_attention_heads": 12,
        "intermediate_size": 2048,   # Smaller intermediate size
        "max_position_embeddings": 1024
    }
    
    # Define regulatory framework
    regulatory_framework = "GDPR"
    
    # Initialize the distillation framework
    distiller = ComplianceAwareDistillation(
        teacher_model=teacher_model,
        student_architecture=student_architecture,
        regulatory_framework=regulatory_framework
    )
    
    # Attach tokenizer to the distiller (needed for token-based constraints)
    distiller.tokenizer = teacher_tokenizer
    
    # Initialize purpose embedder (for purpose limitation constraint)
    distiller.purpose_embedder = create_purpose_embedder(teacher_model.config.hidden_size)
    
    # Print initial model sizes
    print(f"Teacher model parameters: {distiller.count_parameters(teacher_model):,}")
    print(f"Initial student architecture: {student_architecture}")
    
    # Perform distillation
    student_model = distiller.distill(
        epochs=20,
        batch_size=16,
        temperature=2.0,
        alpha=0.7  # Balance between distillation (higher) and compliance (lower)
    )
    
    # Generate compliance report
    report = distiller.generate_compliance_report()
    
    # Print key metrics
    print("\nDistillation Results:")
    print(f"Compression ratio: {report['model_comparison']['compression_ratio']:.2f}x")
    print(f"Teacher compliance: {report['compliance_comparison']['teacher']['overall']:.4f}")
    print(f"Student compliance: {report['compliance_comparison']['student']['overall']:.4f}")
    print(f"Compliance preservation: {report['conclusion']['compliance_preservation']}")
    print(f"Status: {report['conclusion']['status']}")
    print(f"Message: {report['conclusion']['message']}")
    
    # Save student model
    student_model.save_pretrained("gdpr_compliant_student_model")
    teacher_tokenizer.save_pretrained("gdpr_compliant_student_model")
    
    print("\nSaved compliant student model to: gdpr_compliant_student_model/")
    
    return student_model, report


def create_purpose_embedder(hidden_size):
    """
    Create a purpose embedding module for purpose limitation constraints
    
    Args:
        hidden_size: Embedding dimension size
        
    Returns:
        Function that embeds purpose statements
    """
    # Simple text encoder (in practice, would use a proper text encoder)
    # This example uses a dummy encoder for demonstration
    def encode_purpose(purpose_text):
        # Dummy embedding function - would be replaced with actual encoder
        # In a real implementation, this would use a text encoder to create
        # semantic embeddings of purpose statements
        
        # Random embedding for demonstration
        return torch.randn(hidden_size)
    
    return encode_purpose


if __name__ == "__main__":
    main()