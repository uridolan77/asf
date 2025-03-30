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