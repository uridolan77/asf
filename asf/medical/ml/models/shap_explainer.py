"""
SHAP explainer for the Medical Research Synthesizer.

This module provides a SHAP-based explainer for model predictions.
"""

import logging
import shap
import matplotlib.pyplot as plt
import io
import base64


logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    SHAP-based explainer for model predictions.

    This class provides methods for explaining model predictions using SHAP.
    """

    def __init__(self, model_fn: Optional[callable] = None, tokenizer: Optional[Any] = None):
        """
        Initialize the SHAP explainer.

        Args:
            model_fn: Model prediction function (optional)
            tokenizer: Tokenizer for text data (optional)
        """
        self.model_fn = model_fn
        self.tokenizer = tokenizer
        self.explainer = None

        logger.info("SHAP explainer initialized")

    def initialize(self, model_fn: callable, tokenizer: Any = None):
        """
        Initialize the SHAP explainer with a model function and tokenizer.

        Args:
            model_fn: Model prediction function
            tokenizer: Tokenizer for text data
        """
        self.model_fn = model_fn
        self.tokenizer = tokenizer
        self.explainer = None

        logger.info("SHAP explainer initialized with model function and tokenizer")

    def is_initialized(self) -> bool:
        """
        Check if the SHAP explainer is initialized.

        Returns:
            True if the SHAP explainer is initialized, False otherwise
        """
        return self.model_fn is not None

    def _initialize_explainer(self, background_data: List[str]):
        """
        Initialize the SHAP explainer with background data.

        Args:
            background_data: Background data for SHAP
        """
        if self.tokenizer:
            self.explainer = shap.Explainer(self.model_fn, self.tokenizer)
        else:
            self.explainer = shap.KernelExplainer(self.model_fn, background_data)

        logger.info("SHAP explainer initialized with background data")

    def explain_text(
        self,
        text: str,
        background_data: Optional[List[str]] = None,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for text explanation")

        if self.explainer is None and background_data is not None:
            self._initialize_explainer(background_data)

        if self.explainer is None:
            raise ValueError("Explainer has not been initialized")

        shap_values = self.explainer([text])

        tokens = shap_values.data[0]
        values = shap_values.values[0]

        token_importance = {}
        for token, value in zip(tokens, values):
            if token.strip():
                token_importance[token] = float(value)

        sorted_tokens = sorted(
            token_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_words = [token for token, _ in sorted_tokens[:10]]
        top_values = [value for _, value in sorted_tokens[:10]]

        positive_words = [token for token, value in sorted_tokens if value > 0][:5]
        negative_words = [token for token, value in sorted_tokens if value < 0][:5]

        summary = f"The prediction is influenced positively by {', '.join(positive_words)}"
        if negative_words:
            summary += f" and negatively by {', '.join(negative_words)}"

        plt.figure(figsize=(10, 6))
        plt.barh([token for token, _ in sorted_tokens[:10]], [value for _, value in sorted_tokens[:10]])
        plt.xlabel("SHAP Value")
        plt.title("Top Influential Words")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        explanation = {
            "token_importance": token_importance,
            "top_words": top_words,
            "top_values": top_values,
            "summary": summary,
            "visualization": f"data:image/png;base64,{img_str}"
        }

        return explanation

    def explain_contradiction(
        self,
        claim1: str,
        claim2: str,
        background_data: Optional[List[str]] = None,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        combined_text = f"{claim1} [SEP] {claim2}"

        explanation = self.explain_text(
            combined_text,
            background_data,
            num_samples
        )

        explanation["claim1"] = claim1
        explanation["claim2"] = claim2

        return explanation
