"""BioBERT NER wrapper using HuggingFace Hub."""

from typing import List, Tuple
import os

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class BioBertNER:
    """Wrapper around a BioBERT model for NER tagging."""

    def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
        """Load the model using an optional HuggingFace token."""
        hf_token = os.getenv("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            token=hf_token,
        )
        self.pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )

    def tag(self, items: List[Tuple[int, str, list]]) -> List[Tuple[int, str, str]]:
        """Tag a list of OCR items.

        Args:
            items: List of tuples (id, text, coord)

        Returns:
            List of tuples (id, text, entity_label)
        """
        results = []
        for idx, text, _ in items:
            entities = self.pipeline(text)
            label = entities[0]["entity_group"] if entities else "0"
            results.append((idx, text, label))
        return results
