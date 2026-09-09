"""
src/data/dataset.py — PyTorch Dataset for Transformer Models.

Wraps tokenized text sequences and binary classification labels for
BERT and RoBERTa fine-tuning pipelines.
"""

from typing import Any, Dict, List, Optional, Union


class ReviewDataset:
    """
    PyTorch-compatible dataset for review texts.

    Supports right-truncation to max_length (default 256) and dynamic padding
    or pre-tokenized tensor conversion.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Any = None,
        max_length: int = 256,
    ):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch

        text = str(self.texts[idx])
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        else:
            item = {"text": text}

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
