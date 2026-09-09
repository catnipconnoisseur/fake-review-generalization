"""
tests/test_transformer.py — Unit tests for transformer dataset and wrapper.
"""

import pytest
import torch

from src.data.dataset import ReviewDataset
from src.models.transformer_ft import get_default_device


def test_review_dataset_without_tokenizer():
    texts = ["Review one", "Review two"]
    labels = [0, 1]
    dataset = ReviewDataset(texts, labels)

    assert len(dataset) == 2
    item = dataset[0]
    assert item["text"] == "Review one"
    assert item["labels"].item() == 0


def test_default_device_selection():
    device = get_default_device()
    assert isinstance(device, torch.device)
    assert device.type in ["mps", "cuda", "cpu"]
