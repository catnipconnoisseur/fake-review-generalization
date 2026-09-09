"""
src/models/registry.py — Central Model Registry.

Defines the 8 model architectures across both datasets:
- Amazon (A_): A_logreg, A_svm, A_bert, A_roberta
- DOSC (B_):   B_logreg, B_svm, B_bert, B_roberta

Provides standardized discovery, metadata inspection, and serialization loaders.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.models.baseline import BaselineModel

# Canonical list of 8 experiment models
EXPERIMENT_MODELS: Dict[str, Dict[str, Any]] = {
    "A_logreg": {
        "dataset": "amazon",
        "family": "classical",
        "algorithm": "logreg",
        "description": "Logistic Regression with Dual TF-IDF trained on Amazon GPT-2 synthetic reviews",
    },
    "A_svm": {
        "dataset": "amazon",
        "family": "classical",
        "algorithm": "svm",
        "description": "Linear SVC with Dual TF-IDF trained on Amazon GPT-2 synthetic reviews",
    },
    "A_bert": {
        "dataset": "amazon",
        "family": "transformer",
        "algorithm": "bert",
        "description": "BERT-base-uncased fine-tuned on Amazon GPT-2 synthetic reviews",
    },
    "A_roberta": {
        "dataset": "amazon",
        "family": "transformer",
        "algorithm": "roberta",
        "description": "RoBERTa-base fine-tuned on Amazon GPT-2 synthetic reviews",
    },
    "B_logreg": {
        "dataset": "dosc",
        "family": "classical",
        "algorithm": "logreg",
        "description": "Logistic Regression with Dual TF-IDF trained on DOSC MTurk deceptive reviews",
    },
    "B_svm": {
        "dataset": "dosc",
        "family": "classical",
        "algorithm": "svm",
        "description": "Linear SVC with Dual TF-IDF trained on DOSC MTurk deceptive reviews",
    },
    "B_bert": {
        "dataset": "dosc",
        "family": "transformer",
        "algorithm": "bert",
        "description": "BERT-base-uncased fine-tuned on DOSC MTurk deceptive reviews",
    },
    "B_roberta": {
        "dataset": "dosc",
        "family": "transformer",
        "algorithm": "roberta",
        "description": "RoBERTa-base fine-tuned on DOSC MTurk deceptive reviews",
    },
}


def get_model_info(model_id: str) -> Dict[str, Any]:
    """Retrieve metadata for a registered model ID."""
    if model_id not in EXPERIMENT_MODELS:
        raise KeyError(
            f"Unknown model_id '{model_id}'. Registered models: {list(EXPERIMENT_MODELS.keys())}"
        )
    return EXPERIMENT_MODELS[model_id]


def is_classical(model_id: str) -> bool:
    """Check if model belongs to classical ML family (LogReg / SVM)."""
    return get_model_info(model_id)["family"] == "classical"


def is_transformer(model_id: str) -> bool:
    """Check if model belongs to transformer family (BERT / RoBERTa)."""
    return get_model_info(model_id)["family"] == "transformer"


def list_models(
    dataset: Optional[str] = None,
    family: Optional[str] = None,
) -> List[str]:
    """List registered model IDs filtered by dataset and/or family."""
    results = []
    for m_id, info in EXPERIMENT_MODELS.items():
        if dataset and info["dataset"] != dataset:
            continue
        if family and info["family"] != family:
            continue
        results.append(m_id)
    return results


def load_model(model_id: str, models_dir: Union[str, Path]) -> Any:
    """
    Load a trained model artifact from the models directory.
    Dispatches to appropriate loader based on model family.
    """
    models_dir = Path(models_dir)
    info = get_model_info(model_id)

    if info["family"] == "classical":
        model_path = models_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Classical model artifact not found at {model_path}")
        return BaselineModel.load(model_path)
    elif info["family"] == "transformer":
        # Placeholder for transformer checkpoint loader (Phase 3)
        model_dir = models_dir / model_id
        if not model_dir.exists():
            raise FileNotFoundError(f"Transformer model directory not found at {model_dir}")
        from src.models.transformer_ft import load_transformer_model
        return load_transformer_model(model_dir)
    else:
        raise ValueError(f"Unknown family '{info['family']}' for model {model_id}")


def save_model(model: Any, model_id: str, models_dir: Union[str, Path]) -> Path:
    """
    Save a trained model artifact to the models directory.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    info = get_model_info(model_id)

    if info["family"] == "classical":
        out_path = models_dir / f"{model_id}.joblib"
        model.save(out_path)
        return out_path
    elif info["family"] == "transformer":
        out_dir = models_dir / model_id
        model.save_pretrained(out_dir)
        return out_dir
    else:
        raise ValueError(f"Unknown family '{info['family']}' for model {model_id}")
