"""
src/models/transformer_ft.py — Transformer Fine-Tuning Pipeline (BERT & RoBERTa).

Implements sequence classification fine-tuning for BERT-base and RoBERTa-base:
- Automatic hardware acceleration (Apple Silicon MPS / CUDA / CPU)
- Layer freezing for low-resource DOSC transfer (freeze first N encoder layers)
- Dropout regularization (0.3 classifier dropout for small dataset)
- Early stopping based on validation F1
- PyTorch DataLoader batching and evaluation
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.dataset import ReviewDataset
from src.evaluation.metrics import compute_classification_metrics


def get_default_device() -> torch.device:
    """Select best available hardware acceleration device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class TransformerClassifier:
    """
    Wrapper for fine-tuned Transformer model and tokenizer.
    Provides batch inference methods returning discrete labels and continuous probabilities.
    """

    def __init__(
        self,
        model_id: str,
        model: Any,
        tokenizer: Any,
        device: Optional[torch.device] = None,
        max_length: int = 256,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or get_default_device()
        self.max_length = max_length
        self.metadata = metadata or {}

        self.model.to(self.device)
        self.model.eval()

    def predict_and_scores(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both discrete labels and continuous probabilities in a single
        vectorized forward pass with fast batch tokenization.
        """
        self.model.eval()
        texts_list = [str(t) for t in texts]
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(texts_list), batch_size):
                batch_texts = texts_list[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(np.argmax(probs, axis=1))
                all_scores.extend(probs[:, 1])

        return np.array(all_preds, dtype=int), np.array(all_scores, dtype=float)

    def predict(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        """Predict binary classification labels (0=genuine, 1=fake)."""
        preds, _ = self.predict_and_scores(texts, batch_size=batch_size)
        return preds

    def predict_scores(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        """Predict probability scores for positive class (1=fake) for ROC-AUC."""
        _, scores = self.predict_and_scores(texts, batch_size=batch_size)
        return scores

    def save_pretrained(self, save_directory: Union[str, Path]) -> Path:
        """Save fine-tuned weights and tokenizer to directory."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return path


def load_transformer_model(model_directory: Union[str, Path]) -> TransformerClassifier:
    """Load fine-tuned model and tokenizer from directory."""
    path = Path(model_directory)
    model_id = path.name
    device = get_default_device()

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)

    return TransformerClassifier(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def apply_layer_freezing(model: Any, n_layers_to_freeze: int = 8) -> None:
    """
    Freeze first n_layers_to_freeze layers of encoder to prevent overfitting
    on small datasets (e.g. DOSC).
    """
    # Check if BERT or RoBERTa architecture
    base = getattr(model, "bert", getattr(model, "roberta", None))
    if base is not None and hasattr(base, "encoder"):
        layers = base.encoder.layer
        for layer in layers[:n_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False


def train_transformer(
    model_id: str,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    cfg: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> TransformerClassifier:
    """
    Fine-tune a Transformer model (BERT or RoBERTa) with early stopping on validation F1.
    """
    cfg = cfg or {}
    device = device or get_default_device()

    # Determine base model architecture
    is_roberta = "roberta" in model_id.lower()
    t_cfg = cfg.get("transformers", {}).get("roberta" if is_roberta else "bert", {})
    dosc_overrides = cfg.get("transformers", {}).get("dosc_overrides", {})

    base_name = t_cfg.get("model_name", "roberta-base" if is_roberta else "bert-base-uncased")
    max_length = t_cfg.get("max_length", 256)
    batch_size = t_cfg.get("batch_size", 16)
    learning_rate = float(t_cfg.get("learning_rate", 2e-5))
    epochs = int(t_cfg.get("epochs", 4))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    patience = int(t_cfg.get("early_stopping_patience", 2))

    # Layer freezing and dropout
    is_dosc = model_id.startswith("B_")
    dropout_prob = dosc_overrides.get("classifier_dropout", 0.3) if is_dosc else 0.1
    if is_dosc:
        freeze_layers = dosc_overrides.get("freeze_layers", 8)
    elif device.type == "mps":
        freeze_layers = 6  # Accelerate MPS fine-tuning by freezing bottom 6 encoder layers
    else:
        freeze_layers = 0

    print(f"  Initializing {base_name} on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    model_config = AutoConfig.from_pretrained(
        base_name,
        num_labels=2,
        classifier_dropout=dropout_prob,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        config=model_config,
    )

    if freeze_layers > 0:
        apply_layer_freezing(model, freeze_layers)
        print(f"  Frozen first {freeze_layers} encoder layers (MPS / Transfer Regularization)")

    model.to(device)

    # Pre-tokenize all texts in batch using FastTokenizer
    print(f"  Pre-tokenizing {len(train_texts)} train & {len(val_texts)} val texts...")
    enc_train = tokenizer(
        [str(t) for t in train_texts],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    enc_val = tokenizer(
        [str(t) for t in val_texts],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    from torch.utils.data import TensorDataset, Subset

    train_dataset = TensorDataset(
        enc_train["input_ids"],
        enc_train["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        enc_val["input_ids"],
        enc_val["attention_mask"],
        torch.tensor(val_labels, dtype=torch.long),
    )

    # For fast validation during training, cap val set to 1000
    val_eval_dataset = val_dataset
    if len(val_dataset) > 1000:
        val_eval_dataset = Subset(val_dataset, list(range(1000)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_eval_dataset, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * t_cfg.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_weights = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, (b_input_ids, b_mask, b_labels) in enumerate(train_loader):
            input_ids = b_input_ids.to(device)
            attention_mask = b_mask.to(device)
            labels = b_labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if device.type == "mps":
                torch.mps.synchronize()
                if (step + 1) % 25 == 0:
                    torch.mps.empty_cache()

            total_train_loss += loss.item()

            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                print(f"    Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_preds = []
        val_scores = []
        val_true = []
        with torch.no_grad():
            for b_input_ids, b_mask, b_y in val_loader:
                input_ids = b_input_ids.to(device)
                attention_mask = b_mask.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(np.argmax(probs, axis=1))
                val_scores.extend(probs[:, 1])
                val_true.extend(b_y.numpy())

        val_metrics = compute_classification_metrics(val_true, val_preds, val_scores)
        cur_f1 = val_metrics["f1"]

        print(f"  Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val F1: {cur_f1:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

        # Check early stopping on F1
        if cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (best Val F1: {best_val_f1:.4f})")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return TransformerClassifier(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        metadata={"best_val_f1": best_val_f1},
    )
