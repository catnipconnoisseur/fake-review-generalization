"""
src/models/baseline.py — Classical ML Baselines (TF-IDF + LogReg / LinearSVC).

Implements hyperparameter-tuned classical baselines:
1. Logistic Regression with L2 regularization and liblinear solver.
2. Linear Support Vector Classification (LinearSVC).

Strict leakage-free validation:
- Amazon models use GroupKFold on prefix group_ids.
- DOSC models use StratifiedKFold.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.svm import LinearSVC

from src.features.tfidf_features import DualTfidfVectorizer


class BaselineModel:
    """
    Encapsulates a fitted DualTfidfVectorizer and classical classifier.

    Provides high-level inference methods accepting raw text strings,
    computing either discrete predictions or continuous decision scores
    (for ROC-AUC analysis), as well as feature importance extraction.
    """

    def __init__(
        self,
        model_id: str,
        vectorizer: DualTfidfVectorizer,
        classifier: Any,
        best_params: Optional[Dict[str, Any]] = None,
        cv_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.best_params = best_params or {}
        self.cv_score = cv_score
        self.metadata = metadata or {}

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        """Predict binary labels (0=genuine, 1=fake) from raw texts."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_scores(self, texts: Iterable[str]) -> np.ndarray:
        """
        Compute continuous decision scores for ROC-AUC evaluation.
        Uses predict_proba[:, 1] for LogisticRegression and
        decision_function for LinearSVC.
        """
        X = self.vectorizer.transform(texts)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)[:, 1]
        elif hasattr(self.classifier, "decision_function"):
            return self.classifier.decision_function(X)
        else:
            raise AttributeError(f"Classifier {type(self.classifier)} has neither predict_proba nor decision_function")

    def get_top_features(self, n_top: int = 50) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract the top n_top positive (fake/deceptive) and negative (genuine)
        feature weights from the linear model.
        """
        if not hasattr(self.classifier, "coef_"):
            raise AttributeError("Classifier does not have coef_ attribute.")

        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.classifier.coef_[0]

        top_fake_indices = np.argsort(coefs)[-n_top:][::-1]
        top_genuine_indices = np.argsort(coefs)[:n_top]

        top_fake = [(feature_names[i], float(coefs[i])) for i in top_fake_indices]
        top_genuine = [(feature_names[i], float(coefs[i])) for i in top_genuine_indices]

        return {
            "top_fake": top_fake,
            "top_genuine": top_genuine,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize complete model bundle to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaselineModel":
        """Load serialized model bundle from disk."""
        obj = joblib.load(filepath)
        if not isinstance(obj, BaselineModel):
            raise TypeError(f"Expected BaselineModel, got {type(obj)}")
        return obj


def train_baseline_model(
    model_id: str,
    train_texts: Iterable[str],
    train_labels: Iterable[int],
    groups: Optional[Iterable[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    n_jobs: int = -1,
) -> BaselineModel:
    """
    Train a classical baseline model (LogReg or LinearSVC) with Dual TF-IDF features
    and cross-validated hyperparameter tuning.

    Parameters:
    -----------
    model_id : str
        One of 'A_logreg', 'A_svm', 'B_logreg', 'B_svm'
    train_texts : Iterable[str]
        Raw text documents for training
    train_labels : Iterable[int]
        Binary ground truth labels (0=genuine, 1=fake)
    groups : Optional[Iterable[str]]
        Group IDs (e.g. prefix group_id for Amazon dataset).
        If provided, GroupKFold is used to prevent prefix leakage in CV.
    cfg : Optional[Dict[str, Any]]
        Configuration dictionary from config.yaml
    n_jobs : int
        Number of parallel jobs for GridSearchCV

    Returns:
    --------
    BaselineModel
        Fitted model bundle ready for inference and evaluation
    """
    cfg = cfg or {}
    seed = cfg.get("seed", 42)
    cv_folds = cfg.get("cross_eval", {}).get("cv_folds", 5)

    texts_list = list(train_texts)
    y = np.array(list(train_labels), dtype=int)

    # 1. Feature Extraction: Fit DualTfidfVectorizer on training set only
    vectorizer = DualTfidfVectorizer.from_config(cfg)
    X_train = vectorizer.fit_transform(texts_list)

    # 2. Configure Cross-Validation Strategy
    groups_arr = np.array(list(groups)) if groups is not None else None
    if groups_arr is not None:
        cv = GroupKFold(n_splits=cv_folds)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # 3. Configure Classifier and Hyperparameter Search Grid
    is_logreg = "logreg" in model_id.lower()
    is_svm = "svm" in model_id.lower()

    if is_logreg:
        logreg_cfg = cfg.get("models", {}).get("logreg", {})
        c_grid = logreg_cfg.get("C", [0.01, 0.1, 1.0, 10.0])
        solver = logreg_cfg.get("solver", "liblinear")
        class_weight = logreg_cfg.get("class_weight", "balanced")
        max_iter = logreg_cfg.get("max_iter", 1000)

        estimator = LogisticRegression(
            solver=solver,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=seed,
        )
        param_grid = {"C": c_grid}

    elif is_svm:
        svm_cfg = cfg.get("models", {}).get("svm", {})
        c_grid = svm_cfg.get("C", [0.01, 0.1, 1.0, 10.0])
        class_weight = svm_cfg.get("class_weight", "balanced")
        max_iter = svm_cfg.get("max_iter", 5000)

        estimator = LinearSVC(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=seed,
        )
        param_grid = {"C": c_grid}
    else:
        raise ValueError(f"Unsupported model algorithm in model_id: {model_id}")

    # 4. Run GridSearchCV with macro/binary F1 scoring
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )

    fit_params = {}
    if groups_arr is not None:
        fit_params["groups"] = groups_arr

    grid_search.fit(X_train, y, **fit_params)

    # 5. Pack into BaselineModel container
    model = BaselineModel(
        model_id=model_id,
        vectorizer=vectorizer,
        classifier=grid_search.best_estimator_,
        best_params=grid_search.best_params_,
        cv_score=float(grid_search.best_score_),
        metadata={
            "seed": seed,
            "cv_folds": cv_folds,
            "n_train_samples": len(texts_list),
            "n_features": vectorizer.n_features,
            "group_cv_used": groups_arr is not None,
        },
    )

    return model
