import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

SEED = 42

SEVERITY_RULES: Tuple[Tuple[int, int, str], ...] = (
    (0, 3, "low"),
    (4, 6, "medium"),
    (7, 10, "high"),
    (11, 15, "critical"),
)
DEFAULT_SEVERITY = "low"

ProgressFn = Callable[[str, Optional[float]], None]


@dataclass
class TrainingArtifacts:
    best_model_name: str
    metrics: List[Dict[str, float]]
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    classification_report: str
    model_bytes: bytes
    model_path: Path
    metrics_path: Path


def label_severity(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "severity" in result.columns:
        return result
    levels = pd.to_numeric(result.get("rule.level"), errors="coerce")
    severity = pd.Series([DEFAULT_SEVERITY] * len(result), index=result.index)
    for min_level, max_level, label in SEVERITY_RULES:
        mask = levels.between(min_level, max_level, inclusive="both")
        severity.loc[mask] = label
    result["severity"] = severity
    return result


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.lower()
    cleaned = cleaned.str.replace(r"[\x00-\x1f]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str], List[str]]:
    df = label_severity(df)

    if "@timestamp" in df.columns:
        ts = pd.to_datetime(df["@timestamp"], errors="coerce")
        df["_ts"] = ts
        df["hour"] = ts.dt.hour
        df["dayofweek"] = ts.dt.dayofweek
        df["month"] = ts.dt.month

    text_candidates = ["rule.description", "raw", "data.win.eventdata.commandLine", "rule.mitre.id"]
    cat_candidates = ["agent.name", "location"]
    num_candidates = ["hour", "dayofweek", "month"]

    existing_cols = set(df.columns)
    text_cols = [c for c in text_candidates if c in existing_cols]
    cat_cols = [c for c in cat_candidates if c in existing_cols]
    num_cols = [c for c in num_candidates if c in existing_cols]

    for c in text_cols:
        df[c] = _clean_text(df[c])

    feature_cols = text_cols + cat_cols + num_cols
    if not feature_cols:
        raise ValueError("Aucune colonne exploitable pour entraîner le modèle.")

    X = df[feature_cols].copy()
    y = df["severity"].astype(str)
    return X, y, text_cols, cat_cols, num_cols, feature_cols


def _build_preprocessors(text_cols: List[str], cat_cols: List[str], num_cols: List[str]) -> Tuple[ColumnTransformer, ColumnTransformer]:
    text_imputer = SimpleImputer(strategy="constant", fill_value="")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer = SimpleImputer(strategy="median")

    def combine_text(x):
        return pd.DataFrame(x).fillna("").astype(str).agg(" ".join, axis=1)

    text_pipeline = Pipeline(
        [
            ("imputer", text_imputer),
            ("to_text", FunctionTransformer(combine_text, validate=False)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ]
    )
    cat_pipeline = Pipeline([("imputer", cat_imputer), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_pipeline_scaled = Pipeline([("imputer", num_imputer), ("scaler", StandardScaler(with_mean=False))])
    num_pipeline_plain = Pipeline([("imputer", num_imputer)])

    transformers_scaled = []
    transformers_plain = []
    if text_cols:
        transformers_scaled.append(("text", text_pipeline, text_cols))
        transformers_plain.append(("text", text_pipeline, text_cols))
    if cat_cols:
        transformers_scaled.append(("cat", cat_pipeline, cat_cols))
        transformers_plain.append(("cat", cat_pipeline, cat_cols))
    if num_cols:
        transformers_scaled.append(("num", num_pipeline_scaled, num_cols))
        transformers_plain.append(("num", num_pipeline_plain, num_cols))

    return ColumnTransformer(transformers_scaled), ColumnTransformer(transformers_plain)


def _to_dense_matrix(x):
    return x.toarray() if hasattr(x, "toarray") else x


def _evaluate_model(model, X_te, y_te, label: str, class_labels: List[str]) -> Dict[str, float]:
    preds = model.predict(X_te)
    return {
        "model": label,
        "accuracy": accuracy_score(y_te, preds),
        "balanced_accuracy": balanced_accuracy_score(y_te, preds),
        "f1_macro": f1_score(y_te, preds, average="macro"),
        "f1_weighted": f1_score(y_te, preds, average="weighted"),
    }


def train_severity_models(df: pd.DataFrame, reporter: Optional[ProgressFn] = None) -> TrainingArtifacts:
    reporter = reporter or (lambda msg, pct=None: None)
    reporter("Préparation des features...", 0.05)
    X, y, text_cols, cat_cols, num_cols, feature_cols = prepare_features(df)
    preprocess_scaled, preprocess_plain = _build_preprocessors(text_cols, cat_cols, num_cols)

    if "_ts" in df.columns and df["_ts"].notna().any():
        df_sorted = df.sort_values("_ts").reset_index(drop=True)
        split_idx = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        X_train, y_train = train_df[feature_cols], train_df["severity"]
        X_test, y_test = test_df[feature_cols], test_df["severity"]
        cv = TimeSeriesSplit(n_splits=3)
        cv_splits = list(cv.split(X_train))
        reporter("Split temporel (80/20) + TimeSeriesSplit", 0.1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        if "rule.id" in df.columns:
            groups_train = df.loc[X_train.index, "rule.id"]
            cv = GroupKFold(n_splits=3)
            cv_splits = list(cv.split(X_train, y_train, groups=groups_train))
            reporter("Split stratifié + GroupKFold(rule.id)", 0.1)
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            cv_splits = cv
            reporter("Split stratifié + StratifiedKFold", 0.1)

    class_labels = sorted(y.unique())

    reporter("Entraînement Logistic Regression...", 0.2)
    logreg_base = Pipeline(
        [
            ("preprocess", preprocess_scaled),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs", multi_class="auto")),
        ]
    )
    logreg_base.fit(X_train, y_train)

    logreg = Pipeline(
        [
            ("preprocess", preprocess_scaled),
            (
                "clf",
                LogisticRegression(max_iter=1200, class_weight="balanced", penalty="l2", solver="lbfgs", multi_class="auto"),
            ),
        ]
    )
    logreg_param = {"clf__C": [0.01, 0.05, 0.1]}
    logreg_search = RandomizedSearchCV(
        logreg,
        logreg_param,
        n_iter=len(logreg_param["clf__C"]),
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1,
        random_state=SEED,
        verbose=0,
    )
    logreg_search.fit(X_train, y_train)

    reporter("Entraînement Random Forest...", 0.45)
    rf = Pipeline(
        [
            ("preprocess", preprocess_plain),
            ("to_dense", FunctionTransformer(_to_dense_matrix, validate=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=8,
                    min_samples_leaf=8,
                    min_samples_split=16,
                    max_features=0.4,
                    bootstrap=True,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                ),
            ),
        ]
    )
    rf_param = {"clf__max_depth": [6, 8], "clf__min_samples_leaf": [5, 8]}
    rf_search = RandomizedSearchCV(
        rf,
        rf_param,
        n_iter=3,
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1,
        random_state=SEED,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)

    xgb_search = None
    if xgb is not None:
        reporter("Entraînement XGBoost...", 0.7)
        classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight_map = {cls: w for cls, w in zip(classes, class_weights)}
        sample_weight = y_train.map(weight_map).values

        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(classes),
            random_state=SEED,
            n_estimators=80,
            max_depth=4,
            learning_rate=0.2,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=15,
            reg_alpha=8,
            gamma=4,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
        )

        xgb_pipe = Pipeline(
            [
                ("preprocess", preprocess_plain),
                ("svd", TruncatedSVD(n_components=200, random_state=SEED)),
                ("to_dense", FunctionTransformer(_to_dense_matrix, validate=False)),
                ("clf", xgb_model),
            ]
        )

        xgb_param = {"clf__max_depth": [3, 4], "clf__learning_rate": [0.1, 0.2]}
        xgb_search = RandomizedSearchCV(
            xgb_pipe,
            xgb_param,
            n_iter=2,
            scoring="f1_macro",
            cv=cv_splits,
            n_jobs=-1,
            random_state=SEED,
            verbose=0,
        )
        xgb_search.fit(X_train, y_train, clf__sample_weight=sample_weight)
    else:
        reporter("XGBoost non installé : étape ignorée.", 0.7)

    reporter("Évaluation des modèles...", 0.85)
    candidates = [
        ("LogReg baseline", logreg_base),
        ("LogReg tuned", logreg_search.best_estimator_),
        ("RandomForest tuned", rf_search.best_estimator_),
    ]
    if xgb_search is not None:
        candidates.append(("XGBoost tuned", xgb_search.best_estimator_))

    results = []
    best_model = None
    best_score = -1.0
    best_name = ""
    best_preds = None
    for name, model in candidates:
        metrics = _evaluate_model(model, X_test, y_test, name, class_labels)
        results.append(metrics)
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model = model
            best_name = name
            best_preds = model.predict(X_test)

    if best_model is None:
        raise RuntimeError("Impossible de sélectionner un modèle gagnant.")

    cm = confusion_matrix(y_test, best_preds, labels=class_labels).tolist()
    report = classification_report(y_test, best_preds, target_names=class_labels)

    reporter("Sauvegarde du meilleur modèle...", 0.95)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"best_severity_model_{timestamp}.joblib"
    metrics_path = models_dir / f"severity_metrics_{timestamp}.json"

    joblib.dump(best_model, model_path)
    metrics_path.write_text(json.dumps(results, indent=2))

    buffer = io.BytesIO()
    joblib.dump(best_model, buffer)
    buffer.seek(0)

    reporter("Terminé.", 1.0)
    return TrainingArtifacts(
        best_model_name=best_name,
        metrics=results,
        confusion_matrix=cm,
        class_labels=class_labels,
        classification_report=report,
        model_bytes=buffer.read(),
        model_path=model_path,
        metrics_path=metrics_path,
    )
