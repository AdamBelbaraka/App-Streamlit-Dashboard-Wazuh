"""
Wazuh ML - Severity classification
Objectif: entrainer et comparer Logistic Regression, Random Forest et XGBoost pour predire la severity.
Notebook complet, avec preprocessing detaille, tuning hyperparametres, evaluation et visualisations.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

try:
    import xgboost as xgb
except Exception:
    xgb = None

SEED = 42
np.random.seed(SEED)

plt.rcParams["figure.figsize"] = (7, 4)

# Chargement du dataset
DATA_PATH = Path("../data/wazuh_logs.csv")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Shape:", df.shape)
df.head()

# Apercu et types de colonnes
print(df.dtypes)
print("Missing rate top 10:\n", df.isna().mean().sort_values(ascending=False).head(10))

# Nettoyage et normalisation du label severity
raw_severity = df.get("severity", pd.Series([None] * len(df)))
sev = raw_severity.astype(str).str.lower().str.strip()

mapping = {
    "low": "low",
    "medium": "medium",
    "med": "medium",
    "high": "high",
    "critical": "critical",
    "crit": "critical",
}

sev = sev.map(mapping)

valid = {"low", "medium", "high", "critical"}
mask_valid = sev.isin(valid)

if (~mask_valid).any():
    bad_count = int((~mask_valid).sum())
    print(f"Dropping {bad_count} rows with invalid severity.")

df = df.loc[mask_valid].copy()
df["severity"] = sev[mask_valid]

print("Severity distribution:\n", df["severity"].value_counts())

# Encodage label pour compatibilite modele
label_encoder = LabelEncoder()
df["severity_encoded"] = label_encoder.fit_transform(df["severity"])
class_labels = list(label_encoder.classes_)
print("Label mapping:", dict(zip(class_labels, range(len(class_labels)))))

# Nettoyage NA (prepare les imputers, pas encore fit)
text_imputer = SimpleImputer(strategy="constant", fill_value="")
cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")

# Feature engineering timestamp
if "@timestamp" in df.columns:
    ts = pd.to_datetime(df["@timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["_ts"] = ts
    df = df.drop(columns=["@timestamp"])
    print("Timestamp features added: hour, dayofweek, month")
else:
    df["hour"] = np.nan
    df["dayofweek"] = np.nan
    df["month"] = np.nan
    print("@timestamp not found, using NaN for time features")

# Selection des features et nettoyage texte minimal
def clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.lower()
    cleaned = cleaned.str.replace(r"[\x00-\x1f]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned

# Candidats texte
text_candidates = [
    "rule.description",
    "data.win.eventdata.commandLine",
    "raw",
    "rule.mitre.id",
]

# Candidats categoriels
cat_candidates = ["agent.name", "location", "agent.id", "rule.id"]

# Numeriques (rule.level EXCLU pour eviter data leakage)
num_candidates = ["hour", "dayofweek", "month"]

existing_cols = set(df.columns)
text_cols = [c for c in text_candidates if c in existing_cols]
cat_cols = [c for c in cat_candidates if c in existing_cols]
num_cols = [c for c in num_candidates if c in existing_cols]

# Nettoyage texte
for c in text_cols:
    df[c] = clean_text_series(df[c])

# Nettoyage specifique de raw pour masquer les infos de fuite
if "raw" in df.columns:
    patterns = [
        r'"level"\\s*:\\s*\\d+',
        r"'level'\\s*:\\s*\\d+",
        r'"severity"\\s*:\\s*"[a-zA-Z]+"',
        r"'severity'\\s*:\\s*'[a-zA-Z]+'",
        r'"rule"\\s*:\\s*\\{[^}]*"level"\\s*:\\s*\\d+[^}]*\\}',
        r"'rule'\\s*:\\s*\\{[^}]*'level'\\s*:\\s*\\d+[^}]*\\}",
    ]
    raw_series = df["raw"].astype(str)
    for pat in patterns:
        raw_series = raw_series.str.replace(pat, "<MASK>", regex=True)
    df["raw"] = raw_series

# Deduplication sur (rule.id + raw + rule.description) si dispo
subset = [c for c in ["rule.id", "raw", "rule.description"] if c in df.columns]
if subset:
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = before - len(df)
    print(f"Duplicates removed: {removed}")

# Drop categorical ids if cardinality too high
max_card = 50
kept_cat = []
dropped_cat = []
for c in cat_cols:
    n_unique = df[c].nunique(dropna=True)
    if n_unique <= max_card:
        kept_cat.append(c)
    else:
        dropped_cat.append(c)

cat_cols = kept_cat

print("Text cols:", text_cols)
print("Cat cols (kept):", cat_cols)
print("Cat cols (dropped):", dropped_cat)
print("Num cols:", num_cols)

# Define features and target
feature_cols = text_cols + cat_cols + num_cols
X = df[feature_cols].copy()
y = df["severity_encoded"].copy()

print("Final feature columns:", feature_cols)
print("Class labels:", class_labels)

# Split train/test (temporal si possible) + CV
if "_ts" in df.columns and df["_ts"].notna().any():
    df_sorted = df.sort_values("_ts").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df["severity_encoded"]
    X_test = test_df[feature_cols]
    y_test = test_df["severity_encoded"]

    cv = TimeSeriesSplit(n_splits=3)
    cv_splits = list(cv.split(X_train))
    print("Using temporal split + TimeSeriesSplit")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    if "rule.id" in df.columns:
        groups_train = df.loc[X_train.index, "rule.id"]
        cv = GroupKFold(n_splits=3)
        cv_splits = list(cv.split(X_train, y_train, groups=groups_train))
        print("Using stratified split + GroupKFold(rule.id)")
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        cv_splits = cv
        print("Using stratified split + StratifiedKFold")

print("Train size:", X_train.shape, "Test size:", X_test.shape)

# Preprocessor: ColumnTransformer + pipelines
def combine_text(x):
    if hasattr(x, "fillna"):
        return x.fillna("").astype(str).agg(" ".join, axis=1)
    return pd.DataFrame(x).fillna("").astype(str).agg(" ".join, axis=1)


text_pipeline = Pipeline(
    [
        ("imputer", text_imputer),
        ("to_text", FunctionTransformer(combine_text, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ]
)

cat_pipeline = Pipeline(
    [
        ("imputer", cat_imputer),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

num_pipeline_scaled = Pipeline(
    [
        ("imputer", num_imputer),
        ("scaler", StandardScaler(with_mean=False)),
    ]
)

num_pipeline_plain = Pipeline(
    [
        ("imputer", num_imputer),
    ]
)

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

preprocess_scaled = ColumnTransformer(transformers_scaled)
preprocess_plain = ColumnTransformer(transformers_plain)

if not transformers_scaled:
    raise ValueError("No usable feature columns found. Check text/cat/num columns.")


# Helper to avoid lambda in pipelines (picklable)
def to_dense_matrix(x):
    return x.toarray() if hasattr(x, "toarray") else x


# Baseline rapide: Logistic Regression (sans tuning)
logreg_base = Pipeline(
    [
        ("preprocess", preprocess_scaled),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
    ]
)

logreg_base.fit(X_train, y_train)
print("Baseline Logistic Regression trained.")

# Logistic Regression + RandomizedSearchCV (tuning leger, rapide)
logreg = Pipeline(
    [
        ("preprocess", preprocess_scaled),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,  # suffit
                class_weight="balanced",
                penalty="l2",
                solver="liblinear",  # ou saga si dataset plus gros
            ),
        ),
    ]
)

logreg_param = {
    "clf__C": [0.01, 0.05, 0.1],  # pénalisation plus forte => moins d'overfit
}

logreg_search = RandomizedSearchCV(
    logreg,
    logreg_param,
    n_iter=5,
    scoring="f1_macro",
    cv=cv_splits,
    n_jobs=-1,
    random_state=SEED,
    verbose=1,
)

logreg_search.fit(X_train, y_train)
print("Best LogisticRegression params:", logreg_search.best_params_)

best_logreg = logreg_search.best_estimator_
y_pred = best_logreg.predict(X_test)
print("Logistic Regression Evaluation on Test Set:")
print(classification_report(y_test, y_pred))

# Random Forest + RandomizedSearchCV (anti-overfit, rapide)
# RandomForest needs dense inputs
rf = Pipeline(
    [
        ("preprocess", preprocess_plain),
        ("to_dense", FunctionTransformer(to_dense_matrix, validate=False)),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=50,  # Réduction drastique
                max_depth=6,  # Plus petit arbre
                min_samples_leaf=10,
                min_samples_split=20,
                max_features=0.3,
                bootstrap=True,
                class_weight="balanced_subsample",
                random_state=SEED,
            ),
        ),
    ]
)


rf_param = {
    "clf__max_depth": [4, 6, 8],
    "clf__min_samples_leaf": [5, 10],
}

rf_search = RandomizedSearchCV(
    rf,
    rf_param,
    n_iter=4,
    scoring="f1_macro",
    cv=cv_splits,
    n_jobs=-1,
    random_state=SEED,
    verbose=1,
)

rf_search.fit(X_train, y_train)
print("Best RandomForest params:", rf_search.best_params_)

best_rf = rf_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Random Forest Evaluation on Test Set:")
print(classification_report(y_test, y_pred))

# XGBoost + RandomizedSearchCV (anti-overfit strict, rapide)
if xgb is None:
    print("xgboost is not installed. Skipping XGBoost training.")
    xgb_search = None
else:
    # Compute sample weights for class imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_map = {cls: w for cls, w in zip(classes, class_weights)}
    sample_weight = y_train.map(weight_map).values

    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_labels),
        random_state=SEED,
        n_estimators=50,  # Réduction ici
        max_depth=3,
        learning_rate=0.2,  # plus rapide convergence
        subsample=0.6,
        colsample_bytree=0.5,
        reg_lambda=20,  # régularisation forte
        reg_alpha=10,
        gamma=5,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
    )

    xgb_pipe = Pipeline(
        [
            ("preprocess", preprocess_plain),
            ("svd", TruncatedSVD(n_components=200, random_state=SEED)),
            ("to_dense", FunctionTransformer(to_dense_matrix, validate=False)),
            ("clf", xgb_model),
        ]
    )

    xgb_param = {
        "clf__max_depth": [2, 3],
        "clf__subsample": [0.6],
        "clf__learning_rate": [0.1, 0.2],
    }

    xgb_search = RandomizedSearchCV(
        xgb_pipe,
        xgb_param,
        n_iter=2,
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
    )

    xgb_search.fit(X_train, y_train, clf__sample_weight=sample_weight)
    print("Best XGBoost params:", xgb_search.best_params_)

    best_xgb = xgb_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    print("XGBoost Evaluation on Test Set:")
    print(classification_report(y_test, y_pred))

# Fonctions d'evaluation + confusion matrix (matplotlib)
def evaluate_model(model, X_te, y_te, label):
    preds = model.predict(X_te)
    metrics = {
        "model": label,
        "accuracy": accuracy_score(y_te, preds),
        "balanced_accuracy": balanced_accuracy_score(y_te, preds),
        "f1_macro": f1_score(y_te, preds, average="macro"),
        "f1_weighted": f1_score(y_te, preds, average="weighted"),
    }

    print(f"\n[{label}] Metrics:")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    print("\nClassification report:", classification_report(y_te, preds, target_names=class_labels))

    labels = list(range(len(class_labels)))
    cm = confusion_matrix(y_te, preds, labels=labels)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=range(len(labels)), labels=class_labels, rotation=45, ha="right")
    plt.yticks(ticks=range(len(labels)), labels=class_labels)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return metrics


def plot_search_results(search, title):
    if search is None or not hasattr(search, "cv_results_"):
        return
    df = pd.DataFrame(search.cv_results_)
    score_col = "mean_test_score"
    param_cols = [c for c in df.columns if c.startswith("param_")]
    if not param_cols:
        return

    plt.figure(figsize=(7, 4))
    if len(param_cols) == 1:
        x = df[param_cols[0]].astype(str)
        y = df[score_col]
        plt.plot(x, y, marker="o")
        plt.title(title)
        plt.xlabel(param_cols[0].replace("param_", ""))
        plt.ylabel(score_col)
        plt.xticks(rotation=45, ha="right")
    elif len(param_cols) == 2:
        pivot = df.pivot_table(values=score_col, index=param_cols[0], columns=param_cols[1])
        plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.title(title)
        plt.xlabel(param_cols[1].replace("param_", ""))
        plt.ylabel(param_cols[0].replace("param_", ""))
        plt.colorbar(label=score_col)
        plt.xticks(ticks=range(len(pivot.columns)), labels=[str(x) for x in pivot.columns], rotation=45, ha="right")
        plt.yticks(ticks=range(len(pivot.index)), labels=[str(x) for x in pivot.index])
    else:
        top = df.sort_values(score_col, ascending=False).head(15)
        plt.bar(range(len(top)), top[score_col])
        plt.title(title + " (top 15)")
        plt.xlabel("trial")
        plt.ylabel(score_col)

    plt.tight_layout()
    plt.show()


# Evaluation des modeles
results = []

results.append(evaluate_model(logreg_base, X_test, y_test, "LogReg baseline"))
results.append(evaluate_model(logreg_search.best_estimator_, X_test, y_test, "LogReg tuned"))
results.append(evaluate_model(rf_search.best_estimator_, X_test, y_test, "RandomForest tuned"))

if xgb_search is not None:
    results.append(evaluate_model(xgb_search.best_estimator_, X_test, y_test, "XGBoost tuned"))

# Visualisation des recherches hyperparametres
plot_search_results(logreg_search, "LogReg hyperparameter search")
plot_search_results(rf_search, "RandomForest hyperparameter search")
plot_search_results(xgb_search, "XGBoost hyperparameter search")

# Comparaison finale + sauvegarde
results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
print(results_df)

# Bar chart comparison
plt.figure(figsize=(7, 4))
plt.bar(results_df["model"], results_df["f1_macro"], label="F1 macro")
plt.plot(results_df["model"], results_df["accuracy"], marker="o", color="orange", label="Accuracy")
plt.xticks(rotation=30, ha="right")
plt.title("Model comparison")
plt.ylabel("score")
plt.legend()
plt.tight_layout()
plt.show()

# Save best model
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

best_row = results_df.iloc[0]
model_name = best_row["model"]

if model_name == "LogReg baseline":
    best_model = logreg_base
elif model_name == "LogReg tuned":
    best_model = logreg_search.best_estimator_
elif model_name == "RandomForest tuned":
    best_model = rf_search.best_estimator_
elif model_name == "XGBoost tuned":
    best_model = xgb_search.best_estimator_
else:
    best_model = logreg_search.best_estimator_

model_path = MODELS_DIR / "best_severity_model.joblib"
joblib.dump(best_model, model_path)

metrics_path = MODELS_DIR / "severity_metrics.json"
metrics_path.write_text(json.dumps(results, indent=2))

print("Saved model:", model_path)
print("Saved metrics:", metrics_path)

print(
    "Conclusion: Logistic Regression, Random Forest et XGBoost sont compares. "
    "Le meilleur modele et ses metriques sont sauvegardes pour reutilisation."
)
