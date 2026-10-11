"""
cross_corpus_pipeline.py
========================
Cross-corpus (source -> target) evaluation of five classical classifiers for spam detection.

For every experiment (source corpus -> target corpus) and every feature configuration
(text / metadata / text_metadata) each classifier is trained ONCE on the 27,000-email source
training subset and evaluated ONCE on the fixed target sample.

Order of the file
  0. Configuration
  1. Preprocessing (shared by every corpus: metadata is extracted from the RAW text, then text is normalised)
  2. Feature configurations (text / metadata / text+metadata)
  3. Models (identical hyperparameters to the within-corpus experiments)
  4. Evaluation of one (experiment, feature configuration, model) triple
  5. Main loop and output files

Outputs (OUT_DIR)
  cross_corpus_results.csv       metrics, confusion counts, timing and model size for every run
  cross_corpus_predictions.csv   one row per target email and run: source, target, features, model, id, y, p
  run_metadata.json              package versions, seeds and settings used
"""
import json
import pickle
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# =====================================================================
# 0. CONFIGURATION
# =====================================================================
SEED = 42
OUT_DIR = Path("outputs_cross_corpus")

TEXT_COL = "text"
LABEL_COL = "label"                      # 1 = spam, 0 = legitimate (ham)
META_COLS = ["num_exclamations", "num_dollars", "num_urls",
             "num_emails", "num_hashes", "num_digits"]

# Identical TF-IDF settings for every experiment (fitted on the source training data only)
TFIDF_PARAMS = dict(max_features=20000, ngram_range=(1, 2))

FEATURE_CONFIGS = ["text", "metadata", "text_metadata"]

# Each experiment: files of the source training subset (27,000 emails) and the target sample.
#   fmt = "processed": the file already holds cleaned `text` + the six metadata columns + `label`
#   fmt = "raw":       the file holds raw `text` + `label`; it is preprocessed here (also removes duplicate texts)
EXPERIMENTS = [
    dict(source="B", target="C", train_path="B_train_27000.pkl",
         test_path="dataset3.csv", test_fmt="raw"),
    # dict(source="A", target="C", train_path="A_train_27000.pkl", test_path="dataset3.csv", test_fmt="raw"),
    # dict(source="A", target="B", train_path="A_train_27000.pkl", test_path="B_target_9956.pkl", test_fmt="processed"),
    # dict(source="B", target="A", train_path="B_train_27000.pkl", test_path="A_target_9956.pkl", test_fmt="processed"),
]

MODELS_TO_RUN = ["Naive Bayes", "Linear SVM", "Logistic Regression", "XGBoost", "Random Forest"]
SAVE_PREDICTIONS = True                  # needed for exact confusion matrices and exact bootstrap CIs

# =====================================================================
# 1. PREPROCESSING  (the same code must be applied to every corpus)
# =====================================================================
def extract_special_features(raw_text):
    """Six metadata features, counted on the RAW (lower-cased) text BEFORE any cleaning."""
    return {
        "num_exclamations": raw_text.count("!"),
        "num_dollars": raw_text.count("$"),
        "num_urls": len(re.findall(r"http[s]?://\S+", raw_text)),
        "num_emails": len(re.findall(r"\S+@\S+", raw_text)),
        "num_hashes": raw_text.count("#"),
        "num_digits": sum(c.isdigit() for c in raw_text),
    }


def preprocess_corpus(df, text_col=TEXT_COL, label_col=LABEL_COL, drop_duplicates=True):
    """Raw dataframe -> dataframe with cleaned `text`, six metadata columns, `label`, `record_id`."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    df = df.copy()
    df[text_col] = df[text_col].apply(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)
    n_before = len(df)
    if drop_duplicates:                                   # exact-duplicate texts (e.g. 10,000 -> 9,956 for Dataset C)
        df = df.drop_duplicates(subset=[text_col])
    print(f"  preprocess: {n_before} -> {len(df)} records (removed {n_before - len(df)} exact duplicates)")

    cleaned, meta = [], []
    for raw in df[text_col]:
        raw = str(raw).lower()
        meta.append(extract_special_features(raw))        # 1) metadata from the raw text
        text = re.sub(r"[^a-z\s]", "", raw)               # 2) text normalisation
        tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
        cleaned.append(" ".join(tokens))

    out = pd.concat([pd.Series(cleaned, name=TEXT_COL), pd.DataFrame(meta)], axis=1)
    out[LABEL_COL] = df[label_col].astype(int).values
    out["record_id"] = df.index.values                    # original row index -> link to the manifest
    return out.reset_index(drop=True)


def load_corpus(path, fmt):
    path = str(path)
    if fmt == "processed":
        df = pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)
        if "record_id" not in df.columns:
            df = df.assign(record_id=df.index.values)
        return df.reset_index(drop=True)
    raw = pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)
    return preprocess_corpus(raw)


# =====================================================================
# 2. FEATURE CONFIGURATIONS
# =====================================================================
def build_preprocessor(config):
    """ColumnTransformer for one feature configuration. The vectoriser is fitted on the training data only."""
    if config == "text":
        # --- TEXT ONLY: TF-IDF of the cleaned text
        return ColumnTransformer([("text", TfidfVectorizer(**TFIDF_PARAMS), TEXT_COL)])
    if config == "metadata":
        # --- METADATA ONLY: the six raw (unscaled, non-negative) count features
        return ColumnTransformer([("meta", "passthrough", META_COLS)])
    if config == "text_metadata":
        # --- TEXT + METADATA: TF-IDF block concatenated with the six raw count features (no scaling)
        return ColumnTransformer([("text", TfidfVectorizer(**TFIDF_PARAMS), TEXT_COL),
                                  ("meta", "passthrough", META_COLS)])
    raise ValueError(config)


# =====================================================================
# 3. MODELS  (same settings as in the within-corpus experiments)
# =====================================================================
def build_models():
    models = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Linear SVM": LinearSVC(C=1.0, max_iter=20000, random_state=SEED),
        "Logistic Regression": LogisticRegression(solver="saga", C=0.1, max_iter=20000, tol=1e-3,
                                                  random_state=SEED, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
    }
    if "XGBoost" in MODELS_TO_RUN:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                          colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
                                          eval_metric="logloss")
    return {k: models[k] for k in MODELS_TO_RUN}


# =====================================================================
# 4. EVALUATION OF ONE RUN
# =====================================================================
def evaluate(model, config, X_train, y_train, X_test, y_test):
    pipe = Pipeline([("features", build_preprocessor(config)), ("classifier", model)])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)                            # fit once: TF-IDF/passthrough + classifier
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = pipe.predict(X_test)                           # includes the feature transformation of the target
    infer_time = time.perf_counter() - t0

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = dict(
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        mcc=matthews_corrcoef(y_test, pred),
        specificity=tn / (tn + fp), fpr=fp / (tn + fp),
        TN=tn, FP=fp, FN=fn, TP=tp,
        train_time_sec=train_time,
        latency_ms_per_email=infer_time / len(X_test) * 1000,
        model_size_mb=len(pickle.dumps(pipe)) / (1024 * 1024),
    )
    return metrics, pred


# =====================================================================
# 5. MAIN
# =====================================================================
def main():
    OUT_DIR.mkdir(exist_ok=True)
    results, predictions = [], []

    for exp in EXPERIMENTS:
        src, tgt = exp["source"], exp["target"]
        print(f"\n=== {src} -> {tgt} ===")
        train = load_corpus(exp["train_path"], "processed")
        test = load_corpus(exp["test_path"], exp["test_fmt"])
        X_train, y_train = train[[TEXT_COL] + META_COLS], train[LABEL_COL].astype(int)
        X_test, y_test = test[[TEXT_COL] + META_COLS], test[LABEL_COL].astype(int)
        print(f"  train n={len(train)} (spam share {y_train.mean():.4f}) | test n={len(test)} (spam share {y_test.mean():.4f})")

        for config in FEATURE_CONFIGS:
            for name, model in build_models().items():
                print(f"  [{config}] {name} ...", end=" ", flush=True)
                m, pred = evaluate(model, config, X_train, y_train, X_test, y_test)
                print(f"F1={m['f1']:.4f}")
                results.append(dict(source=src, target=tgt, features=config, model=name, **m))
                if SAVE_PREDICTIONS:
                    predictions.append(pd.DataFrame(dict(source=src, target=tgt, features=config, model=name,
                                                         id=test["record_id"].values, y=y_test.values, p=pred)))
        # save after every experiment so a crash does not lose finished runs
        pd.DataFrame(results).to_csv(OUT_DIR / "cross_corpus_results.csv", index=False)
        if SAVE_PREDICTIONS:
            pd.concat(predictions).to_csv(OUT_DIR / "cross_corpus_predictions.csv", index=False)

    meta = dict(seed=SEED, tfidf=TFIDF_PARAMS, meta_cols=META_COLS, feature_configs=FEATURE_CONFIGS,
                experiments=EXPERIMENTS, python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__)
    try:
        import xgboost
        meta["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print("\nDone ->", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
