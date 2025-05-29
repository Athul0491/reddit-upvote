from __future__ import annotations
# ───────────────────────────── imports ─────────────────────────────
import argparse, math, pathlib, re, warnings
from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from   scipy.stats import gaussian_kde

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, r2_score, mean_squared_error,
)
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer, HashingTF, IDF, OneHotEncoder as OHE, VectorAssembler
)
from pyspark.ml.classification import LogisticRegression as SLR, NaiveBayes as SNB
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import RDD, StorageLevel

# optional SHAP
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    warnings.filterwarnings("ignore", message="No module named 'shap'")

# ──────────────────────────── constants ────────────────────────────
THRESHOLD        = 5
SAMPLE_FRACTION  = 0.10
SPARK_EPOCHS     = 5
SPARK_LR         = 0.25
HASH_DIM         = 1 << 18
TOKEN_RE         = re.compile(r"[a-z]{2,}")
PLOT_DIR         = pathlib.Path("plots")
PLOT_DIR.mkdir(exist_ok=True, parents=True)
sns.set_style("whitegrid")

# ────────────────────── Pandas-side helpers  ──────────────────────
REQUIRED_COLS    = ["body", "ups"]
NUMERIC_COLS     = ["stickied", "controversiality", "gilded_flag",
                    "edit_delay", "hour", "weekday"]
CATEGORICAL_COLS = ["subreddit"]
TEXT_COL         = "body"

def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if miss := [c for c in REQUIRED_COLS if c not in df.columns]:
        raise KeyError(f"Missing cols {miss}")

    df = df[~df["body"].isin({"[deleted]", "[removed]"})]
    df = df.dropna(subset=["ups"]).reset_index(drop=True)

    df["stickied"]         = df.get("stickied", 0).fillna(0).astype("uint8")
    df["controversiality"] = df.get("controversiality", 0).astype("uint8")
    df["gilded_flag"]      = (df.get("gilded", 0) > 0).astype("uint8")

    if {"edited", "created_utc"}.issubset(df.columns):
        df["edit_delay"] = np.where(
            df["edited"].astype("int64") == 0,
            0,
            df["edited"].astype("int64") - df["created_utc"].astype("int64")
        )
    else:
        df["edit_delay"] = 0

    if "created_utc" in df.columns:
        ts = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        df["hour"]    = ts.dt.hour.astype("uint8")
        df["weekday"] = ts.dt.weekday.astype("uint8")
    else:
        df["hour"], df["weekday"] = 0, 0

    df.drop(columns=["score","id","link_id","parent_id",
                     "retrieved_on","gilded","edited"],
            errors="ignore", inplace=True)
    return df

def build_preprocess() -> ColumnTransformer:
    tfidf = TfidfVectorizer(strip_accents="unicode", lowercase=True,
                            ngram_range=(1,2), min_df=5, max_df=0.8,
                            sublinear_tf=True)
    return ColumnTransformer([
        ("txt", tfidf, TEXT_COL),
        ("num", StandardScaler(with_mean=False), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
    ])

# ──────────────────────── plot helpers (unchanged) ────────────────────────
def savefig(name: str):
    plt.tight_layout()
    plt.savefig(PLOT_DIR/name, dpi=150)
    plt.close()

def prob_histogram(probs: np.ndarray, title: str, name: str):
    sns.histplot(probs, bins=35, stat="density",
                 color="#d9d9d9", edgecolor="#333333",
                 alpha=.8, label="Histogram")
    kde = gaussian_kde(probs)
    xs  = np.linspace(0,1,400)
    plt.plot(xs, kde(xs), lw=2.5, label="KDE")
    mean, median = probs.mean(), np.median(probs)
    ci_low, ci_high = np.percentile(probs, [2.5, 97.5])
    for x_, ls, lbl in [
        (ci_low,  "--", "95 % CI"),
        (ci_high, "--", None)
    ]:
        plt.axvline(x_, c="#2865c1", ls=ls, lw=1.5, label=lbl)
    plt.axvline(mean,   c="#cb4335", lw=2, label="Mean")
    plt.axvline(median, c="#239b56", lw=2, ls="-.", label="Median")
    plt.legend(frameon=False, fontsize=8)
    plt.title(title); plt.xlabel("Probability"); plt.ylabel("Density")
    savefig(name)

def plot_coef_bar(logreg_pipe, top_k=15):
    try:
        feat_names = logreg_pipe.named_steps["pre"].get_feature_names_out()
    except AttributeError:
        feat_names = np.arange(logreg_pipe.named_steps["clf"].coef_.shape[1])

    coef  = logreg_pipe.named_steps["clf"].coef_[0]
    idx   = np.argsort(coef)
    sel   = np.concatenate([idx[:top_k], idx[-top_k:]])
    vals  = coef[sel]; names = feat_names[sel]
    colors = ["#cb4335" if v < 0 else "#239b56" for v in vals]

    plt.figure(figsize=(6,7))
    plt.barh(range(len(vals)), vals, color=colors)
    plt.yticks(range(len(vals)), names, fontsize=8)
    plt.title("Top coefficients (LogReg)")
    savefig("coef_bar_logreg.png")

def plot_actual_vs_pred(y_true, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=8, alpha=.25)
    mx = max(y_true.max(), y_pred.max())
    plt.plot([0,mx],[0,mx], c="red")
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    plt.title(f"Actual vs Predicted (RMSE {rmse:.2f}, R² {r2:.2f})")
    plt.xlabel("Actual ups"); plt.ylabel("Predicted probability")
    savefig("actual_vs_pred.png")

def plot_residual_density(residuals):
    sns.histplot(residuals, kde=True, color="steelblue",
                 edgecolor="black")
    plt.axvline(0, c="red", ls="--")
    plt.title("Residual distribution")
    plt.xlabel("Residual")
    savefig("residual_density.png")

def plot_error_by_sub(df_eval):
    top = df_eval["subreddit"].value_counts().head(10).index
    sns.boxplot(data=df_eval[df_eval["subreddit"].isin(top)],
                x="absolute_error", y="subreddit", orient="h")
    plt.title("Absolute error by subreddit (top 10)")
    plt.xlabel("|error|"); plt.ylabel("")
    savefig("error_box_by_sub.png")

def plot_error_vs_length(df_eval):
    sns.scatterplot(x="length", y="absolute_error",
                    data=df_eval, alpha=.3, s=15)
    plt.title("|error| vs. comment length")
    plt.xlabel("Comment length (chars)")
    plt.ylabel("|error|")
    savefig("error_vs_length.png")

def plot_daily_mae(df_eval):
    daily = df_eval.groupby("date")["absolute_error"].mean()
    daily.plot(figsize=(8,3))
    plt.title("Daily MAE")
    plt.ylabel("MAE"); plt.xlabel("")
    savefig("daily_mae.png")

def plot_shap_beeswarm(logreg_pipe, X):
    explainer = shap.Explainer(logreg_pipe.predict_proba, X, silent=True)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False, plot_size=(8,5))
    plt.title("SHAP beeswarm")
    savefig("shap_beeswarm.png")

# ─────────────── scratch logistic regression (RDD) ───────────────
def sigmoid(z: float) -> float:
    return 1/(1+math.exp(-z))

def train_scratch_lr(rdd: RDD, dim=HASH_DIM, epochs=SPARK_EPOCHS,
                     lr=SPARK_LR) -> np.ndarray:
    w = np.zeros(dim, dtype="float32")
    n = rdd.count()

    def part_grad(iterator, w_b):
        buf = np.zeros(dim, dtype="float32")
        for y, sv in iterator:
            z = sum(w_b[i]*v for i, v in zip(sv.indices, sv.values))
            err = sigmoid(z) - y
            for i, v in zip(sv.indices, sv.values):
                buf[i] += err * v
        yield buf

    for ep in range(1, epochs+1):
        w_b = rdd.context.broadcast(w)
        grad = (rdd.mapPartitions(lambda it: part_grad(it, w_b))
                   .reduce(lambda a,b: a+b))
        w -= lr * grad / n
        w_b.unpersist()
        print(f"[Scratch LR] epoch {ep}/{epochs}")
    return w

def predict_scratch(rdd: RDD, w: np.ndarray):
    w_b = rdd.context.broadcast(w)
    return rdd.map(lambda sv: sigmoid(sum(w_b[i]*v
                                   for i, v in zip(sv.indices, sv.values))))

# ───────────────────────── data ingest (Spark) ────────────────────
def ingest_spark(path: str, spark: SparkSession) -> DataFrame:
    """Load JSON/JSONL and engineer all columns without collecting to driver."""
    sdf = (spark.read.json(path, multiLine=path.endswith(".json"))
           .filter(~F.col("body").isin("[deleted]", "[removed]"))
           .filter(F.col("ups").isNotNull()))

    # ----- edit_delay (handles boolean or numeric 'edited') -----
    if "edited" in sdf.columns and "created_utc" in sdf.columns:
        sdf = sdf.withColumn("edited_num", F.col("edited").cast("double"))
        sdf = sdf.withColumn(
            "edit_delay",
            F.when((F.col("edited_num").isNull()) | (F.col("edited_num") == 0),
                   F.lit(0))
             .otherwise(F.col("edited_num") -
                        F.col("created_utc").cast("double"))
        ).drop("edited_num")
    else:
        sdf = sdf.withColumn("edit_delay", F.lit(0))

    # ----- remaining engineered cols -----
    sdf = (sdf.withColumn("gilded_flag", (F.col("gilded") > 0).cast("int"))
               .withColumn("stickied",    F.coalesce("stickied", F.lit(0)))
               .withColumn("controversiality",
                           F.coalesce("controversiality", F.lit(0)))
               .withColumn("hour",
                           F.when(F.col("created_utc").isNotNull(),
                                  F.hour(F.from_unixtime("created_utc")))
                            .otherwise(F.lit(0)))
               .withColumn("weekday",
                           F.when(F.col("created_utc").isNotNull(),
                                  F.date_format(F.from_unixtime("created_utc"), "u")
                                   .cast("int"))
                            .otherwise(F.lit(0)))
               .withColumn("target", (F.col("ups") >= THRESHOLD).cast("int"))
               .cache())
    return sdf

# ───────────────────────────── main ───────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Up-vote models – Spark optimised")
    ap.add_argument("data",
        help="Path to JSON / JSONL(.gz) – local FS, HDFS or gs://")
    args = ap.parse_args(argv)

    spark = (SparkSession.builder
                .appName("reddit-upvote-pipeline")
                .getOrCreate())

    # ---------- ingest ----------
    sdf = ingest_spark(args.data, spark)
    print(f"Loaded {sdf.count():,} rows")

    # ---------- MLlib pipeline (LogReg & NB) ----------
    tokenizer = RegexTokenizer(inputCol="body", outputCol="tokens",
                               pattern=TOKEN_RE.pattern)
    tf        = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=HASH_DIM)
    idf       = IDF(inputCol="tf", outputCol="tfidf")
    ohe       = OHE(inputCols=["subreddit"], outputCols=["subreddit_ohe"])
    assembler = VectorAssembler(
        inputCols=["tfidf","subreddit_ohe","stickied","controversiality",
                   "gilded_flag","hour","weekday","edit_delay"],
        outputCol="features"
    )

    pipe_base = [tokenizer, tf, idf, ohe, assembler]
    lr_mllib = SLR(labelCol="target", featuresCol="features")
    nb_mllib = SNB(labelCol="target", featuresCol="features")

    trainDF, testDF = sdf.randomSplit([0.85, 0.15], seed=42)

    model_lr = Pipeline(stages=pipe_base + [lr_mllib]).fit(trainDF)
    model_nb = Pipeline(stages=pipe_base + [nb_mllib]).fit(trainDF)

    evaluator = BinaryClassificationEvaluator(labelCol="target",
                                              rawPredictionCol="probability")

    for name, mdl in (("MLlib LR", model_lr), ("MLlib NB", model_nb)):
        pred = mdl.transform(testDF)
        auc = evaluator.evaluate(pred)
        acc = (pred.selectExpr("float(target=prediction) as ok")
                     .agg(F.mean("ok")).first()[0])
        print(f"{name:9}: Acc {acc:.3f}  AUC {auc:.3f}")

    # ---------- scratch LR (no MLlib) ----------
    tfidf_pipe = Pipeline(stages=pipe_base[:3]).fit(trainDF)  # tokenizer + tf + idf
    train_rdd = (tfidf_pipe.transform(trainDF)
                   .select("target", "tfidf")
                   .rdd
                   .persist(StorageLevel.MEMORY_ONLY))
    test_rdd  = (tfidf_pipe.transform(testDF)
                   .select("target", "tfidf")
                   .rdd
                   .persist(StorageLevel.MEMORY_ONLY))
    w = train_scratch_lr(train_rdd)
    scratch_prob = (predict_scratch(test_rdd.map(lambda r: r[1]), w)
                    .collect())
    scratch_pred = (np.array(scratch_prob) >= 0.5).astype("uint8")
    y_true_sp    = np.array(test_rdd.map(lambda r: r[0]).collect(), dtype="uint8")
    acc_sp  = (scratch_pred == y_true_sp).mean()
    auc_sp  = roc_auc_score(y_true_sp, scratch_prob)
    print(f"Scratch LR: Acc {acc_sp:.3f}  AUC {auc_sp:.3f}")

    prob_histogram(np.array(scratch_prob), "Spark LR – prob.",
                   "sparklr_prob_hist.png")

    # ---------- driver-side sample for sklearn + diagnostics ----------
    df_small = (sdf.sample(False, SAMPLE_FRACTION, seed=42)
                   .toPandas())
    df_small = clean_comments(df_small)
    df_small["target"] = (df_small["ups"] >= THRESHOLD).astype("uint8")

    X = df_small.drop(columns=["ups","target"])
    y = df_small["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    preprocess = build_preprocess()

    sk_lr = SKPipeline([
        ("pre", preprocess),
        ("clf", LogisticRegression(max_iter=1000,
                                   class_weight="balanced",
                                   solver="liblinear",
                                   n_jobs=-1)),
    ]).fit(X_tr, y_tr)
    sk_nb = SKPipeline([
        ("pre", preprocess),
        ("clf", MultinomialNB()),
    ]).fit(X_tr, y_tr)

    lr_prob = sk_lr.predict_proba(X_te)[:,1]
    nb_prob = sk_nb.predict_proba(X_te)[:,1]
    lr_pred = (lr_prob >= 0.5).astype("uint8")

    # histograms
    prob_histogram(lr_prob, "LogReg – prob.",  "logreg_prob_hist.png")
    prob_histogram(nb_prob, "NB – prob.",      "naivebayes_prob_hist.png")

    # coefficient bar
    plot_coef_bar(sk_lr)

    # SHAP beeswarm (≤300 rows)
    if _HAS_SHAP:
        plot_shap_beeswarm(sk_lr, X_te.iloc[:300])

    # extended diagnostics
    df_eval = X_te.copy()
    df_eval["prob"]  = lr_prob
    df_eval["pred"]  = lr_pred
    df_eval["actual"]= y_te
    df_eval["length"]= df_eval["body"].str.len()
    if "created_utc" in df_small.columns:
        df_eval["date"] = pd.to_datetime(df_small.loc[X_te.index,"created_utc"],
                                         unit="s").dt.date
    else:
        df_eval["date"] = pd.to_datetime("1970-01-01").date()
    df_eval["absolute_error"] = np.abs(df_eval["actual"] - df_eval["pred"])

    plot_actual_vs_pred(y_te, lr_prob)
    plot_residual_density(y_te - lr_prob)
    plot_error_by_sub(df_eval)
    plot_error_vs_length(df_eval)
    plot_daily_mae(df_eval)

    # ---------- metric table ----------
    summaries = [
        ("MLlib LR",  model_lr.transform(testDF)),
        ("MLlib NB",  model_nb.transform(testDF)),
    ]
    for name, pred_df in summaries:
        y_prob = np.array(pred_df.select("probability").rdd
                          .map(lambda r: float(r[0][1])).collect())
        y_pred = np.array(pred_df.select("prediction").rdd
                          .map(lambda r: int(r[0])).collect())
        y_true = np.array(pred_df.select("target").rdd
                          .map(lambda r: int(r[0])).collect())
        m = dict(
            acc  = (y_pred==y_true).mean(),
            prec = precision_score(y_true, y_pred, zero_division=0),
            rec  = recall_score(y_true, y_pred, zero_division=0),
            f1   = f1_score(y_true, y_pred, zero_division=0),
            auc  = roc_auc_score(y_true, y_prob),
        )
        print(f"{name}: Acc {m['acc']:.3f}  Prec {m['prec']:.3f} "
              f"Rec {m['rec']:.3f}  F1 {m['f1']:.3f}  AUC {m['auc']:.3f}")

    print("\nDetailed classification report (sklearn LogReg):")
    print(classification_report(y_te, lr_pred, zero_division=0))

    spark.stop()

# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
