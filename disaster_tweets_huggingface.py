# %% [markdown]
# # Disaster Tweets — Frozen Transformer Features
#
# **Project Question:** Does changing the text representation improve disaster classification when the classifier and evaluation rows stay fixed?
# **Goal:** Compare MiniLM embeddings with the existing TF-IDF results, using Logistic Regression and LinearSVC.
# This assistant-authored extension complements `disaster_tweets_classification.py`; that file remains the source of the original baseline.
#
# **Reading route:** Same observations → frozen encoder → training-only CV → retrospective holdout → insights → ethics → visual comparison.
# **Execution evidence — 2026-10-29:** All eleven code cells ran in the project `.venv`, including 20 CV fits and two final classifier fits.
# MiniLM improved both classifiers' mean CV F1 relative to their historical TF-IDF references; detailed results and limits appear below.
# Run the percent cells from top to bottom in the existing project `.venv`.
#
# > **⚠️ Watch Out — An encoder does not supply disaster labels**
# >
# > MiniLM produces numerical features. Our classifiers must still learn the relationship between those features and the recorded training labels.
# > This is supervised classification with a frozen encoder, not zero-shot classification or Transformer fine-tuning.

# %% [markdown]
# ## 1. Preserve the observations and evaluation boundaries
#
# **❓ Main Question:** Are we comparing representations on the same observations?
# **🎯 Goal & Approach:** Reuse the main analysis's normalization, retention policy, split and ten training folds.
# **🔎 Guiding Question:** What does equal membership control, and what uncertainty remains?
#
# The supplied `DATA/train.csv` contains text and recorded labels: `0` non-disaster, `1` disaster.
# Lowercase and normalize whitespace, exclude conflicting normalized-text groups, then retain the first occurrence of each consistent group.
# These are existing project decisions; neither the dataset nor the baseline preparation is changed here.

# %%
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

# Resolve the existing dataset beside this script, or in the Interactive Window's project folder.
project_root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_path = project_root / "DATA" / "train.csv"
if not data_path.is_file():
    raise FileNotFoundError(f"Cannot find {data_path}. Open the Interactive Window in the repository root.")

print("Python:", sys.version.split()[0], "Interpreter:", sys.executable)
print("pandas:", pd.__version__, "scikit-learn:", sklearn.__version__)
tweets = pd.read_csv(data_path)

# Keep original text and source-row indices; prepared_text is the identical baseline input string.
prepared = tweets.copy()
prepared["prepared_text"] = prepared["text"].str.lower()
prepared["prepared_text"] = prepared["prepared_text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Each grouped value counts distinct labels for one normalized string, not observations.
labels_per_text = prepared.groupby("prepared_text")["target"].nunique()
conflicting_keys = labels_per_text[labels_per_text > 1].index
consistent_rows = prepared.loc[~prepared["prepared_text"].isin(conflicting_keys)]
model_tweets = consistent_rows.drop_duplicates(subset="prepared_text", keep="first").copy()

# Split whole rows so text, labels and original source indices keep the same membership and order.
train_rows, holdout_rows = train_test_split(
    model_tweets, test_size=0.20, random_state=42, stratify=model_tweets["target"]
)
X_train, y_train = train_rows["prepared_text"], train_rows["target"]
X_holdout, y_holdout = holdout_rows["prepared_text"], holdout_rows["target"]

# Each fold contains positional row numbers within training, not CSV source indices.
# Materializing this list makes both classifiers use exactly the same ten partitions.
cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_folds = list(cv_splitter.split(X_train, y_train))
print("Original / retained / training / holdout rows:", len(tweets), len(model_tweets), len(train_rows), len(holdout_rows))
print("Training source indices, first five:", train_rows.index[:5].tolist())
print("Holdout source indices, first five:", holdout_rows.index[:5].tolist())
print("Shared prepared strings:", len(set(X_train) & set(X_holdout)))
print("Validation rows per training fold:", [len(validation) for _, validation in cv_folds])

# %% [markdown]
# ### ✅ Evidence & Interpretation
# The preparation check on 2026-10-29 reproduced 7,613 source rows, 7,483 retained rows, 5,986 training rows and 1,497 holdout rows.
# Temporary verification compared complete tables, source indices, labels, row order and every fold against the main analysis.
# Equal membership controls which observations each comparison uses; it does not establish independent events or correct annotations.
#
# > **⚠️ Watch Out — A familiar holdout is not a fresh final test**
# >
# > This project has already inspected its holdout. Any additional holdout results are retrospective educational evidence.
# > Both classifiers are specified before evaluation; holdout results must not choose the encoder, classifier settings or preparation.
#
# **🧠 Learning → Better Question:** Would the comparison hold across genuinely separate disaster events?
# **Current answer: not established.** Matching rows cannot answer that question without event relationships and fresh evaluation evidence.

# %% [markdown]
# ## 2. Turn each tweet into a frozen representation
#
# **❓ Main Question:** What does the pretrained encoder supply, and what will our classifiers still learn?
# **🎯 Goal & Approach:** Produce one 384-number vector per tweet using `sentence-transformers/all-MiniLM-L6-v2`.
# **🔎 Guiding Questions:** How do token vectors become one tweet vector, and why may this representation be computed before CV?
#
# A Transformer makes token representations depend on surrounding tokens. Pooling then combines those token vectors into one fixed-length representation.
# The encoder has no disaster classification head; Logistic Regression and LinearSVC will learn that task from our labels.
#
# | Model decision | Fixed choice for this experiment |
# | --- | --- |
# | Checkpoint revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
# | Purpose and language | English sentence / short-paragraph embeddings |
# | Architecture | Six Transformer layers, approximately 22.7 million parameters |
# | Representation | Attention-mask mean pooling, followed by per-vector L2 normalization |
# | Input limit | 256 word pieces including special tokens; count any truncated inputs |
# | License | Apache-2.0, as declared by the model card |
# | Runtime | CPU; existing PyTorch 2.15.0 and Transformers 5.21.0 |
# | Setup | Nine cached model/tokenizer/configuration files, 91.6 MB in total; no new Python package |
#
# Sources: [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
# [architecture](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config.json),
# [pooling configuration](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/1_Pooling/config.json),
# and [sequence limit](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/sentence_bert_config.json).
#
# **📋 Teaching provenance:** The MasterSchool guided notebook motivates classifier comparison and fold-level F1.
# DataCamp's [Embedding restaurant reviews](https://campus.datacamp.com/courses/introduction-to-embeddings-with-the-openai-api/embeddings-for-ai-applications?ex=9)
# and [Classifying review sentiment](https://campus.datacamp.com/courses/introduction-to-embeddings-with-the-openai-api/embeddings-for-ai-applications?ex=10)
# illustrate embeddings followed by a classification decision; their inspected exercises use label similarity and an API.
# Here, the encoder runs locally and supervised classifiers learn from tweet labels; no course code, API calls or course scores are transferred.

# %% [markdown]
# ### 🗺️ Mental Model — Keep representation and decision separate
#
# <div style="height: 210px; overflow: auto;">
#
# ```mermaid
# flowchart LR
#     T["Prepared<br/>tweet"] --> E["Frozen<br/>MiniLM"]
#     E --> V["384-number<br/>vector"]
#     V --> L["Logistic<br/>Regression"]
#     V --> S["LinearSVC"]
#     L --> D["Disaster /<br/>non-disaster"]
#     S --> D
# ```
#
# </div>

# %% [markdown]
# > **⚠️ Watch Out — Pretraining is additional information**
# >
# > MiniLM learned from large external corpora before this project. Its card lists mixed sentence-pair sources, not a disaster-label training task.
# > Exact overlap with these tweets cannot be ruled out from the card alone; equal evaluation rows do not imply equal training information.
#
# **WHY this can precede CV:** A fixed encoder transforms each text independently without fitting our vocabulary, labels or corpus-wide statistics.
# Mean pooling and L2 normalization also operate within each example. Only the downstream classifiers learn from our training folds.
# This differs from TF-IDF, whose learned vocabulary and IDF must remain inside each fold's Pipeline.
#
# **HOW pooling works:** Ignore padding positions, average the remaining token vectors, then divide each resulting vector by its length.
# This produces comparable vector lengths; it neither guarantees preserved meaning nor makes individual dimensions interpretable as words.
# Word pieces are tokenizer units that can split a word; the 256-position limit also includes special start/end tokens.

# %%
import torch
import torch.nn.functional as functional
import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoModel, AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model_revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
model_files = [
    "1_Pooling/config.json", "config.json", "model.safetensors", "modules.json",
    "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.txt",
]

# Locate an already downloaded, pinned snapshot. local_files_only forbids a network download.
# Match the approved file subset; the model repository also contains unused backend formats.
try:
    model_path = snapshot_download(
        repo_id=model_id, revision=model_revision, allow_patterns=model_files, local_files_only=True
    )
except LocalEntryNotFoundError as error:
    raise FileNotFoundError(
        f"Approved model setup is required for {model_id} at {model_revision}. "
        "No model files were downloaded. Stop here until that separate setup is complete."
    ) from error

# Read only cached tokenizer/configuration and safetensors weights; no repository Python code is used.
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
encoder = AutoModel.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
).to("cpu")
encoder.eval()  # Disable training-time dropout, so inference uses the fixed model consistently.
encoder.requires_grad_(False)  # No encoder parameter will receive a training gradient.
print("PyTorch:", torch.__version__, "Transformers:", transformers.__version__)
print("Encoder:", model_id, "Revision:", model_revision)

# %%
# These dictionaries keep matrices and timings separate for the two unchanged partitions.
# Every matrix row follows its corresponding X Series; each column is one embedding dimension.
embedding_matrices = {}
encoding_seconds = {}
maximum_tokens = 256
batch_size = 32

for partition_name, texts in [("training", X_train), ("holdout", X_holdout)]:
    started = perf_counter()
    text_values = texts.tolist()

    # Measure lengths before truncation; these counts include the tokenizer's special tokens.
    full_token_ids = tokenizer(text_values, truncation=False)["input_ids"]
    token_lengths = np.array([len(token_ids) for token_ids in full_token_ids])
    print(partition_name, "inputs exceeding the limit:", int((token_lengths > maximum_tokens).sum()))

    # Batches limit working memory; appending in input order preserves label alignment.
    batch_embeddings = []
    for start in range(0, len(text_values), batch_size):
        batch = text_values[start:start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=maximum_tokens, return_tensors="pt"
        )
        with torch.inference_mode():
            # Shape: tweets in this batch × padded token positions × 384 numerical features.
            token_vectors = encoder(**encoded).last_hidden_state
            # The extra dimension lets each 0/1 padding flag apply to all 384 token features.
            token_mask = encoded["attention_mask"].unsqueeze(-1).to(token_vectors.dtype)
            vector_sums = (token_vectors * token_mask).sum(dim=1)
            token_counts = token_mask.sum(dim=1).clamp(min=1)
            tweet_vectors = vector_sums / token_counts
            unit_vectors = functional.normalize(tweet_vectors, p=2, dim=1)
        batch_embeddings.append(unit_vectors.cpu().numpy())

    embedding_matrices[partition_name] = np.concatenate(batch_embeddings, axis=0)
    encoding_seconds[partition_name] = perf_counter() - started
    print(partition_name, "embedding shape:", embedding_matrices[partition_name].shape)
    print(partition_name, "encoding seconds:", round(encoding_seconds[partition_name], 2))

train_embeddings = embedding_matrices["training"]
holdout_embeddings = embedding_matrices["holdout"]
print("Training feature range:", train_embeddings.min(), train_embeddings.max())
print("Training vector lengths, min/max:", np.linalg.norm(train_embeddings, axis=1).min(), np.linalg.norm(train_embeddings, axis=1).max())

# %% [markdown]
# ### ✅ Evidence & Interpretation
# **Measured answer:** The training matrix has shape `(5986, 384)`; the holdout matrix has shape `(1497, 384)`.
# No input exceeded 256 token positions in either partition, so this run truncated no tweet.
# Training features ranged from approximately `-0.2563` to `0.2679`; vector lengths were approximately one, within floating-point precision.
#
# Pooling produced one vector per tweet; only the later classifiers learned from our labels.
# The cells print encoding times for the current machine; these timings describe local execution, not a benchmark against TF-IDF.
#
# > **⚠️ Watch Out — MultinomialNB is not a drop-in classifier for these features**
# >
# > Dense embedding dimensions can be negative and do not represent term counts. MultinomialNB requires nonnegative inputs with different modeling assumptions.
# > Its existing TF-IDF results remain references; no shifted, clipped or relabeled Naive Bayes experiment is introduced.
#
# **🧠 Learning → Better Question:** Do the transferred features help either classifier on the same validation observations?
# **Measured answer:** Both classifiers have higher mean CV F1 than their historical TF-IDF references, as shown in the next unit.
# Successful encoding alone would not establish that improvement; the labeled validation scores supply the evidence.

# %% [markdown]
# ## 3. Compare the same classifiers in the same ten folds
#
# **❓ Main Question:** Does MiniLM improve training-only CV F1 for Logistic Regression or LinearSVC?
# **🎯 Goal & Approach:** Fit fresh classifiers on nine folds and score the remaining fold, repeating all ten partitions.
# **🔎 Guiding Questions:** Are gains consistent across folds, and what does the fold standard deviation describe?
#
# Both classifiers retain the main file's settings: `LogisticRegression()` and `LinearSVC(random_state=42)`.
# The encoder remains fixed; this adds 20 classifier fits, without Transformer fine-tuning or parameter search.
# Unigrams and bigrams are TF-IDF settings, not alternative settings for these sentence embeddings.
#
# > **⚠️ Watch Out — Equal settings do not imply equally optimized models**
# >
# > The same `C=1` is a controlled starting point, not proof of equally suitable regularization for different feature representations.
# > Results compare these fixed workflows; they cannot establish the best achievable version of either approach.

# %%
# Options are unfitted templates; clone creates an independent classifier for each fold.
classifier_options = [
    ("Logistic Regression", LogisticRegression()),
    ("LinearSVC", LinearSVC(random_state=42)),
]
fold_records = []
embedding_summary_records = []

for classifier_name, classifier in classifier_options:
    classifier_f1 = []
    for fold_number, (fit_positions, validation_positions) in enumerate(cv_folds, start=1):
        fold_classifier = clone(classifier)
        fold_classifier.fit(train_embeddings[fit_positions], y_train.iloc[fit_positions])
        predictions = fold_classifier.predict(train_embeddings[validation_positions])

        # iloc selects labels by fold position; NumPy matrices use the same positional convention.
        validation_labels = y_train.iloc[validation_positions]
        score = f1_score(validation_labels, predictions, pos_label=1, zero_division=0)
        classifier_f1.append(score)
        fold_records.append({"classifier": classifier_name, "fold": fold_number, "f1": score})
        print(f"{classifier_name}, fold {fold_number}: F1={score:.6f}")

    # ddof=0 matches the original CV summary; it describes these ten scores, not a confidence interval.
    scores = np.array(classifier_f1)
    embedding_summary_records.append({
        "classifier": classifier_name, "representation": "MiniLM embeddings",
        "mean_f1": scores.mean(), "std_f1": scores.std(ddof=0), "evidence": "Current run",
    })

# Each table row identifies a measured fold or one classifier's ten-fold summary.
fold_results = pd.DataFrame(fold_records)
embedding_summary = pd.DataFrame(embedding_summary_records)
print("\nActual training-only fold F1:")
print(fold_results.pivot(index="fold", columns="classifier", values="f1").round(6).to_string())
print("\nEmbedding CV summaries:")
print(embedding_summary.round(6).to_string(index=False))

# %%
# Historical values are the six published summaries documented in the main analysis, rounded to four decimals.
# This file does not rerun or replace those 60 TF-IDF fits; the evidence column makes that distinction visible.
reference_rows = [
    ("Logistic Regression", "TF-IDF (1, 1)", 0.7506, 0.0197),
    ("Logistic Regression", "TF-IDF (1, 2)", 0.7293, 0.0181),
    ("LinearSVC", "TF-IDF (1, 1)", 0.7554, 0.0246),
    ("LinearSVC", "TF-IDF (1, 2)", 0.7529, 0.0170),
    ("MultinomialNB", "TF-IDF (1, 1)", 0.7219, 0.0147),
    ("MultinomialNB", "TF-IDF (1, 2)", 0.7113, 0.0156),
]
tfidf_reference = pd.DataFrame(reference_rows, columns=["classifier", "representation", "mean_f1", "std_f1"])
tfidf_reference["evidence"] = "Earlier run; rounded reference"
cv_comparison = pd.concat([tfidf_reference, embedding_summary], ignore_index=True)
cv_comparison["configuration"] = cv_comparison["classifier"] + " + " + cv_comparison["representation"]
print("\nSame training rows and folds; separate historical/current evidence:")
print(cv_comparison.drop(columns="configuration").round(4).to_string(index=False))

# Compare each new mean with that same classifier's two historical TF-IDF means, never with holdout F1.
for _, current in embedding_summary.iterrows():
    references = tfidf_reference.loc[tfidf_reference["classifier"] == current["classifier"]]
    for _, reference in references.iterrows():
        difference = current["mean_f1"] - reference["mean_f1"]
        print(f"{current['classifier']}: MiniLM minus {reference['representation']} mean F1 = {difference:+.4f} (approximate)")

# %%
# Lines connect fold identifiers, not time. Each point uses the same validation membership across classifiers.
fold_figure = px.line(
    fold_results, x="fold", y="f1", color="classifier", markers=True,
    title="MiniLM: class-1 F1 across ten training folds",
    labels={"fold": "Training validation fold", "f1": "Class-1 F1", "classifier": "Classifier"},
)
fold_figure.update_layout(height=360)
fold_figure.update_xaxes(dtick=1)
fold_figure.update_yaxes(range=[0, 1])
fold_figure.show()

# Error bars display fold spread, not uncertainty bounds or an independent statistical significance test.
comparison_figure = px.scatter(
    cv_comparison, x="mean_f1", y="configuration", error_x="std_f1", color="representation",
    symbol="evidence", hover_data=["evidence", "std_f1"],
    title="Training CV: mean F1 and fold spread",
    labels={"mean_f1": "Mean class-1 F1", "configuration": "Configuration", "representation": "Representation", "evidence": "Evidence"},
)
comparison_figure.update_layout(height=520, showlegend=False)
comparison_figure.update_xaxes(range=[0, 1])
# Wrap long configuration labels so they remain readable beside a split editor.
comparison_figure.update_yaxes(
    title=None, tickvals=cv_comparison["configuration"],
    ticktext=cv_comparison["configuration"].str.replace(" + ", "<br>+ ", regex=False),
)
comparison_figure.show()

# %% [markdown]
# ### ✅ Evidence & Interpretation
# **Measured answer:** Both MiniLM workflows increased mean training-only F1 compared with the same classifier's historical TF-IDF configurations.
# These are fixed-workflow comparisons; the original six TF-IDF runs were not repeated or modified by this experiment.
#
# | Classifier with MiniLM | Mean F1 | Fold SD |
# | --- | --- | --- |
# | Logistic Regression | 0.773690 | 0.017233 |
# | LinearSVC | 0.768419 | 0.018668 |
#
# Compared with TF-IDF `(1, 1)` / `(1, 2)`, mean F1 increased by `0.0231` / `0.0444` for Logistic Regression.
# Corresponding LinearSVC increases were `0.0130` / `0.0155`; these differences use historical means rounded to four decimals.
#
# <details>
# <summary>Recorded fold F1 — 2026-10-29</summary>
#
# | Fold | Logistic Regression | LinearSVC |
# | --- | --- | --- |
# | 1 | 0.803960 | 0.797595 |
# | 2 | 0.748466 | 0.726891 |
# | 3 | 0.776660 | 0.783133 |
# | 4 | 0.770492 | 0.770791 |
# | 5 | 0.747475 | 0.748971 |
# | 6 | 0.757515 | 0.760563 |
# | 7 | 0.782609 | 0.777778 |
# | 8 | 0.790224 | 0.778004 |
# | 9 | 0.783133 | 0.775758 |
# | 10 | 0.776371 | 0.764706 |
#
# </details>
#
# **Consistency answer:** Logistic Regression led LinearSVC in six of these ten folds; neither classifier led in every fold.
# Consistent paired gains over TF-IDF are not established here because that comparison uses rounded historical summaries, without its original fold arrays.
#
# **Spread answer:** The standard deviation describes variation among these ten scores; overlapping training sets prevent treating folds as independent experiments.
# Neither a higher mean nor these error bars establish statistical significance.
#
# **🧠 Learning → Better Question:** Would a measured average gain also reduce consequential errors on other events?
# **Current answer: not established.** Ten overlapping training folds describe this experiment, not independent populations or operational consequences.

# %% [markdown]
# ## 4. Report the two predeclared models on the existing holdout
#
# **❓ Main Question:** What do the two frozen-feature workflows predict on the unchanged 1,497 holdout rows?
# **🎯 Goal & Approach:** Refit each classifier on all training embeddings, then evaluate the holdout once with fixed settings.
# Both classifiers are evaluated regardless of their CV ranking; this adds two final classifier fits without choosing a CV winner.
#
# **🔎 Guiding Question:** Which comparison has a matching historical holdout baseline?
# Only TF-IDF + Logistic Regression has an existing holdout result. LinearSVC's historical figures are CV results and belong in the preceding comparison.

# %%
holdout_records = []
holdout_predictions = {}
print("Holdout support by recorded label:", y_holdout.value_counts().sort_index().to_dict())

for classifier_name, classifier in classifier_options:
    fitted_classifier = clone(classifier)
    fitted_classifier.fit(train_embeddings, y_train)
    predictions = fitted_classifier.predict(holdout_embeddings)
    holdout_predictions[classifier_name] = predictions

    # Every metric uses these same labels and predictions; class 1 is always the positive class.
    holdout_records.append({
        "configuration": classifier_name + " + MiniLM",
        "f1": f1_score(y_holdout, predictions, pos_label=1, zero_division=0),
        "precision": precision_score(y_holdout, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_holdout, predictions, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_holdout, predictions),
        "evidence": "Current retrospective run",
    })
    counts = confusion_matrix(y_holdout, predictions, labels=[0, 1])
    print("\n", classifier_name, "holdout matrix; rows=recorded, columns=predicted, labels=[0, 1]:")
    print(counts)
    confusion_figure = px.imshow(
        counts, x=["0: non-disaster", "1: disaster"], y=["0: non-disaster", "1: disaster"],
        text_auto=True, color_continuous_scale="Blues", zmin=0, zmax=int(y_holdout.value_counts().max()),
        labels={"x": "Predicted label", "y": "Recorded label", "color": "Tweets"},
        title=classifier_name + " + MiniLM: holdout",
    )
    confusion_figure.update_layout(height=360)
    confusion_figure.show()

# Recover historical baseline metrics from its documented confusion counts, not a new baseline fit.
baseline_tn, baseline_fp, baseline_fn, baseline_tp = 752, 108, 208, 429
baseline_record = {
    "configuration": "Logistic Regression + TF-IDF (1, 1)",
    "f1": 2 * baseline_tp / (2 * baseline_tp + baseline_fp + baseline_fn),
    "precision": baseline_tp / (baseline_tp + baseline_fp),
    "recall": baseline_tp / (baseline_tp + baseline_fn),
    "accuracy": (baseline_tn + baseline_tp) / (baseline_tn + baseline_fp + baseline_fn + baseline_tp),
    "evidence": "Earlier baseline; documented confusion counts",
}
holdout_summary = pd.DataFrame([baseline_record] + holdout_records)
print("\nRetrospective holdout metrics; do not compare this table with CV means:")
print(holdout_summary.round(4).to_string(index=False))

# %%
# Preserve original wording and source-row indices when inspecting prediction errors.
# These first three errors per classifier are an illustration, not an estimate of error-type prevalence.
holdout_review = holdout_rows[["text", "target"]].copy()
for classifier_name, predictions in holdout_predictions.items():
    holdout_review[classifier_name] = predictions
    error_mask = holdout_review[classifier_name] != holdout_review["target"]
    print("\nFirst three holdout errors:", classifier_name)
    print(holdout_review.loc[error_mask, ["text", "target", classifier_name]].head(3).to_string())

# %% [markdown]
# ### ✅ Evidence & Interpretation
# **Measured answer:** Both workflows were evaluated on the same 1,497 observations: 860 recorded non-disasters and 637 disasters.
# The following metrics use disaster as the positive class; the confusion plots share a zero-based count scale.
#
# **Table key:** LR = Logistic Regression; SVC = LinearSVC. The TF-IDF column is the earlier Logistic Regression baseline.
#
# | Metric | Earlier TF-IDF + LR | MiniLM + LR | MiniLM + SVC |
# | --- | --- | --- | --- |
# | F1 | 0.7308 | 0.7650 | 0.7718 |
# | Precision | 0.7989 | 0.7906 | 0.7963 |
# | Recall | 0.6735 | 0.7410 | 0.7488 |
# | Accuracy | 0.7889 | 0.8063 | 0.8116 |
# | False negatives | 208 | 165 | 160 |
# | False positives | 108 | 125 | 122 |
#
# **Matched comparison:** With Logistic Regression, MiniLM increased holdout F1 by approximately `0.0342`, while precision decreased.
# It missed 43 fewer recorded disasters but produced 17 more false positives; a higher F1 does not improve every error measure.
# LinearSVC lacks a historical holdout result for the same classifier, so its representation comparison rests on training CV.
#
# **Error inspection:** Both classifiers predicted `0` for source rows `7165` and `7273`, whose recorded labels are `1`.
# One asks a hypothetical war question; the other's wording appears figurative, making annotation interpretation a reasonable review question.
# These observations do not establish incorrect labels or explain the models' internal causes; the labels and settings remain unchanged.
#
# > **⚠️ Watch Out — More context does not guarantee correct negation or factual truth**
# >
# > An embedding can encode context and still support a wrong prediction. A high F1 does not verify that a described disaster occurred.
# > Negation handling was not specifically tested here; dedicated counterexamples would be needed to assess it.

# %% [markdown]
# ## 5. Insights, limits and a justified next question
#
# **❓ Main Question:** What does this experiment establish?
# **🎯 Goal:** Connect measured differences to a proportionate conclusion and identify the evidence still needed.
# **Measured answer:** MiniLM features improved both classifiers' mean CV F1 on the unchanged training partitions.
# The matched Logistic Regression holdout comparison also improved F1, with fewer false negatives and more false positives.
# These findings support this specific feature-transfer experiment, not universal Transformer superiority or a new final-test claim.
#
# **Learning:** External pretraining changes the available information, while frozen features keep our fold-specific learning confined to the classifiers.
# Fold scores are Data; matching labels, rows and settings provide Information. The measured differences support Knowledge about these particular workflows.
# Wisdom requires weighing its consistency, error types, computation and evidence limits before recommending a more complex approach.
#
# **Better Question:** Would any observed improvement survive evaluation on separately collected, independently annotated disaster reports?
# **Current answer: not established.** This dataset and retrospective holdout cannot supply that new evidence.

# %% [markdown]
# ## 6. Ethical interpretation and responsible use
#
# **❓ Main Question:** What may we responsibly claim from the observed improvements?
# **🎯 Goal:** Connect measured error changes to their possible consequences without inventing real-world benefits.
#
# **Evidence and answer:** MiniLM improves the measured average CV F1 under this project's preparation and evaluation choices.
# With Logistic Regression, it misses 43 fewer recorded disaster labels but creates 17 additional false positives on the familiar holdout.
# These counts establish a prediction trade-off; they do not measure people helped, response times or consequences of false alarms.
#
# **Labels and coverage:** Ambiguous wording in the inspected errors makes annotation review relevant; it does not prove the recorded labels wrong.
# Scores describe the retained English tweets, after the agreed exclusions; fairness across populations and reliability across new events remain unmeasured.
# External pretraining and possible source overlap also limit what we can attribute to the representation alone.
#
# > **⚠️ Watch Out — A better F1 is not an operational recommendation**
# >
# > The relative costs of missed reports and false alarms depend on a defined use context, which this experiment does not establish.
# > A retrospective holdout cannot establish readiness for emergency decisions; fresh event-level evaluation and annotation review would be needed.
#
# **🧠 Learning → Better Question:** How should a proposed use case weigh missed reports against false alarms?
# **Current answer: not established.** That decision needs input from affected users and evidence about consequences, beyond the metrics measured here.
# **⚖️ Ethical Takeaway:** Report the error trade-off, annotation uncertainty and evaluation limits together; neither representation establishes emergency-response readiness.

# %% [markdown]
# ## 7. Visual comparison — TF-IDF and Hugging Face features
#
# **❓ Main Question:** How large are the measured differences, and which errors change when we replace TF-IDF with MiniLM?
# **🎯 Goal & Approach:** Finish with three complementary Plotly views, derived from the result tables and predictions already computed above.
# **🔎 Guiding Question:** Which comparisons keep the classifier fixed, and where is a matching baseline unavailable?
#
# **Comparison boundary:** CV includes both classifiers against their two historical TF-IDF configurations, using the same training rows and folds.
# Holdout views compare Logistic Regression only, because LinearSVC has no historical holdout baseline for isolating the representation change.
# MiniLM + LinearSVC's holdout result remains in Section 4; these figures do not select a model or refit anything.
#
# ### A. Change in training-only mean F1
# **WHY:** A difference plot makes the size of each change easier to read than eight nearby averages.
# **HOW:** Subtract each classifier's historical TF-IDF mean from its current MiniLM mean; positive bars indicate higher mean F1.
# These are absolute F1 differences, not relative percentage improvements; historical rounding makes the differences approximate.
#
# Plotting reference: [Plotly grouped bars and value labels](https://plotly.com/python/bar-charts/).

# %%
# One row compares the same classifier with one historical TF-IDF configuration.
# Reuse measured summaries; no fitting or holdout metrics enter this table.
cv_change_records = []
for _, current in embedding_summary.iterrows():
    references = tfidf_reference.loc[tfidf_reference["classifier"] == current["classifier"]]
    for _, reference in references.iterrows():
        cv_change_records.append({
            "classifier": current["classifier"],
            "tfidf_reference": reference["representation"],
            "earlier_mean_f1": reference["mean_f1"],
            "current_mean_f1": current["mean_f1"],
            "f1_change": current["mean_f1"] - reference["mean_f1"],
        })
cv_changes = pd.DataFrame(cv_change_records)
# Format only the displayed labels; keep the measured values unchanged for plotting and comparison.
cv_changes["change_label"] = [f"{value:+.4f}" for value in cv_changes["f1_change"]]

cv_change_figure = px.bar(
    cv_changes, x="classifier", y="f1_change", color="tfidf_reference", barmode="group",
    text="change_label", title="Training CV: MiniLM minus TF-IDF",
    labels={
        "classifier": "Classifier", "f1_change": "Change in mean F1", "tfidf_reference": "Earlier reference",
        "earlier_mean_f1": "Earlier TF-IDF mean F1", "current_mean_f1": "Current MiniLM mean F1",
    },
    hover_data={"earlier_mean_f1": ":.4f", "current_mean_f1": ":.4f", "f1_change": ":+.4f", "change_label": False},
)
cv_change_figure.update_traces(texttemplate="%{text}", textposition="outside", textangle=0, cliponaxis=False)
cv_change_figure.update_layout(height=400, legend=dict(orientation="h", y=-0.25), margin=dict(b=100))
cv_change_figure.update_xaxes(title=None)
cv_change_figure.update_yaxes(rangemode="tozero", tickformat=".3f")
cv_change_figure.show()

# %% [markdown]
# **Measured answer:** All four mean differences are positive: approximately `+0.0231` / `+0.0444` for Logistic Regression and `+0.0130` / `+0.0155` for LinearSVC.
# The larger bigram comparison partly reflects its lower TF-IDF reference; it does not identify the best achievable model.
# Historical means alone cannot establish paired improvements in every fold or statistical significance.
#
# ### B. Holdout metrics with Logistic Regression fixed
# **WHY:** F1 summarizes a trade-off, so show precision, recall and accuracy beside it rather than treating one score as the whole result.
# **HOW:** Display the earlier TF-IDF and current MiniLM scores side by side on the same zero-to-one scale.
# This is retrospective evidence on the same 1,497 observations; class-1 F1, precision and recall concern the recorded disaster label.

# %%
# Select only the two workflows that share Logistic Regression; keep their evidence labels for hover details.
representation_names = {
    "Logistic Regression + TF-IDF (1, 1)": "TF-IDF (earlier)",
    "Logistic Regression + MiniLM": "MiniLM (current)",
}
matched_holdout = holdout_summary.loc[holdout_summary["configuration"].isin(representation_names)].copy()
matched_holdout["representation"] = matched_holdout["configuration"].map(representation_names)

# melt turns four metric columns into rows: each row now identifies one representation and one metric.
# Keeping representation and evidence as identifiers preserves which workflow supplied each score.
holdout_metric_comparison = matched_holdout.melt(
    id_vars=["representation", "evidence"], value_vars=["f1", "precision", "recall", "accuracy"],
    var_name="metric", value_name="score",
)
holdout_metric_comparison["metric"] = holdout_metric_comparison["metric"].str.capitalize()
holdout_metric_comparison["score_label"] = [f"{value:.4f}" for value in holdout_metric_comparison["score"]]
representation_colors = {"TF-IDF (earlier)": "#4C78A8", "MiniLM (current)": "#B279A2"}

holdout_metric_figure = px.bar(
    holdout_metric_comparison, x="score", y="metric", color="representation", orientation="h", barmode="group",
    color_discrete_map=representation_colors, text="score_label",
    hover_data={"evidence": True, "score": ":.4f", "score_label": False},
    title="Holdout metrics: Logistic Regression",
    labels={"score": "Score", "metric": "Metric", "representation": "Representation", "evidence": "Evidence"},
    category_orders={"metric": ["F1", "Precision", "Recall", "Accuracy"]},
)
holdout_metric_figure.update_traces(texttemplate="%{text}", textposition="outside", textangle=0, cliponaxis=False)
holdout_metric_figure.update_layout(height=440, legend=dict(orientation="h", y=-0.2), legend_title_text=None, margin=dict(b=100))
holdout_metric_figure.update_xaxes(range=[0, 1])
holdout_metric_figure.update_yaxes(title=None)
holdout_metric_figure.show()

# %% [markdown]
# **Measured answer:** F1 rises from `0.7308` to `0.7650`, recall from `0.6735` to `0.7410`, and accuracy from `0.7889` to `0.8063`.
# Precision falls from `0.7989` to `0.7906`; the representation change does not improve every metric.
# The bars reuse existing results; reshaping with [pandas `melt`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html) does not recalculate or average scores.
#
# ### C. Translate the trade-off into error counts
# **WHY:** Error counts make the measured trade-off concrete without assigning an unsupported cost to either error type.
# **HOW:** Subtract earlier Logistic Regression error counts from its MiniLM counts; negative means fewer errors of that type.
# False negatives miss recorded disasters; false positives predict disaster for recorded non-disasters.

# %%
# Reuse existing holdout predictions; rows are recorded labels and columns are predictions, in order [0, 1].
minilm_lr_counts = confusion_matrix(y_holdout, holdout_predictions["Logistic Regression"], labels=[0, 1])

# One row per error type keeps the earlier and current counts visible in the hover details.
error_changes = pd.DataFrame({
    "error_type": ["Missed disasters", "False alarms"],
    "earlier_tfidf": [baseline_fn, baseline_fp],
    "current_minilm": [int(minilm_lr_counts[1, 0]), int(minilm_lr_counts[0, 1])],
})
error_changes["count_change"] = error_changes["current_minilm"] - error_changes["earlier_tfidf"]
error_changes["change_label"] = [f"{value:+d}" for value in error_changes["count_change"]]

error_change_figure = px.bar(
    error_changes, x="count_change", y="error_type", orientation="h", color="error_type", text="change_label",
    hover_data={"earlier_tfidf": True, "current_minilm": True, "change_label": False},
    title="Holdout: change in error counts",
    labels={
        "count_change": "MiniLM minus TF-IDF: tweet count", "error_type": "Error type",
        "earlier_tfidf": "Earlier TF-IDF count", "current_minilm": "Current MiniLM count",
    },
    color_discrete_map={"Missed disasters": "#4C78A8", "False alarms": "#E45756"},
    category_orders={"error_type": ["Missed disasters", "False alarms"]},
)
# Symmetric limits keep positive and negative changes comparable, with room for value labels.
error_axis_limit = max(1, int(error_changes["count_change"].abs().max())) * 1.3
error_change_figure.update_xaxes(
    range=[-error_axis_limit, error_axis_limit], zeroline=True, zerolinewidth=2, zerolinecolor="#64748b",
)
error_change_figure.update_yaxes(title=None)
error_change_figure.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
error_change_figure.update_layout(height=320, showlegend=False)
error_change_figure.show()

# %% [markdown]
# **Measured answer:** Missed disaster labels decrease from `208` to `165` (`−43`); false positives increase from `108` to `125` (`+17`).
# These are net count differences, not proof that exactly 43 particular tweets were corrected; some predictions may change in opposite directions.
# The two error types have different label-group denominators, so these bars show counts rather than comparable error rates.
#
# **🧠 Learning → Better Question:** Does this error trade-off justify using the more complex representation in a defined real-world setting?
# **Current answer: not established.** The charts support a measured educational comparison; deciding usefulness also requires fresh evaluation and evidence about error costs.
# Read the visual gains together with the ethical limits in Section 6, rather than declaring an overall winner from one score.
