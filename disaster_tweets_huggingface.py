# %% [markdown]
# # Disaster Tweets — Frozen Transformer Features
#
# **What is the Question ❓** Does changing the text representation improve disaster classification when the classifier and evaluation rows stay fixed?
#
# **🎯 Goal & Approach — WHY:** Compare MiniLM embeddings with the existing TF-IDF results, using Logistic Regression and LinearSVC.
# This assistant-authored extension complements `disaster_tweets_classification.py`; that file remains the source of the original baseline.
#
# **Reading route:** Same observations → frozen encoder → training-only CV → retrospective holdout → insights → ethics → visual comparison.
#
# **Initial execution — 2026-10-29:** The original comparison ran locally, including 20 CV fits and two final classifier fits.
# MiniLM improved both classifiers' mean CV F1 relative to their historical TF-IDF references; detailed results and limits appear below.
# Section 8 adds a separate training-only regularization comparison; the original workflows and holdout results remain unchanged.
# Run the percent cells from top to bottom in the existing project `.venv`.
# Restart the Interactive Window kernel first; README describes dependencies and the project-local cache.
#
# > **⚠️ Watch Out — An encoder does not supply disaster labels**
# >
# > MiniLM produces numerical features. Our classifiers must still learn the relationship between those features and the recorded training labels.
# > This is supervised classification with a frozen encoder, not zero-shot classification or Transformer fine-tuning.

# %% [markdown]
# ## 1. Preserve the observations and evaluation boundaries
#
# **What is the Question ❓** Are we comparing representations on the same observations?
#
# **🎯 Goal & Approach — WHY:** Reuse the main analysis's normalization, retention policy, split and ten training folds.
#
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

# Recreate baseline rows with source indices; cv_folds stores positions within the training partition.
if "__file__" in globals():
    project_root = Path(__file__).resolve().parent
else:
    project_root = Path.cwd()
data_path = project_root / "DATA" / "train.csv"
if not data_path.is_file():
    raise FileNotFoundError(f"Cannot find {data_path}. Open the Interactive Window in the repository root.")

print("Python:", sys.version.split()[0], "Interpreter:", sys.executable)
print("pandas:", pd.__version__, "scikit-learn:", sklearn.__version__)
tweets = pd.read_csv(data_path)
print("Rows and columns:", tweets.shape)

# %% [markdown]
# Reproduce the main file's preparation in visible steps before defining any model features.
# The source indices and retained row order must survive this sequence.

# %%
# Keep original text and source indices; this new column is the baseline's prepared input.
prepared = tweets.copy()
prepared["prepared_text"] = prepared["text"].str.lower()
prepared["prepared_text"] = prepared["prepared_text"].str.replace(r"\s+", " ", regex=True)
prepared["prepared_text"] = prepared["prepared_text"].str.strip()
print("Original/prepared example:")
print(prepared[["text", "prepared_text", "target"]].head(1).to_string())

# %% [markdown]
# The copy retains original wording and source indices beside the prepared input.
# Group by that prepared string to reproduce the main file's conflict policy.

# %%
# One value counts distinct labels per normalized text; conflicting_keys contains disputed text keys.
labels_per_text = prepared.groupby("prepared_text")["target"].nunique()
conflicting_keys = labels_per_text[labels_per_text > 1].index
conflict_mask = prepared["prepared_text"].isin(conflicting_keys)
consistent_rows = prepared.loc[~conflict_mask]
print("Conflicting groups / excluded rows:", len(conflicting_keys), conflict_mask.sum())
print("Rows available for repeat checking:", len(consistent_rows))

# %% [markdown]
# Removing all 56 rows in 18 conflicting groups leaves 7,557 rows.
# Use `consistent_rows` to remove later repeats while preserving each group's first source occurrence.

# %%
# The mask flags later matching inputs; the retained copy keeps source order and index gaps.
repeat_mask = consistent_rows.duplicated(subset="prepared_text", keep="first")
model_tweets = consistent_rows.loc[~repeat_mask].copy()
print("Later repeats / retained rows:", repeat_mask.sum(), len(model_tweets))
print(model_tweets[["prepared_text", "target"]].head(3).to_string())

# %% [markdown]
# Removing 74 later repeats leaves the same 7,483-row population used by the baseline.
# Split this whole table next, keeping each text paired with its recorded label.

# %% [markdown]
# Split whole observations before selecting their text and label columns.

# %%
# Split whole rows so text, labels and original source indices keep the same membership and order.
train_rows, holdout_rows = train_test_split(
    model_tweets, test_size=0.20, random_state=42, stratify=model_tweets["target"]
)
X_train = train_rows["prepared_text"]
y_train = train_rows["target"]
X_holdout = holdout_rows["prepared_text"]
y_holdout = holdout_rows["target"]
print("Training / holdout rows:", len(train_rows), len(holdout_rows))

# %% [markdown]
# The split assigns 5,986 rows to training and 1,497 to holdout, with text and labels selected from the same rows.
# Reuse the training partition to define the ten folds; the holdout remains outside them.

# %% [markdown]
# Keep the ten training-fold memberships for both classifiers to reuse.
# Their positions refer to rows within training, whereas the pandas indices still identify CSV source rows.

# %%
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
# <details><summary>💡 Explain this code</summary>
#
# The seeded split retains aligned text, labels, source indices, and row order from the approved preparation.
# `list(cv_splitter.split(...))` stores ten pairs of training-relative positions; both classifiers reuse them, while printed sizes and overlap checks expose the populations.
# Fold positions are not CSV source indices, and matching observations does not establish independent events or accurate annotations.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# The preparation check on 2026-10-29 reproduced 7,613 source rows, 7,483 retained rows, 5,986 training rows and 1,497 holdout rows.
# Temporary verification compared complete tables, source indices, labels, row order and every fold against the main analysis.
# Equal membership controls which observations each comparison uses; it does not establish independent events or correct annotations.
#
# > **⚠️ Watch Out — A familiar holdout is not a fresh final test**
# >
# > This project has already inspected its holdout. Any additional holdout results are retrospective educational evidence.
# > Both classifiers are specified before evaluation; holdout results must not choose the encoder, classifier settings or preparation.
#
# **🧠 What We Learned**
# Matching observations makes the workflows comparable, but does not establish performance
# across separate events. That needs event relationships and fresh evaluation evidence.

#
# ### 🤔 Question Yourself
# Two workflows use the same rows and split. Does that establish independence from previously seen events?
#
# <details><summary>💡 Show answer</summary>
#
# No. Matching rows controls which observations are compared; it does not establish event separation or annotation accuracy.
# The already inspected holdout remains retrospective evidence, even when both workflows reuse exactly the same observations.
#
# </details>

# %% [markdown]
# ## 2. Turn each tweet into a frozen representation
#
# **What is the Question ❓** What does the pretrained encoder supply, and what will our classifiers still learn?
#
# **🎯 Goal & Approach — WHY:** Produce one 384-number vector per tweet using `sentence-transformers/all-MiniLM-L6-v2`.
#
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
# **📋 Source & Context:** The MasterSchool guided notebook motivates classifier comparison and fold-level F1.
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
# **💻 Code / Method — HOW:** Ignore padding positions, average the remaining token vectors, then divide each resulting vector by its length.
# This produces comparable vector lengths; it neither guarantees preserved meaning nor makes individual dimensions interpretable as words.
# Word pieces are tokenizer units that can split a word; the 256-position limit also includes special start/end tokens.
#
# **Setup:** The pinned snapshot must already exist in `.cache/huggingface/hub/`, with the nine files listed below.
# Explicit local lookup avoids downloading replacements or silently using a global cache; see README for separate resource setup.

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

# Locate only the pinned project snapshot; local_files_only prevents downloading missing model resources.
model_cache_dir = project_root / ".cache" / "huggingface" / "hub"
try:
    model_path = Path(snapshot_download(
        repo_id=model_id, revision=model_revision, cache_dir=model_cache_dir,
        allow_patterns=model_files, local_files_only=True,
    ))
except LocalEntryNotFoundError as error:
    raise FileNotFoundError(
        f"Missing {model_id} at {model_revision} in {model_cache_dir}. "
        "See README for separate resource setup; nothing was downloaded."
    ) from error

# An existing snapshot can be incomplete; require all nine files before loading the encoder.
missing_model_files = [name for name in model_files if not (model_path / name).is_file()]
if missing_model_files:
    raise FileNotFoundError(f"Incomplete snapshot {model_path}: {missing_model_files}. See README; no download was attempted.")

# %% [markdown]
# Load the verified local snapshot, then explicitly keep the encoder frozen and in evaluation mode.

# %%
# Read only cached tokenizer/configuration and safetensors weights; no repository Python code is used.
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
encoder = AutoModel.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
)
encoder = encoder.to("cpu")
encoder.eval()  # Disable training-time dropout, so inference uses the fixed model consistently.
encoder.requires_grad_(False)  # No encoder parameter will receive a training gradient.
print("PyTorch:", torch.__version__, "Transformers:", transformers.__version__)
print("Encoder:", model_id, "Revision:", model_revision)
print("Model snapshot:", model_path)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The model ID, pinned revision, and required-file list locate a complete local snapshot; the tokenizer and encoder load with remote code disabled.
# `.to("cpu")` selects the device, `eval()` disables training-time dropout, and `requires_grad_(False)` prevents encoder parameter gradients.
# These externally learned features are frozen here; that does not establish whether pretraining overlapped the tweets.
#
# </details>

# %% [markdown]
# ### A small pooling example before processing tweets
# These original, invented values illustrate the calculation; they are not encoder output or dataset measurements.
# One example has two real token positions, one padding position and three features instead of MiniLM's 384.
# The [model card's Transformers example](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-huggingface-transformers)
# supports the mask → mean → normalization sequence used here.

# %%
# Shape is (examples, token positions, features); the last position is artificial padding.
pooling_demo_vectors = torch.tensor([[
    [2.0, 4.0, 6.0],
    [4.0, 2.0, 0.0],
    [99.0, 99.0, 99.0],
]])
pooling_demo_attention = torch.tensor([[1, 1, 0]])
print("Invented token vectors:", pooling_demo_vectors)
print("Attention mask:", pooling_demo_attention)
print("Vector shape:", tuple(pooling_demo_vectors.shape))

# %% [markdown]
# The mask's two ones identify real positions; its zero identifies padding.
# Expand that mask across the feature axis before multiplying it with the vectors.

# %%
# Add a feature axis: (1, 3) becomes (1, 3, 1), then match the vectors' numeric type.
pooling_demo_mask = pooling_demo_attention.unsqueeze(-1)
pooling_demo_mask = pooling_demo_mask.to(pooling_demo_vectors.dtype)
# Broadcasting applies each position's flag to all three of its features.
pooling_demo_masked = pooling_demo_vectors * pooling_demo_mask
print("Expanded mask shape:", tuple(pooling_demo_mask.shape))
print("Masked vectors:", pooling_demo_masked)

# %% [markdown]
# Padding's `[99, 99, 99]` now contributes `[0, 0, 0]`; the two real vectors remain unchanged.
# Sum the real vectors and divide by their count, rather than by all three padded positions.

# %%
# Sum along token positions, leaving one feature vector and one non-padding count per example.
pooling_demo_sums = pooling_demo_masked.sum(dim=1)
pooling_demo_counts = pooling_demo_mask.sum(dim=1)
pooling_demo_safe_counts = pooling_demo_counts.clamp(min=1)
pooling_demo_mean = pooling_demo_sums / pooling_demo_safe_counts
print("Vector sums:", pooling_demo_sums)
print("Non-padding count:", pooling_demo_counts)
print("Mean vector:", pooling_demo_mean)

# %% [markdown]
# `[6, 6, 6] / 2` gives the mean vector `[3, 3, 3]`.
# The count guard prevents division by zero; next scale this vector to unit Euclidean length.

# %%
# Normalize each example's feature vector; this fixed operation learns no parameters.
pooling_demo_unit = functional.normalize(pooling_demo_mean, p=2, dim=1)
print("Unit vector:", pooling_demo_unit)
print("Vector length:", torch.linalg.vector_norm(pooling_demo_unit, dim=1))

# %% [markdown]
# The coordinates become approximately `[0.5774, 0.5774, 0.5774]`, with vector length approximately one.
# The tweet loop applies these same operations to 384-feature token vectors, retaining special tokens while ignoring padding.
# This illustration changes neither the encoder nor the tweet inputs.

# %%
# Limit each input and process 32 tweets together, using the existing frozen encoder.
maximum_tokens = 256
batch_size = 32

# %% [markdown]
# First inspect untruncated input lengths; reuse these same ordered text lists during encoding.

# %%
# Each dictionary value is one partition's ordered tweet list, aligned with its existing labels.
partition_text_values = {}
for partition_name, texts in [("training", X_train), ("holdout", X_holdout)]:
    text_values = texts.tolist()
    partition_text_values[partition_name] = text_values
    full_token_ids = tokenizer(text_values, truncation=False)["input_ids"]
    token_lengths = np.array([len(token_ids) for token_ids in full_token_ids])
    exceeds_token_limit = token_lengths > maximum_tokens
    print(partition_name, "inputs exceeding the limit:", int(exceeds_token_limit.sum()))

# %% [markdown]
# Neither partition contains an input exceeding 256 token positions, including special tokens.
# The configured limit therefore truncates no tweet; encode the inspected lists in batches next.

# %%
# Rows will follow input order; each partition's matrix will have 384 feature columns.
embedding_matrices = {}
encoding_seconds = {}
for partition_name, text_values in partition_text_values.items():
    started = perf_counter()
    batch_embeddings = []
    for start in range(0, len(text_values), batch_size):
        batch = text_values[start:start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=maximum_tokens, return_tensors="pt"
        )
        with torch.inference_mode():
            # Encode: each token receives a contextual 384-feature vector.
            encoder_output = encoder(**encoded)
            token_vectors = encoder_output.last_hidden_state

            # Mask: expand each position's flag across features, then zero padding vectors.
            attention_mask = encoded["attention_mask"]
            token_mask = attention_mask.unsqueeze(-1)
            token_mask = token_mask.to(token_vectors.dtype)
            masked_token_vectors = token_vectors * token_mask

            # Pool: average non-padding positions; special tokens still contribute.
            vector_sums = masked_token_vectors.sum(dim=1)
            token_counts = token_mask.sum(dim=1)
            safe_token_counts = token_counts.clamp(min=1)
            tweet_vectors = vector_sums / safe_token_counts
            unit_vectors = functional.normalize(tweet_vectors, p=2, dim=1)

        # Inspect one actual batch per partition without repeating encoder inference.
        if start == 0:
            print(partition_name, "first batch token IDs:", tuple(encoded["input_ids"].shape))
            print(partition_name, "contextual token vectors:", tuple(token_vectors.shape))
            print(partition_name, "expanded padding mask:", tuple(token_mask.shape))
            print(partition_name, "pooled tweet vectors:", tuple(tweet_vectors.shape))

        # Collect: convert each batch to NumPy, retaining tweet order across batches.
        batch_array = unit_vectors.cpu().numpy()
        batch_embeddings.append(batch_array)

    embedding_matrices[partition_name] = np.concatenate(batch_embeddings, axis=0)
    encoding_seconds[partition_name] = perf_counter() - started
    print(partition_name, "embedding shape:", embedding_matrices[partition_name].shape)
    print(partition_name, "encoding seconds:", round(encoding_seconds[partition_name], 2))

# %% [markdown]
# Inspect the finished matrices separately from the batch-processing loop.
# Each matrix row must still align with the corresponding partition's label position.

# %%
# Retrieve each partition's matrix; the range and vector lengths describe features, not prediction scores.
train_embeddings = embedding_matrices["training"]
holdout_embeddings = embedding_matrices["holdout"]
print("Training feature range:", train_embeddings.min(), train_embeddings.max())
print("Training vector lengths, min/max:", np.linalg.norm(train_embeddings, axis=1).min(), np.linalg.norm(train_embeddings, axis=1).max())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The batch loop repeats the illustrated mask, average and normalization operations on contextual token vectors.
# `inference_mode()` disables gradient tracking; concatenating NumPy batches preserves the input order needed for label alignment.
# Negative feature values are valid, while pooling can lose information; the next classifiers learn from these fixed vectors.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# The training matrix has shape `(5986, 384)`; the holdout matrix has shape `(1497, 384)`.
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
# **🧠 What We Learned**
# Encoding produces features; labeled validation supplies evidence of their usefulness.
# The next unit shows higher mean CV F1 for both classifiers than their historical TF-IDF references.

#
#
# ### 🤔 Question Yourself
# In the invented pooling example, why is the masked sum divided by two rather than three?
# What is the mean vector before L2 normalization?
#
# <details><summary>💡 Show answer</summary>
#
# The mask `[1, 1, 0]` selects two real token positions and excludes the padding position.
# Their vector sum is `[6, 6, 6]`, so dividing by two gives `[3, 3, 3]`.
# Padding's invented `[99, 99, 99]` must contribute neither to the sum nor to the count.
#
# </details>

# %% [markdown]
# ## 3. Compare the same classifiers in the same ten folds
#
# **What is the Question ❓** Does MiniLM improve training-only CV F1 for Logistic Regression or LinearSVC?
#
# **🎯 Goal & Approach — WHY:** Fit fresh classifiers on nine folds and score the remaining fold, repeating all ten partitions.
#
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
# Clone a fresh classifier per fold; positional embedding rows must match y_train.iloc positions.
classifier_options = [
    ("Logistic Regression", LogisticRegression()),
    ("LinearSVC", LinearSVC(random_state=42)),
]
# %% [markdown]
# Follow one classifier through the same ten folds, keeping fitting, prediction, and scoring as separate steps.

# %%
# Each record describes one classifier and one withheld fold; feature rows and labels share fold positions.
fold_records = []
for classifier_name, classifier in classifier_options:
    for fold_number, fold_positions in enumerate(cv_folds, start=1):
        fit_positions, validation_positions = fold_positions
        fit_vectors = train_embeddings[fit_positions]
        fit_labels = y_train.iloc[fit_positions]
        validation_vectors = train_embeddings[validation_positions]
        validation_labels = y_train.iloc[validation_positions]

        # A fresh clone learns only from this fold's fitting observations.
        fold_classifier = clone(classifier)
        fold_classifier.fit(fit_vectors, fit_labels)
        predictions = fold_classifier.predict(validation_vectors)
        score = f1_score(validation_labels, predictions, pos_label=1, zero_division=0)
        fold_records.append({"classifier": classifier_name, "fold": fold_number, "f1": score})
        print(f"{classifier_name}, fold {fold_number}: F1={score:.6f}")

# %% [markdown]
# Build one row per fold to compare the two classifiers on matching validation observations.

# %%
# pivot changes the display arrangement, retaining all 20 measured fold scores.
fold_results = pd.DataFrame(fold_records)
fold_score_table = fold_results.pivot(index="fold", columns="classifier", values="f1")
print("Actual training-only fold F1:")
print(fold_score_table.round(6).to_string())

# %% [markdown]
# Logistic Regression leads in six of ten folds, while LinearSVC leads in four.
# Calculate each classifier's mean and spread from these same scores to summarize the comparison.

# %%
# Keep classifier order and the original NumPy calculations; each summary row describes ten fold scores.
embedding_summary_records = []
for classifier_name, _ in classifier_options:
    classifier_rows = fold_results["classifier"] == classifier_name
    scores = fold_results.loc[classifier_rows, "f1"].to_numpy()
    embedding_summary_records.append({
        "classifier": classifier_name, "representation": "MiniLM embeddings",
        "mean_f1": scores.mean(), "std_f1": scores.std(ddof=0), "evidence": "Current run",
    })
embedding_summary = pd.DataFrame(embedding_summary_records)
print("Embedding CV summaries:")
print(embedding_summary.round(6).to_string(index=False))

# %% [markdown]
# Mean F1 is approximately 0.773690 for Logistic Regression and 0.768419 for LinearSVC.
# These averages summarize these folds; use the same classifier's historical TF-IDF means for the representation comparison next.

# %%
# Label the six TF-IDF summaries as rounded historical evidence; this file does not rerun those fits.
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

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The reference table labels six rounded historical TF-IDF summaries; concatenation adds the two current MiniLM summaries with a new display index.
# Compare rows for the same classifier; Section 7 calculates their mean differences before plotting them.
# Concatenation neither reruns historical fits nor recovers their original paired fold scores.
#
# </details>

# %%
# Connect scores by fold identifier, not time; matching fold numbers use the same validation observations.
fold_figure = px.line(
    fold_results, x="fold", y="f1", color="classifier", markers=True,
    title="MiniLM: class-1 F1 across ten training folds",
    labels={"fold": "Training validation fold", "f1": "Class-1 F1", "classifier": "Classifier"},
)
fold_figure.update_layout(height=360)
fold_figure.update_xaxes(dtick=1)
fold_figure.update_yaxes(range=[0, 1])
fold_figure.show()

# %% [markdown]
# The fold scores vary and the leading classifier changes across folds.
# Compare configuration means and descriptive spreads next, using the already prepared comparison table.

# %%
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
# <details><summary>💡 Explain this code</summary>
#
# `px.line()` shows classifier F1 by fold; `px.scatter()` shows configuration means with fold standard deviations.
# Symbols and hover retain provenance, while `<br>` only wraps display labels; neither chart fits a model.
# F1 axes span zero to one. Error bars describe fold variation, not statistical significance or independent uncertainty bounds.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# Both MiniLM workflows increased mean training-only F1 compared with the same classifier's historical TF-IDF configurations.
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
# **🧠 What We Learned**
# A higher mean describes these folds. It does not establish fewer consequential errors
# on other events; ten overlapping training folds cannot supply independent population evidence.

#
# ### 🤔 Question Yourself
# A classifier has a higher mean fold F1 and overlapping error bars. Have we established a statistically reliable advantage?
#
# <details><summary>💡 Show answer</summary>
#
# No. These standard deviations describe variation among the ten fold scores, not a significance test.
# Training subsets overlap, and a higher mean does not establish reliability on independently collected events.
#
# </details>

# %% [markdown]
# ## 4. Report the two predeclared models on the existing holdout
#
# **What is the Question ❓** What do the two frozen-feature workflows predict on the unchanged 1,497 holdout rows?
#
# **🎯 Goal & Approach — WHY:** Refit each classifier on all training embeddings, then evaluate the holdout once with fixed settings.
# Both classifiers are evaluated regardless of their CV ranking; this adds two final classifier fits without choosing a CV winner.
#
# **🔎 Guiding Question:** Which comparison has a matching historical holdout baseline?
# Only TF-IDF + Logistic Regression has an existing holdout result. LinearSVC's historical figures are CV results and belong in the preceding comparison.

# %%
# Fit both fixed templates on all training vectors; store predictions in the unchanged holdout order.
holdout_predictions = {}
for classifier_name, classifier in classifier_options:
    fitted_classifier = clone(classifier)
    fitted_classifier.fit(train_embeddings, y_train)
    predictions = fitted_classifier.predict(holdout_embeddings)
    holdout_predictions[classifier_name] = predictions
    print(classifier_name, "holdout predictions:", len(predictions))

# %% [markdown]
# Each classifier returns 1,497 labels aligned with the existing holdout inputs.
# Compare those arrays with the recorded labels next; their length alone measures no correctness.

# %%
# Reuse stored predictions for every metric; dictionaries preserve the classifier names and input order.
holdout_records = []
holdout_confusion_counts = {}
print("Holdout support by recorded label:", y_holdout.value_counts().sort_index().to_dict())
for classifier_name, predictions in holdout_predictions.items():
    holdout_records.append({
        "configuration": classifier_name + " + MiniLM",
        "f1": f1_score(y_holdout, predictions, pos_label=1, zero_division=0),
        "precision": precision_score(y_holdout, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_holdout, predictions, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_holdout, predictions),
        "evidence": "Current retrospective run",
    })
    # Recorded-label rows and prediction columns fix missed positives at [1, 0] and false alarms at [0, 1].
    counts = confusion_matrix(y_holdout, predictions, labels=[0, 1])
    holdout_confusion_counts[classifier_name] = counts
    print("\n", classifier_name, "holdout matrix; rows=recorded, columns=predicted, labels=[0, 1]:")
    print(counts)

# %% [markdown]
# Logistic Regression misses 165 recorded positives and produces 125 false positives; LinearSVC has 160 and 122, respectively.
# Use the stored matrices for the heatmaps, then the metric records for the baseline comparison.
# Both remain retrospective results on the already inspected holdout.

# %% [markdown]
# Draw the stored confusion matrices after inspecting their counts; plotting performs no additional fitting or prediction.

# %%
# Each dictionary entry supplies one classifier name and its already computed 2-by-2 count matrix.
for classifier_name, counts in holdout_confusion_counts.items():
    confusion_figure = px.imshow(
        counts, x=["0: non-disaster", "1: disaster"], y=["0: non-disaster", "1: disaster"],
        text_auto=True, color_continuous_scale="Blues", zmin=0, zmax=int(y_holdout.value_counts().max()),
        labels={"x": "Predicted label", "y": "Recorded label", "color": "Tweets"},
        title=classifier_name + " + MiniLM: holdout",
    )
    confusion_figure.update_layout(height=360)
    confusion_figure.show()

# %% [markdown]
# Compare the two current result rows with the documented baseline in a separate summary table.

# %%
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

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Documented TN, FP, FN, and TP counts reconstruct the original baseline metrics without refitting it.
# The summary combines that labeled historical row with the two current retrospective records; rounding affects display only.
# Neither classifier is selected by its holdout score, and this examined holdout is not an independent final test.
#
# </details>

# %%
# Attach arrays before filtering; head(3) retains holdout order rather than sorting source indices.
holdout_review = holdout_rows[["text", "target"]].copy()
for classifier_name, predictions in holdout_predictions.items():
    holdout_review[classifier_name] = predictions
    error_mask = holdout_review[classifier_name] != holdout_review["target"]
    print("\nFirst three holdout errors:", classifier_name)
    print(holdout_review.loc[error_mask, ["text", "target", classifier_name]].head(3).to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Assign existing prediction arrays by position to the copied, source-indexed holdout table before filtering mismatches with `.loc`.
# `head(3)` selects the first three errors in holdout order, potentially mixing false positives and false negatives.
# This differs from the baseline's separately selected, source-sorted error examples; neither selection establishes prevalence or model causes.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# Both workflows were evaluated on the same 1,497 observations: 860 recorded non-disasters and 637 disasters.
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

#
# ### 🤔 Question Yourself
# If the classifiers were specified before this comparison, why is the holdout still not a fresh independent final test?
#
# <details><summary>💡 Show answer</summary>
#
# This project had already inspected that holdout before the MiniLM study.
# Predeclaring these classifiers does not undo earlier exposure.
# The scores describe the fixed workflows on familiar observations; broader claims require fresh evaluation evidence.
#
# </details>

# %% [markdown]
# ## 5. Insights and evidence limits
#
# **What is the Question ❓** What does this experiment establish?
#
# **🎯 Goal & Approach — WHY:** Connect measured differences to a proportionate conclusion.
#
# **✅ Evidence, Answer & Limits:** MiniLM increased both classifiers' mean CV F1 on the unchanged training folds.
# With Logistic Regression fixed, holdout F1 increased with fewer false negatives and more false positives.
# These findings describe the specified workflows on familiar observations, not universal Transformer superiority or a fresh final test.
#
# **🧠 What We Learned:** Matching labels, rows and settings makes the measured differences interpretable.
# External pretraining supplies additional information, even though only our classifiers learn from disaster labels.
# Improvement on separately collected, independently annotated reports remains unestablished.

#
# ### 🤔 Question Yourself
# Does the measured MiniLM advantage establish that Transformers will outperform TF-IDF on another collection of tweets?
#
# <details><summary>💡 Show answer</summary>
#
# No. The comparison supports differences between these specified workflows on these observations.
# Frozen MiniLM also brings information from external pretraining.
# A claim about another population needs a task-aligned comparison on representative, independently evaluated examples.
#
# </details>

# %% [markdown]
# ## 6. Ethical interpretation and responsible use
#
# **What is the Question ❓** What may we responsibly claim from the observed improvements?
#
# **🎯 Goal & Approach — WHY:** Interpret the error trade-off without inventing operational benefits.
#
# **✅ Evidence, Answer & Limits:** MiniLM + Logistic Regression misses 43 fewer recorded positives and produces 17 additional false positives.
# These counts measure prediction differences; they establish neither people helped nor consequences of false alarms.
# Annotation uncertainty, retained English-only examples and possible pretraining overlap limit broader claims.
#
# > **⚠️ Watch Out — A better F1 is not an operational recommendation**
# >
# > Fairness across populations, reliability on new events and real-world error costs remain unmeasured.
# > Assessing them needs a defined use context, affected users, annotation review and fresh event-level evaluation.
#
# **🧠 What We Learned:** The appropriate balance between missed reports and false alarms is not established here.
# Use the following charts to inspect measured differences while retaining these evidence limits.
#
# **⚖️ Ethical Takeaway:** Report error counts and uncertainty together; neither representation establishes emergency-response readiness.

#
# ### 🤔 Question Yourself
# Fewer missed reports come with more false alarms. Can F1 alone decide which balance is appropriate?
#
# <details><summary>💡 Show answer</summary>
#
# No. F1 summarizes agreement with recorded labels but does not measure the consequences or costs of either mistake.
# That judgment needs a defined use context, acceptable error trade-offs, and fresh evaluation evidence.
#
# </details>

# %% [markdown]
# ## 7. Visual comparison — TF-IDF and Hugging Face features
#
# **What is the Question ❓** How large are the measured differences, and which errors change when we replace TF-IDF with MiniLM?
#
# **🎯 Goal & Approach — WHY:** Finish with three complementary Plotly views, derived from the result tables and predictions already computed above.
#
# **🔎 Guiding Question:** Which comparisons keep the classifier fixed, and where is a matching baseline unavailable?
#
# **Comparison boundary:** CV includes both classifiers against their two historical TF-IDF configurations, using the same training rows and folds.
# Holdout views compare Logistic Regression only, because LinearSVC has no historical holdout baseline for isolating the representation change.
# MiniLM + LinearSVC's holdout result remains in Section 4; these figures do not select a model or refit anything.
#
# ### A. Change in training-only mean F1
# **🎯 Goal & Approach — WHY:** A difference plot makes the size of each change easier to read than eight nearby averages.
#
# **💻 Code / Method — HOW:** Subtract each classifier's historical TF-IDF mean from its current MiniLM mean; positive bars indicate higher mean F1.
# These are absolute F1 differences, not relative percentage improvements; historical rounding makes the differences approximate.
#
# Plotting reference: [Plotly grouped bars and value labels](https://plotly.com/python/bar-charts/).

# %%
# Match classifier names before subtracting CV means; each row compares one historical/current representation pair.
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
print("Chart input: matched-classifier changes in mean CV F1:")
print(cv_changes.drop(columns="change_label").round(4).to_string(index=False))

# %% [markdown]
# All four differences are positive: approximately +0.0231 / +0.0444 for Logistic Regression and +0.0130 / +0.0155 for LinearSVC.
# Each row compares the same classifier across representations; plot these differences without fitting another model.

# %%
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
# <details><summary>💡 Explain this code</summary>
#
# `iterrows()` visits MiniLM summaries; a Boolean filter selects the same classifier's historical TF-IDF settings.
# Each record retains both means and their difference; signed strings format labels while bars use stored values.
# These are absolute F1 differences, not percentage growth or paired fold gains; historical rounding limits precision.
#
# </details>

# %% [markdown]
# **✅ Evidence, Answer & Limits:** The bars display the four differences calculated in the preceding table.
# The larger bigram comparison partly reflects its lower TF-IDF reference; it does not identify the best achievable model.
# Historical means alone cannot establish paired improvements in every fold or statistical significance.
#
# ### B. Holdout metrics with Logistic Regression fixed
# **🎯 Goal & Approach — WHY:** F1 summarizes a trade-off, so show precision, recall and accuracy beside it rather than treating one score as the whole result.
#
# **💻 Code / Method — HOW:** Display the earlier TF-IDF and current MiniLM scores side by side on the same zero-to-one scale.
# This is retrospective evidence on the same 1,497 observations; class-1 F1, precision and recall concern the recorded disaster label.

# %%
# Select the two Logistic Regression workflows and retain evidence labels for the retrospective comparison.
representation_names = {
    "Logistic Regression + TF-IDF (1, 1)": "TF-IDF (earlier)",
    "Logistic Regression + MiniLM": "MiniLM (current)",
}
matched_configuration_mask = holdout_summary["configuration"].isin(representation_names)
matched_holdout = holdout_summary.loc[matched_configuration_mask].copy()
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
print("Chart input: one row per representation and metric:")
print(holdout_metric_comparison[["representation", "metric", "score"]].round(4).to_string(index=False))

# %% [markdown]
# Eight rows represent two Logistic Regression workflows crossed with four metrics.
# MiniLM has higher F1, recall and accuracy, with lower precision; show those existing scores side by side next.

# %%
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
# <details><summary>💡 Explain this code</summary>
#
# `isin()` selects the two Logistic Regression workflows; `map()` supplies display names while retaining their evidence labels.
# `melt()` creates one row per representation/metric pair; grouped bars display those existing scores on the same zero-to-one scale.
# Reshaping changes no metric and makes no retrospective holdout result an independent final evaluation.
#
# </details>

# %% [markdown]
# **✅ Evidence, Answer & Limits:** F1 rises from `0.7308` to `0.7650`, recall from `0.6735` to `0.7410`, and accuracy from `0.7889` to `0.8063`.
# Precision falls from `0.7989` to `0.7906`; the representation change does not improve every metric.
# The bars reuse existing results; reshaping with [pandas `melt`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html) does not recalculate or average scores.
#
# ### C. Translate the trade-off into error counts
# **🎯 Goal & Approach — WHY:** Error counts make the measured trade-off concrete without assigning an unsupported cost to either error type.
#
# **💻 Code / Method — HOW:** Subtract earlier Logistic Regression error counts from its MiniLM counts; negative means fewer errors of that type.
# False negatives miss recorded disasters; false positives predict disaster for recorded non-disasters.

# %%
# Recover missed positives at [1, 0] and false alarms at [0, 1] from the existing predictions.
minilm_lr_counts = confusion_matrix(y_holdout, holdout_predictions["Logistic Regression"], labels=[0, 1])

# One row per error type keeps the earlier and current counts visible in the hover details.
error_changes = pd.DataFrame({
    "error_type": ["Missed disasters", "False alarms"],
    "earlier_tfidf": [baseline_fn, baseline_fp],
    "current_minilm": [int(minilm_lr_counts[1, 0]), int(minilm_lr_counts[0, 1])],
})
error_changes["count_change"] = error_changes["current_minilm"] - error_changes["earlier_tfidf"]
error_changes["change_label"] = [f"{value:+d}" for value in error_changes["count_change"]]
print("Chart input: earlier counts, current counts and signed changes:")
print(error_changes.drop(columns="change_label").to_string(index=False))

# %% [markdown]
# The two rows show 43 fewer missed positives and 17 additional false alarms.
# Plot their signed changes on a shared scale to make this trade-off visible.

# %%
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
# <details><summary>💡 Explain this code</summary>
#
# The fixed confusion-matrix order supplies missed positives and false alarms; subtracting earlier counts gives signed net changes.
# Symmetric axis limits use the largest absolute change so increases and decreases share a comparable scale.
# Net differences do not identify particular corrected tweets, and the two error types have different recorded-label denominators.
#
# </details>

# %% [markdown]
# **✅ Evidence, Answer & Limits:** These are net count differences, not proof that exactly 43 particular tweets were corrected.
# Some predictions may change in opposite directions.
# The two error types have different label-group denominators, so these bars show counts rather than comparable error rates.
#
# **🧠 What We Learned**
# The charts establish a measured error trade-off; usefulness in a real setting remains unestablished.
# That decision needs fresh evaluation and evidence about error costs, alongside the ethical limits in Section 6.

#
# ### 🤔 Question Yourself
# The chart shows 43 fewer missed positives. Does that mean exactly 43 previously missed tweets were corrected?
#
# <details><summary>💡 Show answer</summary>
#
# No. It is a net difference between error counts.
# Some previously missed tweets may be found while other previously found tweets become misses.
# Identifying those movements requires comparing paired predictions for each tweet.
#
# </details>

# %% [markdown]
# ## 8. Beyond the baseline — adjust the classifiers, keep MiniLM fixed
#
# **What is the Question ❓** Can a small change to classifier regularization improve the frozen-MiniLM results without Transformer fine-tuning?
#
# **🎯 Goal & Approach — WHY:** Compare `C=0.1, 1, 10` for Logistic Regression and LinearSVC on the same ten training folds.
# The encoder, embeddings, preprocessing, class weights, and prediction rules stay fixed; only the downstream classifiers learn again.
#
# **WHY C:** Smaller values penalize large classifier coefficients more strongly; larger values allow more flexibility.
# Equal `C` values do not make Logistic Regression and LinearSVC equivalent because their learning objectives differ.
# The earlier TF-IDF comparison cannot determine which setting suits MiniLM, whose features describe text differently.
#
# MasterSchool Sprint 3 lists optional regularization trials; DataCamp's [Regularizing models with Twitter data](https://campus.datacamp.com/courses/sentiment-analysis-in-python/lets-predict-the-sentiment?ex=11)
# illustrates the idea for sentiment. This assistant-authored adaptation uses our recorded disaster labels and existing training folds.
# Installed-version references: [LogisticRegression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html),
# [LinearSVC](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html), and [cross_val_predict](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.cross_val_predict.html).
#
# **💻 Code / Method — HOW:** Reuse the existing training embeddings; each row receives one prediction from a classifier trained on the other nine folds.
# Six configurations require 60 classifier fits; neither the encoder nor the original classifier templates are changed.
# The repeated `C=1` settings provide matching references and withheld predictions, which the earlier CV retained only as scores.
#
# Pooled recall, precision, F1, and error counts summarize all 5,986 withheld predictions per configuration.
# Mean fold F1 and its standard deviation summarize ten separate scores; pooled F1 is a different calculation.
# These are exploratory training comparisons, not new holdout results or evidence of operational reliability.

# %%
from sklearn.model_selection import cross_val_predict

# Fix classifier C choices without re-encoding; each prediction column retains training source-row alignment.
minilm_c_values = [0.1, 1.0, 10.0]
minilm_c_oof_predictions = pd.DataFrame(index=y_train.index)

# %% [markdown]
# First generate one out-of-fold prediction column per classifier/C combination, reusing the same fixed embedding rows and folds.

# %%
for minilm_c_name, minilm_c_template in classifier_options:
    for minilm_c_value in minilm_c_values:
        # Clone before changing C, preserving the original C=1 templates and their remaining settings.
        minilm_c_candidate = clone(minilm_c_template)
        minilm_c_candidate.set_params(C=minilm_c_value)
        minilm_c_key = f"{minilm_c_name}, C={minilm_c_value:g}"

        # Each fold fits a fresh classifier; returned labels follow the original training-row order.
        # Fixed per-tweet embeddings need no refitting inside folds; default predict rules remain unchanged.
        minilm_c_oof_predictions[minilm_c_key] = cross_val_predict(
            minilm_c_candidate, train_embeddings, y_train, cv=cv_folds, method="predict", n_jobs=1,
        )
        print(f"Completed ten training-only folds: {minilm_c_key}")

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Clone each classifier before changing C, keeping the original templates available for the earlier comparisons.
# `cross_val_predict()` fits fresh classifiers inside the recorded folds and returns predictions in training-row order.
# Each column identifies one fixed configuration; the encoder is not retrained and the holdout does not choose C.
#
# </details>

# %% [markdown]
# Evaluate those stored columns without fitting again; keep pooled metrics separate from the mean and spread of fold F1.

# %%
# First summarize all withheld predictions together, keeping one row per classifier/C combination.
minilm_c_pooled_records = []
for minilm_c_key, minilm_c_predictions in minilm_c_oof_predictions.items():
    minilm_c_confusion_counts = confusion_matrix(y_train, minilm_c_predictions, labels=[0, 1])
    # The fixed order gives TN, FP, FN and TP against recorded labels.
    minilm_c_tn, minilm_c_fp, minilm_c_fn, minilm_c_tp = minilm_c_confusion_counts.ravel()
    minilm_c_pooled_records.append({
        "configuration": minilm_c_key,
        "pooled_recall": recall_score(y_train, minilm_c_predictions, pos_label=1),
        "pooled_precision": precision_score(y_train, minilm_c_predictions, pos_label=1),
        "pooled_f1": f1_score(y_train, minilm_c_predictions, pos_label=1),
        "missed_positives": minilm_c_fn,
        "false_alarms": minilm_c_fp,
    })
minilm_c_pooled_summary = pd.DataFrame(minilm_c_pooled_records).set_index("configuration")
print("Pooled results from all withheld training predictions:")
print(minilm_c_pooled_summary.round(4).to_string())

# %% [markdown]
# Logistic Regression at `C=1` has the highest pooled F1 among these six settings.
# The table combines 5,986 predictions per setting; inspect fold-level F1 separately before averaging it.

# %%
# Reuse each fold's positions to score stored predictions; no classifier or encoder is fitted here.
minilm_c_fold_scores_by_configuration = {}
for minilm_c_key, minilm_c_predictions in minilm_c_oof_predictions.items():
    minilm_c_fold_scores = []
    for _, minilm_c_valid_positions in cv_folds:
        minilm_c_fold_labels = y_train.iloc[minilm_c_valid_positions]
        minilm_c_fold_predictions = minilm_c_predictions.iloc[minilm_c_valid_positions]
        minilm_c_fold_f1 = f1_score(
            minilm_c_fold_labels, minilm_c_fold_predictions, pos_label=1,
        )
        minilm_c_fold_scores.append(minilm_c_fold_f1)
    minilm_c_fold_scores_by_configuration[minilm_c_key] = minilm_c_fold_scores

# Preview one complete column; the dictionary retains all six matching sets of ten scores.
minilm_c_example_key = "Logistic Regression, C=1"
minilm_c_example_scores = pd.Series(
    minilm_c_fold_scores_by_configuration[minilm_c_example_key], index=range(1, 11), name="fold_f1",
)
print("Example fold F1:", minilm_c_example_key)
print(minilm_c_example_scores.round(6).to_string())

# %% [markdown]
# This column contains ten withheld-fold scores for one fixed configuration; the other five have matching fold positions.
# Compute the mean and descriptive spread of each stored score list, then align them with its pooled results.

# %%
# Preserve the original NumPy mean and ddof=0 standard deviation on the same ordered fold arrays.
minilm_c_fold_summary_records = []
for minilm_c_key, minilm_c_fold_scores in minilm_c_fold_scores_by_configuration.items():
    minilm_c_scores = np.array(minilm_c_fold_scores)
    minilm_c_fold_summary_records.append({
        "configuration": minilm_c_key,
        "mean_fold_f1": minilm_c_scores.mean(),
        "sd_fold_f1": minilm_c_scores.std(ddof=0),
    })
minilm_c_fold_summary = pd.DataFrame(minilm_c_fold_summary_records).set_index("configuration")
# Configuration names align the summaries; columns distinguish pooled metrics from fold summaries.
minilm_c_summary = minilm_c_pooled_summary.copy()
minilm_c_summary.insert(3, "mean_fold_f1", minilm_c_fold_summary["mean_fold_f1"])
minilm_c_summary.insert(4, "sd_fold_f1", minilm_c_fold_summary["sd_fold_f1"])
print(f"\nMiniLM training-only OOF comparison: {len(y_train)} rows; "
      f"{y_train.eq(1).sum()} recorded positives; {y_train.eq(0).sum()} recorded negatives.")
print(minilm_c_summary.round(4).to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Pooled metrics summarize all withheld predictions together; fold mean and spread summarize ten separate F1 scores.
# Matching configuration indices align the stored results without fitting again.
# Choosing a setting from these same folds remains exploratory.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# The training-only run on 2026-11-01 completed all 60 classifier fits without warnings in the existing environment.
# The repeated `C=1` mean fold F1 values reproduce the earlier Logistic Regression `0.773690` and LinearSVC `0.768419`.
# The frozen encoder was reused locally; no Transformer training, new download, or holdout evaluation was performed for this comparison.
#
# All rows below summarize the same 5,986 training examples: 2,549 recorded positives and 3,437 recorded negatives.
# LR means Logistic Regression; SVC means LinearSVC. Error counts and pooled metrics use one withheld prediction per example.
#
# | Classifier | C | Pooled recall | Pooled precision | Pooled F1 | Mean fold F1 | Missed positives | False alarms |
# | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
# | LR | 0.1 | 0.7273 | 0.8068 | 0.7650 | 0.7651 | 695 | 444 |
# | LR | 1.0 | 0.7466 | 0.8030 | 0.7737 | 0.7737 | 646 | 467 |
# | LR | 10.0 | 0.7399 | 0.7995 | 0.7685 | 0.7684 | 663 | 473 |
# | SVC | 0.1 | 0.7438 | 0.8010 | 0.7714 | 0.7712 | 653 | 471 |
# | SVC | 1.0 | 0.7383 | 0.8015 | 0.7686 | 0.7684 | 667 | 466 |
# | SVC | 10.0 | 0.7391 | 0.7916 | 0.7645 | 0.7643 | 665 | 496 |
#
# **Answer:** LinearSVC with `C=0.1` has 14 fewer missed positives and five more false alarms than its own `C=1` reference.
# Its pooled F1 rises about `0.0027`; these net count differences do not mean exactly 14 individual predictions were corrected.
# Neither alternative improves Logistic Regression's F1, and its existing `C=1` setting retains the highest observed F1 among these six configurations.
#
# > **⚠️ Watch Out — A small observed gain is not a validated improvement**
# >
# > Selecting a setting from these same folds introduces selection optimism; fold standard deviations are not significance tests.
# > These training results cannot be compared directly with the earlier 74.10% or 74.88% holdout recall as a measured gain.
#
# ### 🧠 What We Learned
# **Does this justify replacing the existing MiniLM configuration?** This limited comparison provides no F1-based reason to replace Logistic Regression at `C=1`.
# The small LinearSVC change remains exploratory; a claim about new events would require independent evaluation and an agreed error trade-off.
# Stop this bounded comparison here: preserve the original workflows and report all six settings without refitting a selected winner.
#
#
# ### 🤔 Question Yourself
# Does changing `C` from `1` to `0.1` strengthen or weaken regularization?
# Why does a larger `C` not guarantee a higher F1?
#
# <details><summary>💡 Show answer</summary>
#
# It strengthens regularization: smaller `C` penalizes large classifier coefficients more strongly.
# A larger `C` permits more flexibility, which can also fit patterns that do not transfer to withheld examples.
# Compare settings on the fixed training folds; choosing among those results does not establish an independent improvement.
#
# </details>
