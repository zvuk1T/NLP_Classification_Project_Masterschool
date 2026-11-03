# %% [markdown]
# # Disaster Tweets — Frozen Transformer Features
#
# **❓ What is the Question?** Does changing the text representation improve disaster classification when the classifier and evaluation rows stay fixed?
# **🎯 Goal & Approach — WHY:** Compare MiniLM embeddings with the existing TF-IDF results, using Logistic Regression and LinearSVC.
# This assistant-authored extension complements `disaster_tweets_classification.py`; that file remains the source of the original baseline.
#
# **Reading route:** Same observations → frozen encoder → training-only CV → retrospective holdout → insights → ethics → visual comparison.
# **Execution evidence — 2026-10-29:** All eleven code cells ran in the project `.venv`, including 20 CV fits and two final classifier fits.
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
# **❓ What is the Question?** Are we comparing representations on the same observations?
# **🎯 Goal & Approach — WHY:** Reuse the main analysis's normalization, retention policy, split and ten training folds.
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
# <details><summary>💡 Explain this code</summary>
#
# **📖 Overview — WHAT**
#
# To isolate the representation change, reproduce the original normalization, retention policy, train/holdout split, and training-fold membership.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. The file path resolves beside the script or from the interactive working folder; `read_csv()` creates the source-row index.
# 2. The copy retains original text; normalized strings identify conflicting groups, then `drop_duplicates()` retains the first consistent occurrence.
# 3. Splitting whole rows keeps text and labels together; selecting columns preserves their matching source indices and order.
# 4. `list(cv_splitter.split(...))` stores ten pairs of positional fit/validation arrays, reused for both classifiers.
#
# ```python
# cv_folds = list(cv_splitter.split(X_train, y_train))
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# Fold positions are not CSV source indices; matching membership controls observations but does not establish independent events or accurate annotations.
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
# **Better Question:** Would the comparison hold across genuinely separate disaster events?
# **Current answer: not established.** Matching rows cannot answer that question without event relationships and fresh evaluation evidence.

# %% [markdown]
# ## 2. Turn each tweet into a frozen representation
#
# **❓ What is the Question?** What does the pretrained encoder supply, and what will our classifiers still learn?
# **🎯 Goal & Approach — WHY:** Produce one 384-number vector per tweet using `sentence-transformers/all-MiniLM-L6-v2`.
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

# Read only cached tokenizer/configuration and safetensors weights; no repository Python code is used.
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
encoder = AutoModel.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
).to("cpu")
encoder.eval()  # Disable training-time dropout, so inference uses the fixed model consistently.
encoder.requires_grad_(False)  # No encoder parameter will receive a training gradient.
print("PyTorch:", torch.__version__, "Transformers:", transformers.__version__)
print("Encoder:", model_id, "Revision:", model_revision)
print("Model snapshot:", model_path)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# **📖 Overview — WHAT**
#
# To compare transferred features without new Transformer training, load the fixed MiniLM encoder and matching tokenizer from existing local files.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. The model ID, revision, and file list identify the approved snapshot; local lookup uses the project cache and rejects missing snapshots.
# 2. A separate file check rejects incomplete copies; unused backend formats in the model repository are not required.
# 3. The tokenizer and encoder load locally with remote code disabled; `.to("cpu")` selects the execution device.
# 4. `eval()` disables training-time dropout; `requires_grad_(False)` prevents parameter gradients, keeping the encoder frozen for later inference.
#
# ```python
# encoder.eval()
# encoder.requires_grad_(False)
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# This loads externally learned features, not disaster predictions; freezing weights does not establish whether pretraining overlapped these tweets.
#
# </details>

# %%
# Keep one feature matrix per partition: rows follow input tweets; columns are 384 embedding dimensions.
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
            # Exclude padding before averaging; special tokens still contribute.
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
# <details><summary>💡 Explain this code</summary>
#
# **📖 Overview — WHAT**
#
# To compare transferred features with TF-IDF, convert each prepared tweet into 384 fixed numerical features. `embedding_matrices` keeps training and holdout separate.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. `texts.tolist()` retains input order; batches contain at most 32 tweets to limit working memory.
#    `full_token_ids` holds untruncated ID lists; their lengths feed the check for inputs exceeding 256 positions.
# 2. `tokenizer(...)` converts word pieces into vocabulary IDs, adds special tokens, pads each batch, and caps inputs at 256 positions.
#    `return_tensors="pt"` returns PyTorch tensors; `attention_mask` contains 1 for non-padding positions and 0 for padding.
# 3. `encoder(**encoded).last_hidden_state` passes the dictionary entries as named inputs and returns contextual token vectors.
#    For B tweets and T padded token positions, their shape is `(B, T, 384)`; these are not disaster predictions.
#    `torch.inference_mode()` disables gradient tracking; the encoder was already frozen and put in evaluation mode above.
# 4. `unsqueeze(-1)` gives the mask shape `(B, T, 1)`; `.to(token_vectors.dtype)` matches its numeric type to the vectors.
#    In `(token_vectors * token_mask).sum(dim=1)`, each flag applies across 384 features, then summation removes the token axis.
#    `vector_sums` therefore has shape `(B, 384)`, with padding contributing zero.
# 5. `token_mask.sum(dim=1).clamp(min=1)` produces a `(B, 1)` non-padding count, guarded against a zero denominator.
#    `vector_sums / token_counts` averages each tweet's remaining token vectors; special tokens remain included.
# 6. `functional.normalize(tweet_vectors, p=2, dim=1)` scales along the 384-feature axis using each row's Euclidean length.
#    A tiny denominator floor guards against division by zero; these nonzero tweet vectors become unit length.
#    Their entries can still be negative and are not probabilities.
# 7. `.cpu().numpy()` converts each batch to a NumPy array; `np.concatenate(batch_embeddings, axis=0)` stacks batches in their original order.
#    Each partition's matrix has shape `(len(texts), 384)`; positional row order, not a pandas index, preserves alignment with its labels.
#
# ```python
# vector_sums = (token_vectors * token_mask).sum(dim=1)
# token_counts = token_mask.sum(dim=1).clamp(min=1)
# tweet_vectors = vector_sums / token_counts
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# These fixed per-tweet operations learn no vocabulary, corpus statistics, or label relationships here; only the downstream classifiers fit our labels.
# Masking excludes padding, not unimportant words; pooling and possible truncation can still lose information relevant to disaster classification.
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
# **Better Question:** Do the transferred features help either classifier on the same validation observations?
# **Answer:** Both classifiers have higher mean CV F1 than their historical TF-IDF references, as shown in the next unit.
# Successful encoding alone would not establish that improvement; the labeled validation scores supply the evidence.

# %% [markdown]
# ## 3. Compare the same classifiers in the same ten folds
#
# **❓ What is the Question?** Does MiniLM improve training-only CV F1 for Logistic Regression or LinearSVC?
# **🎯 Goal & Approach — WHY:** Fit fresh classifiers on nine folds and score the remaining fold, repeating all ten partitions.
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

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# **📖 Overview — WHAT**
#
# To compare classifiers on the same frozen features, fit each template separately within every stored training fold.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. `clone()` copies estimator settings without fitted parameters; each fold fits only its designated training positions.
# 2. NumPy indexing selects embedding rows, while `.iloc` selects labels by the same positions; prediction uses the withheld positions.
# 3. `f1_score(..., pos_label=1, zero_division=0)` scores recorded disasters and returns zero for an otherwise undefined F1.
# 4. Records hold one classifier/fold result each; `pivot()` arranges scores by fold, while means and `std(ddof=0)` summarize ten scores.
#
# ```python
# fold_classifier = clone(classifier)
# fold_classifier.fit(
#     train_embeddings[fit_positions],
#     y_train.iloc[fit_positions],
# )
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# Only classifiers learn here; fixed per-tweet features can precede CV, but overlapping training subsets make fold spread unsuitable as a confidence interval.
#
# </details>

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

# Compare each new mean with that same classifier's two historical TF-IDF means, never with holdout F1.
for _, current in embedding_summary.iterrows():
    references = tfidf_reference.loc[tfidf_reference["classifier"] == current["classifier"]]
    for _, reference in references.iterrows():
        difference = current["mean_f1"] - reference["mean_f1"]
        print(f"{current['classifier']}: MiniLM minus {reference['representation']} mean F1 = {difference:+.4f} (approximate)")

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# **📖 Overview — WHAT**
#
# To compare representations while preserving provenance, combine six historical TF-IDF summaries with the two MiniLM summaries from this run.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. Tuple records become a DataFrame with named metric columns; `evidence` identifies historical values rounded to four decimals.
# 2. `pd.concat(..., ignore_index=True)` stacks rows and assigns a new display index, not source-tweet identifiers.
# 3. For each MiniLM classifier, filtering selects its matching historical classifier rows; subtraction compares CV means, never holdout scores.
#
# ```python
# cv_comparison = pd.concat(
#     [tfidf_reference, embedding_summary],
#     ignore_index=True,
# )
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# Rounded references make differences approximate; concatenation neither reproduces the earlier 60 fits nor supplies paired historical fold scores.
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
# **📖 Overview — WHAT**
#
# To inspect variation and overall ranking separately, visualize existing fold scores and configuration summaries without fitting models.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. `px.line()` uses one point per classifier/fold; markers show measured F1 and lines connect fold identifiers.
# 2. `px.scatter()` positions each configuration at its mean F1; `error_x="std_f1"` adds its fold standard deviation.
# 3. Symbols and hover fields retain evidence provenance; replacing label separators with `<br>` wraps display text without changing scores.
#
# **💡 Why this matters — WHY & LIMITS**
#
# Both axes use F1 on a zero-to-one scale; error bars describe fold variation, not significance or independent uncertainty bounds.
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
# **Better Question:** Would a measured average gain also reduce consequential errors on other events?
# **Current answer: not established.** Ten overlapping training folds describe this experiment, not independent populations or operational consequences.

# %% [markdown]
# ## 4. Report the two predeclared models on the existing holdout
#
# **❓ What is the Question?** What do the two frozen-feature workflows predict on the unchanged 1,497 holdout rows?
# **🎯 Goal & Approach — WHY:** Refit each classifier on all training embeddings, then evaluate the holdout once with fixed settings.
# Both classifiers are evaluated regardless of their CV ranking; this adds two final classifier fits without choosing a CV winner.
#
# **🔎 Guiding Question:** Which comparison has a matching historical holdout baseline?
# Only TF-IDF + Logistic Regression has an existing holdout result. LinearSVC's historical figures are CV results and belong in the preceding comparison.

# %%
# Refit both fixed classifier templates on all training embeddings; the known holdout remains retrospective evidence.
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

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# **📖 Overview — WHAT**
#
# To document both predeclared workflows, fit fresh classifiers on all training vectors and evaluate them on the familiar holdout.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. `clone()` preserves templates; each fitted classifier predicts a position-aligned array stored under its name in `holdout_predictions`.
# 2. Metrics compare each array with the same `y_holdout`; matrices use recorded-label rows and prediction columns in 0/1 order.
# 3. Both heatmaps display counts on the same scale; they create no additional evaluation observations.
# 4. Historical TN, FP, FN, and TP counts reconstruct baseline metrics; the summary joins that documented row with the current retrospective records.
#
# ```python
# fitted_classifier.fit(train_embeddings, y_train)
# predictions = fitted_classifier.predict(holdout_embeddings)
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# The baseline is not retrained, and neither current classifier is chosen by its holdout score; this inspected holdout is not an independent final test.
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
# **📖 Overview — WHAT**
#
# To inspect concrete disagreements, copy original holdout text and labels, then attach each classifier’s existing predictions.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. Array assignment uses row positions; the review table retains the original source indices.
# 2. `!=` marks every mismatch for that classifier; `.loc` selects those rows and the relevant columns.
# 3. `head(3)` shows the first three mismatches in holdout order, potentially mixing false positives and false negatives.
#
# ```python
# error_mask = (
#     holdout_review[classifier_name]
#     != holdout_review["target"]
# )
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# This differs from the baseline’s separately filtered, source-index-sorted examples; three selected errors establish neither prevalence nor internal model causes.
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

# %% [markdown]
# ## 5. Insights, limits and a justified next question
#
# **❓ What is the Question?** What does this experiment establish?
# **🎯 Goal & Approach — WHY:** Connect measured differences to a proportionate conclusion and identify the evidence still needed.
# **✅ Evidence, Answer & Limits:** MiniLM features improved both classifiers' mean CV F1 on the unchanged training partitions.
# The matched Logistic Regression holdout comparison also improved F1, with fewer false negatives and more false positives.
# These findings support this specific feature-transfer experiment, not universal Transformer superiority or a new final-test claim.
#
# **🧠 What We Learned:** External pretraining changes the available information, while frozen features keep our fold-specific learning confined to the classifiers.
# Fold scores are Data; matching labels, rows and settings provide Information. The measured differences support Knowledge about these particular workflows.
# Wisdom requires weighing its consistency, error types, computation and evidence limits before recommending a more complex approach.
#
# **Better Question:** Would any observed improvement survive evaluation on separately collected, independently annotated disaster reports?
# **Current answer: not established.** This dataset and retrospective holdout cannot supply that new evidence.

# %% [markdown]
# ## 6. Ethical interpretation and responsible use
#
# **❓ What is the Question?** What may we responsibly claim from the observed improvements?
# **🎯 Goal & Approach — WHY:** Connect measured error changes to their possible consequences without inventing real-world benefits.
#
# **✅ Evidence, Answer & Limits:** MiniLM improves the measured average CV F1 under this project's preparation and evaluation choices.
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
# **🧠 What We Learned**
# **Better Question:** How should a proposed use case weigh missed reports against false alarms?
# **Current answer: not established.** That decision needs input from affected users and evidence about consequences, beyond the metrics measured here.
# **⚖️ Ethical Takeaway:** Report the error trade-off, annotation uncertainty and evaluation limits together; neither representation establishes emergency-response readiness.

# %% [markdown]
# ## 7. Visual comparison — TF-IDF and Hugging Face features
#
# **❓ What is the Question?** How large are the measured differences, and which errors change when we replace TF-IDF with MiniLM?
# **🎯 Goal & Approach — WHY:** Finish with three complementary Plotly views, derived from the result tables and predictions already computed above.
# **🔎 Guiding Question:** Which comparisons keep the classifier fixed, and where is a matching baseline unavailable?
#
# **Comparison boundary:** CV includes both classifiers against their two historical TF-IDF configurations, using the same training rows and folds.
# Holdout views compare Logistic Regression only, because LinearSVC has no historical holdout baseline for isolating the representation change.
# MiniLM + LinearSVC's holdout result remains in Section 4; these figures do not select a model or refit anything.
#
# ### A. Change in training-only mean F1
# **🎯 Goal & Approach — WHY:** A difference plot makes the size of each change easier to read than eight nearby averages.
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
# **📖 Overview — WHAT**
#
# To make representation differences visible, derive a table of matched-classifier changes from existing CV summaries.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. `iterrows()` visits MiniLM summaries; a Boolean filter selects the same classifier’s two historical TF-IDF configurations.
# 2. Each record retains both means and their difference; no fit or holdout score enters the table.
# 3. Signed strings format displayed labels only; grouped bars use the unrounded differences stored in `f1_change`.
#
# **💡 Why this matters — WHY & LIMITS**
#
# These are absolute F1 differences, not percentage growth or paired per-fold gains; rounding in historical references limits precision.
#
# </details>

# %% [markdown]
# **✅ Evidence, Answer & Limits:** All four mean differences are positive: approximately `+0.0231` / `+0.0444` for Logistic Regression and `+0.0130` / `+0.0155` for LinearSVC.
# The larger bigram comparison partly reflects its lower TF-IDF reference; it does not identify the best achievable model.
# Historical means alone cannot establish paired improvements in every fold or statistical significance.
#
# ### B. Holdout metrics with Logistic Regression fixed
# **🎯 Goal & Approach — WHY:** F1 summarizes a trade-off, so show precision, recall and accuracy beside it rather than treating one score as the whole result.
# **💻 Code / Method — HOW:** Display the earlier TF-IDF and current MiniLM scores side by side on the same zero-to-one scale.
# This is retrospective evidence on the same 1,497 observations; class-1 F1, precision and recall concern the recorded disaster label.

# %%
# Select the two Logistic Regression workflows and retain evidence labels for the retrospective comparison.
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
# <details><summary>💡 Explain this code</summary>
#
# **📖 Overview — WHAT**
#
# To compare four holdout metrics with classifier choice fixed, select the two Logistic Regression workflows and reshape their existing scores.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. `isin(representation_names)` tests dictionary-key membership; `map()` assigns a shorter representation label to each selected row.
# 2. `melt()` turns four metric columns into rows, retaining representation and evidence as identifiers: one score per representation/metric pair.
# 3. Display strings round labels only; grouped horizontal bars share a zero-to-one axis and preserve provenance in hover details.
#
# ```python
# matched_holdout = holdout_summary.loc[
#     holdout_summary["configuration"].isin(
#         representation_names
#     )
# ].copy()
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# Reshaping does not average or recalculate metrics; the familiar holdout supports a retrospective comparison, not a new unbiased final evaluation.
#
# </details>

# %% [markdown]
# **✅ Evidence, Answer & Limits:** F1 rises from `0.7308` to `0.7650`, recall from `0.6735` to `0.7410`, and accuracy from `0.7889` to `0.8063`.
# Precision falls from `0.7989` to `0.7906`; the representation change does not improve every metric.
# The bars reuse existing results; reshaping with [pandas `melt`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html) does not recalculate or average scores.
#
# ### C. Translate the trade-off into error counts
# **🎯 Goal & Approach — WHY:** Error counts make the measured trade-off concrete without assigning an unsupported cost to either error type.
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
# **📖 Overview — WHAT**
#
# To express the representation trade-off as counts, compare MiniLM Logistic Regression errors with the documented baseline errors.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. `confusion_matrix(..., labels=[0, 1])` fixes recorded-label rows and prediction columns; two off-diagonal entries supply FN and FP counts.
# 2. The two-row table retains earlier/current counts; column subtraction gives signed net changes for each error type.
# 3. The largest absolute change sets symmetric axis limits, so increases and decreases share a comparable scale.
#
# ```python
# error_changes["count_change"] = (
#     error_changes["current_minilm"]
#     - error_changes["earlier_tfidf"]
# )
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# Net changes do not identify individual corrected tweets or error costs; the two types also have different recorded-label denominators.
#
# </details>

# %% [markdown]
# **✅ Evidence, Answer & Limits:** Missed disaster labels decrease from `208` to `165` (`−43`); false positives increase from `108` to `125` (`+17`).
# These are net count differences, not proof that exactly 43 particular tweets were corrected; some predictions may change in opposite directions.
# The two error types have different label-group denominators, so these bars show counts rather than comparable error rates.
#
# **🧠 What We Learned**
# **Better Question:** Does this error trade-off justify using the more complex representation in a defined real-world setting?
# **Current answer: not established.** The charts support a measured educational comparison; deciding usefulness also requires fresh evaluation and evidence about error costs.
# Read the visual gains together with the ethical limits in Section 6, rather than declaring an overall winner from one score.

# %% [markdown]
# ## 8. Beyond the baseline — adjust the classifiers, keep MiniLM fixed
#
# **❓ What is the Question?** Can a small change to classifier regularization improve the frozen-MiniLM results without Transformer fine-tuning?
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
minilm_c_records = []

for minilm_c_name, minilm_c_template in classifier_options:
    for minilm_c_value in minilm_c_values:
        # Clone before changing C, preserving the original C=1 templates and their remaining settings.
        minilm_c_candidate = clone(minilm_c_template).set_params(C=minilm_c_value)
        minilm_c_key = f"{minilm_c_name}, C={minilm_c_value:g}"

        # Each fold fits a fresh classifier; returned labels follow the original training-row order.
        # Fixed per-tweet embeddings need no refitting inside folds; default predict rules remain unchanged.
        minilm_c_oof_predictions[minilm_c_key] = cross_val_predict(
            minilm_c_candidate, train_embeddings, y_train, cv=cv_folds, method="predict", n_jobs=1,
        )
        minilm_c_predictions = minilm_c_oof_predictions[minilm_c_key]

        # Rows are recorded labels, columns are predictions; [0, 1] fixes the confusion-matrix orientation.
        # Flattening yields correct negatives, false alarms, missed positives, and correct positives.
        minilm_c_tn, minilm_c_fp, minilm_c_fn, minilm_c_tp = confusion_matrix(
            y_train, minilm_c_predictions, labels=[0, 1],
        ).ravel()

        # Summarize stored predictions within each known fold; this loop performs no additional fitting.
        # iloc uses positional memberships, while the Series retain original source indices.
        minilm_c_fold_scores = []
        for _, minilm_c_valid_positions in cv_folds:
            minilm_c_fold_scores.append(f1_score(
                y_train.iloc[minilm_c_valid_positions],
                minilm_c_predictions.iloc[minilm_c_valid_positions], pos_label=1,
            ))
        minilm_c_scores = np.array(minilm_c_fold_scores)

        # Recall uses recorded positives; precision uses predicted positives. Fold SD is descriptive, not a confidence interval.
        minilm_c_records.append({
            "configuration": minilm_c_key,
            "pooled_recall": recall_score(y_train, minilm_c_predictions, pos_label=1),
            "pooled_precision": precision_score(y_train, minilm_c_predictions, pos_label=1),
            "pooled_f1": f1_score(y_train, minilm_c_predictions, pos_label=1),
            "mean_fold_f1": minilm_c_scores.mean(),
            "sd_fold_f1": minilm_c_scores.std(ddof=0),
            "missed_positives": minilm_c_fn,
            "false_alarms": minilm_c_fp,
        })
        print(f"Completed ten training-only folds: {minilm_c_key}")

# One row per configuration keeps every tested setting visible; rounding changes display only.
minilm_c_summary = pd.DataFrame(minilm_c_records).set_index("configuration")
print(f"\nMiniLM training-only OOF comparison: {len(y_train)} rows; "
      f"{y_train.eq(1).sum()} recorded positives; {y_train.eq(0).sum()} recorded negatives.")
print(minilm_c_summary.round(4).to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# **📖 Overview — WHAT**
#
# To test regularization while keeping MiniLM fixed, collect withheld predictions for two classifiers at three `C` settings.
#
# **🔢 Step-by-step breakdown — HOW**
#
# 1. `clone(...).set_params(C=...)` changes only a copied template; the original classifier objects remain untouched.
# 2. `cross_val_predict()` fits a fresh classifier per stored fold and returns labels in input order; embeddings are reused without refitting.
# 3. Matrix flattening extracts TN, FP, FN, TP; positional `.iloc` selection scores stored predictions separately within each validation fold.
# 4. Each summary row contains pooled metrics and errors alongside the mean and standard deviation of ten individual fold F1 scores.
#
# ```python
# minilm_c_candidate = clone(
#     minilm_c_template
# ).set_params(C=minilm_c_value)
# ```
#
# **💡 Why this matters — WHY & LIMITS**
#
# `C` changes the coefficient penalty, not the number of training rows; the extra scoring loop performs no fitting, and selection remains exploratory.
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
