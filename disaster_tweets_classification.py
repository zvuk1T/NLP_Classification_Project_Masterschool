# %% [markdown]
# # Disaster Tweets — Understanding the Dataset
#
# **Question:** What do the dataset structure, target distribution, and duplicate
# tweets tell us before modeling?
#
# **Context and source:** MasterSchool DA112 Sprint 3 asks us to classify tweets
# about real disasters. The supplied `DATA/train.csv` contains `text` and `target`.
# Label `1` means disaster; label `0` means non-disaster, according to the dataset.
# The labels are annotations, not independently verified facts about each event.
#
# The Project Map records the school requirements and dataset provenance limits.
# This is a text-only teaching copy; its exact transformation history is unknown.
#
# **Goal and approach:** Count what is present before deciding what to change.
# We expect a structural summary, class counts, and evidence of repeated or
# contradictory examples. These checks cannot establish model performance.
#
# Spock-authored implementation for Data's approved Slice 2A. No rows or labels
# are changed, and no preprocessing, splitting, or modeling happens here.

# %% [markdown]
# ## 1. Set up the execution context
#
# Open this repository folder in VS Code. Select `.venv/bin/python` as the
# Interactive Window kernel, start a fresh kernel, and run cells from top to bottom.
# Dependencies for this slice: `pandas==2.2.3` and `ipykernel`.
#
# `Path` builds a location without hardcoding a particular computer's username.
# A script has `__file__`; an interactive session may not. Without it, start the
# kernel in the repository root. The check below reports an incorrect location.
#
# Terminal alternative, from the repository root:
# `.venv/bin/python disaster_tweets_classification.py`

# %%
from pathlib import Path
import sys

import pandas as pd

if "__file__" in globals():
    project_root = Path(__file__).resolve().parent
else:
    project_root = Path.cwd()

data_path = project_root / "DATA" / "train.csv"
if not data_path.is_file():
    raise FileNotFoundError(
        f"Cannot find {data_path}. Start the Interactive Window in the repository root."
    )

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
print("pandas:", pd.__version__)
print("CSV:", data_path)

# %% [markdown]
# ## 2. Read the original rows
#
# `read_csv` loads the supplied file into a DataFrame: a table with named columns.
# Each row is one labeled observation. The first rows help us inspect the format;
# they are not a representative sample of all tweets.

# %%
tweets = pd.read_csv(data_path)
print("Rows and columns:", tweets.shape)
print("Column types:")
print(tweets.dtypes.to_string())
print("First five observations:")
print(tweets.head().to_string(index=False))

# %% [markdown]
# ## 3. Check missing values and valid labels
#
# Missing values and blank strings are different: spaces can occupy a non-missing
# field. The temporary `strip()` check detects blank text without changing `tweets`.
# `isin([0, 1])` checks membership in the two allowed labels.

# %%
missing_counts = tweets.isna().sum()
blank_text_count = tweets["text"].str.strip().eq("").sum()
invalid_label_count = (~tweets["target"].isin([0, 1])).sum()

print("Missing values by column:")
print(missing_counts.to_string())
print("Blank text rows:", blank_text_count)
print("Missing or invalid target rows:", invalid_label_count)

if missing_counts.any() or blank_text_count or invalid_label_count:
    raise ValueError("Unexpected missing, blank, or invalid values; inspect before continuing.")

# %% [markdown]
# ## 4. Compare the target classes
#
# `value_counts()` counts observations per label. Dividing by the total gives
# each class's share; sorting the index keeps labels in the order `0`, then `1`.
# Class balance gives context for later evaluation, but does not measure accuracy.

# %%
target_counts = tweets["target"].value_counts().sort_index()
target_summary = target_counts.to_frame(name="rows")
target_summary["percent"] = (target_counts / len(tweets) * 100).round(2)
print(target_summary.to_string())

# %% [markdown]
# ## 5. Count exact-text duplicates
#
# Exact equality preserves case, punctuation, and whitespace. Similar wording may
# therefore remain in different groups. We do not normalize text in this slice.
#
# A group repeated three times contributes three participating rows, but only two
# additional occurrences. `duplicated()` marks occurrences after the first;
# it does not delete them. Full-row duplication also requires the label to match.

# %%
text_counts = tweets["text"].value_counts()
repeated_text_counts = text_counts[text_counts > 1]
extra_text_rows = tweets["text"].duplicated().sum()
extra_identical_rows = tweets.duplicated().sum()

print("Unique exact texts:", len(text_counts))
print("Repeated exact-text groups:", len(repeated_text_counts))
print("Rows belonging to repeated groups:", repeated_text_counts.sum())
print("Additional exact-text occurrences:", extra_text_rows)
print("Additional identical (text, target) rows:", extra_identical_rows)

# %% [markdown]
# ## 6. Find identical texts with conflicting labels
#
# `groupby("text")` collects rows with identical original text. Selecting `target`
# and calling `nunique()` counts distinct labels within each group.
# With valid labels, a count of two means disagreement; it cannot tell us which
# label is correct. `isin()` then identifies all original rows in these groups.
#
# DataCamp bridge: [Initial exploratory analysis — Analyzing Marketing Campaigns
# with pandas](https://campus.datacamp.com/courses/analyzing-marketing-campaigns-with-pandas/pandas-1?ex=8).
# This verified lesson explains grouping and counting distinct values; our tweet
# analysis adapts the concept, not its marketing example or course code.

# %%
labels_per_text = tweets.groupby("text")["target"].nunique()
conflicting_texts = labels_per_text[labels_per_text > 1].index
conflicting_row_count = tweets["text"].isin(conflicting_texts).sum()

print("Exact-text groups with both labels:", len(conflicting_texts))
print("Rows belonging to conflicting groups:", conflicting_row_count)
print("Original rows still present:", len(tweets))

# %% [markdown]
# ## 7. Evidence, meaning, and the next question
#
# **Observed result:** 7,613 rows and two columns; no missing values, blank texts,
# or invalid labels. Class `0` has 4,342 rows (57.03%); class `1` has 3,271 (42.97%).
#
# There are 7,503 unique exact texts. The 69 repeated groups contain 179 rows,
# including 110 additional text occurrences. There are 92 additional identical
# `(text, target)` rows. Of the repeated groups, 18 contain both labels: 55 rows.
# These counts were reproduced during Slice 2A; rerun cells after any data change.
#
# **Interpretation:** Non-disaster labels are moderately more common. Repeated
# texts mean row count overstates the number of distinct exact-text examples.
# Conflicting labels show that identical input does not always have one recorded
# answer. We have not established the cause or identified the correct labels.
#
# **Transferable lesson:** A complete table can still contain inconsistent
# supervision. Counting missing values alone does not establish data quality.
#
# **Next question:** How should repeated and contradictory examples be handled
# before constructing an honest evaluation? Data must approve that policy first.
# Nothing has been removed, relabeled, normalized, split, vectorized, or trained.
#
# **Transfer check:** Why could random row splitting overstate performance here?
# Identical text could appear on both sides, making the holdout less independent.
# This prepared explanation does not assess Data's understanding: **UNASSESSED**.

# %% [markdown]
# ## 8. Work block 1 — Prepare the data for modeling
#
# **Main Question:** How can we prepare these tweets for a fair, reproducible
# model evaluation?
#
# **Context and instructions:** MasterSchool Sprint 3 requires text preparation
# and evaluation on unseen labeled data. The Project Map records those sources.
# Our duplicate policy is a project decision, not a prescribed school rule.
# Data approved the policy below on 2026-10-27 after reviewing its affected counts.
# This is Spock-authored implementation; the earlier raw-data analysis stays intact.
#
# **Goal:** Produce traceable training and holdout rows without shared text under
# our approved comparison rule. We expect counts, examples, and reproducibility
# checks here, not evidence of model quality.
#
# **Guiding Questions:** Which observations become equivalent after normalization?
# How do exclusions change the class distribution and the evaluation population?
# Can we reproduce the split while preserving labels and preventing text overlap?

# %% [markdown]
# ### Approved method — Define equivalence before removing rows
#
# Compare text after lowercasing, collapsing consecutive whitespace to one space,
# and trimming the ends. Use that same string as the later model input.
# This fixed rule learns nothing from the corpus, so applying it before splitting
# does not fit vocabulary or features on the holdout.
#
# Keep hashtags, mentions, URLs, punctuation, negation, and other words; only case
# and whitespace change. We avoid stemming, stop-word deletion, and speculative
# encoding repairs. Lowercasing loses emphasis and case distinctions, including
# those inside names or URLs; original text remains available for inspection.
#
# **Approved duplicate policy:** Exclude every row in a normalized-text group
# containing both labels. For a group with one consistent label, retain its first
# source row. Never infer a correct label or overwrite the supplied CSV.
#
# **Trade-off:** This defines an evaluation population with consistent recorded
# labels. Excluding ambiguous examples may make evaluation easier; a future score
# will not describe every original tweet or prove the retained labels are true.
# Near-duplicates and reports about the same event can still cross partitions.
#
# Work on a copy and retain source row indices. The prepared string is also the
# duplicate key, so normalization collisions cannot escape the duplicate check.
# `\s+` means one or more whitespace characters, including line breaks.

# %%
from hashlib import sha256

source_sha256 = sha256(data_path.read_bytes()).hexdigest()
prepared = tweets.copy()
prepared["prepared_text"] = (
    prepared["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
)

labels_per_prepared_text = prepared.groupby("prepared_text")["target"].nunique()
original_forms_per_key = prepared.groupby("prepared_text")["text"].nunique()
collision_keys = original_forms_per_key[original_forms_per_key > 1].index
conflicting_keys = labels_per_prepared_text[labels_per_prepared_text > 1].index

print("Normalized keys merging distinct original texts:", len(collision_keys))
print("Original/prepared example:")
print(prepared[["text", "prepared_text", "target"]].head(1).to_string())
print("All normalization collisions (source row indices retained):")
print(prepared.loc[prepared["prepared_text"].isin(collision_keys)].to_string())

# %% [markdown]
# ### Apply the approved exclusions and account for every row
#
# A conflict concerns the entire normalized group, not just its minority label.
# Exclude those groups first; then identify redundant occurrences in consistent
# groups. Keeping the first occurrence follows the original CSV order.
# The accounting table separates these two reasons for exclusion.

# %%
conflict_mask = prepared["prepared_text"].isin(conflicting_keys)
conflicting_rows = prepared.loc[conflict_mask]
consistent_rows = prepared.loc[~conflict_mask]
repeat_mask = consistent_rows.duplicated(subset="prepared_text", keep="first")
redundant_rows = consistent_rows.loc[repeat_mask]
model_tweets = consistent_rows.loc[~repeat_mask].copy()

preparation_counts = pd.DataFrame({
    "original": tweets["target"].value_counts(),
    "conflicts excluded": conflicting_rows["target"].value_counts(),
    "repeats excluded": redundant_rows["target"].value_counts(),
    "prepared": model_tweets["target"].value_counts(),
}).T.reindex(columns=[0, 1], fill_value=0).fillna(0).astype(int)
preparation_counts["total"] = preparation_counts.sum(axis=1)
print("Rows by stage and target label:")
print(preparation_counts.to_string())
print("Conflicting normalized groups:", len(conflicting_keys))

# %% [markdown]
# ### Separate training and holdout data
#
# **Approved design:** Reserve 20% as holdout, stratify by `target`, and fix
# `random_state=42`. Stratification keeps class proportions close despite integer
# rounding; the seed reproduces row selection for this input order and environment.
#
# Split the complete table, then select text and labels from each partition.
# This keeps original text, prepared text, labels, and source indices aligned.
# Each approved duplicate group now has at most one row, so ordinary stratified
# splitting cannot distribute that group across both partitions.
#
# Work block 1 additionally uses `scikit-learn==1.5.2` in the existing `.venv`.
# Existing dependency pins constrain installation; the full requirements file is
# not needed for this block. The code reports the versions actually executing.
#
# DataCamp bridge: [Measuring model performance — Supervised Learning with
# scikit-learn](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/classification-1?ex=7).
# Its train/test explanation covers stratification and reproducible random seeds.
# We adapt that splitting concept; no lesson model or score is reused.
#
# No vocabulary, TF-IDF weights, or model is fitted here. The next approved block
# must fit learned transformations on training data only and apply the same
# deterministic preparation to future inputs.

# %%
import sklearn
from sklearn.model_selection import train_test_split

print("scikit-learn:", sklearn.__version__)
train_rows, holdout_rows = train_test_split(
    model_tweets, test_size=0.20, random_state=42, stratify=model_tweets["target"]
)
X_train = train_rows["prepared_text"]
y_train = train_rows["target"]
X_holdout = holdout_rows["prepared_text"]
y_holdout = holdout_rows["target"]

for name, frame in [("original", tweets), ("prepared", model_tweets),
                    ("training", train_rows), ("holdout", holdout_rows)]:
    counts = frame["target"].value_counts().sort_index()
    summary = counts.to_frame(name="rows")
    summary["percent"] = (counts / len(frame) * 100).round(2)
    print(f"\n{name}: {len(frame)} rows")
    print(summary.to_string())

# %% [markdown]
# ### Verify preservation, alignment, separation, and repeatability
#
# Assertions stop execution if an expected relationship fails. Check that every
# source row is accounted for, retained labels are unchanged, and both partitions
# cover the prepared data exactly once. Check original and prepared text overlap.
#
# Repeat only the split to verify identical membership and ordering. Compare the
# raw table with a fresh CSV read and check the file hash for accidental changes.
# These checks establish preparation integrity, not model generalization or
# independence between different texts about the same event.

# %%
assert len(tweets) == len(conflicting_rows) + len(redundant_rows) + len(model_tweets)
assert model_tweets["prepared_text"].is_unique
assert not model_tweets["prepared_text"].isna().any()
assert not model_tweets["prepared_text"].eq("").any()
assert model_tweets["target"].equals(tweets.loc[model_tweets.index, "target"])
assert model_tweets["text"].equals(tweets.loc[model_tweets.index, "text"])
assert X_train.index.equals(y_train.index)
assert X_holdout.index.equals(y_holdout.index)
assert y_train.equals(tweets.loc[X_train.index, "target"])
assert y_holdout.equals(tweets.loc[X_holdout.index, "target"])
assert set(train_rows.index).isdisjoint(holdout_rows.index)
assert len(train_rows) + len(holdout_rows) == len(model_tweets)
assert set(train_rows.index) | set(holdout_rows.index) == set(model_tweets.index)

original_overlap = set(train_rows["text"]) & set(holdout_rows["text"])
prepared_overlap = set(X_train) & set(X_holdout)
assert not original_overlap and not prepared_overlap
for partition in [train_rows, holdout_rows]:
    proportions = partition["target"].value_counts(normalize=True).sort_index()
    prepared_proportions = model_tweets["target"].value_counts(normalize=True).sort_index()
    assert (proportions - prepared_proportions).abs().le(1 / len(partition)).all()

repeat_train, repeat_holdout = train_test_split(
    model_tweets, test_size=0.20, random_state=42, stratify=model_tweets["target"]
)
assert train_rows.equals(repeat_train) and holdout_rows.equals(repeat_holdout)
assert tweets.equals(pd.read_csv(data_path))
assert sha256(data_path.read_bytes()).hexdigest() == source_sha256

print("Original/prepared text overlap:", len(original_overlap), len(prepared_overlap))
print("PASS: row accounting, labels, coverage, class proportions, and reproducible split.")
print("PASS: original DataFrame and source CSV preserved.")
print("CSV SHA-256:", source_sha256)

# %% [markdown]
# ### Evidence & Interpretation — What changed, and what did not?
#
# **Observed on 2026-10-27:** Normalization merged two pairs of distinct original
# strings. Rows 3340/7604 differ in whitespace; row 6119 adds a case variant to
# an existing conflicting group. Thus 18 conflicting groups contain 56 rows,
# versus 55 under exact comparison. These are zero-based source row indices.
#
# Excluding those 56 rows and 74 redundant consistent occurrences leaves 7,483
# distinct prepared texts. Original rows and labels remain intact in `tweets`
# and the CSV. All checks above passed with the approved split settings.
#
# | Population | Rows | Label 0 | Label 1 | Label 1 share |
# | --- | ---: | ---: | ---: | ---: |
# | Original | 7,613 | 4,342 | 3,271 | 42.97% |
# | Prepared | 7,483 | 4,297 | 3,186 | 42.58% |
# | Training | 5,986 | 3,437 | 2,549 | 42.58% |
# | Holdout | 1,497 | 860 | 637 | 42.55% |
#
# **Interpretation:** The prepared class mix changed slightly, and stratification
# preserved it closely in both partitions. Original and prepared text overlap are
# both zero; repeated splitting gives identical rows and ordering.
# These facts support reproducible separation under our rule, not event-level
# independence, correct annotations, or future model performance.
#
# ### Learning → Better Questions
#
# **Learning:** Preparation defines both the model's inputs and the population
# we can evaluate. Checking collisions after normalization prevents superficially
# different strings from bypassing the duplicate policy. Excluding conflicts
# simplifies evaluation without resolving why those annotations disagree.
#
# **Next Main Question:** How well does a TF-IDF + Logistic Regression baseline
# classify the held-out tweets, and where does it fail? That requires the next
# approved work block; no vectorizer or model has been fitted here.
#
# **Transfer check:** Does stratification prevent duplicate leakage?
# No. It controls label proportions, not shared text. Our approved grouping and
# exclusions address text overlap before splitting; related events may remain.
# This written explanation does not establish Data's Learning PASS: **UNASSESSED**.
