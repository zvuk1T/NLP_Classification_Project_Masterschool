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
