# %% [markdown]
# # Disaster Tweets — From Recorded Rows to an Evaluation Population
#
# **Project Question:** Can tweet text help distinguish disaster from non-disaster
# examples, and what would make evidence of that ability trustworthy?
#
# ### 📋 Context, Source & Task
# MasterSchool DA112 Sprint 3 asks us to classify disaster tweets. The supplied
# `DATA/train.csv` records text and a target: `1` for disaster, `0` for non-disaster.
# These are recorded annotations, not independently verified facts about events.
# The Project Map documents this teaching copy and its unknown transformation history.
#
# This assistant-authored analysis retrospectively explains the completed dataset
# investigation and preparation. The questions organize its reasoning; they do not
# claim to reconstruct a learner's past thoughts or demonstrate understanding.
#
# The local Sprint 2 `guided.ipynb` supplies the school reference for sequential
# text preparation and keeping text paired with labels. We adapt those principles
# to tweets; the limited normalization and retention policy below are project choices.
# DataCamp links support particular operations, using original explanations here.

# %% [markdown]
# ## 1. Understand the observations and labels
#
# ### ❓ Main Question
# What do the dataset structure and target distribution tell us before modeling?
#
# **🎯 Goal & Approach — What & Why:** Establish what one observation contains
# and whether its required fields are usable. Count labels to describe the starting
# population; these proportions provide context, not a measure of predictive ability.
#
# **🔎 Guiding Questions:** Are text or labels missing or invalid? How many
# observations belong to each recorded class?
#
# **✅ Notes for Students:** Open this repository in VS Code, select its existing
# `.venv/bin/python` kernel, restart it, and run cells from top to bottom.
# This file uses pandas and scikit-learn; ipykernel supports the Interactive Window.
# A terminal alternative from the root is `.venv/bin/python disaster_tweets_classification.py`.

# %%
from pathlib import Path
import sys

import pandas as pd

# A script resolves data beside this file; an interactive session uses its folder.
# project_root and data_path are filesystem locations, not data tables.
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

# Read one labeled observation per row; its index is the zero-based source position.
# Keep tweets unchanged so later subsets can be traced back to these original rows.
tweets = pd.read_csv(data_path)
print("Rows and columns:", tweets.shape)
print("Column types:")
print(tweets.dtypes.to_string())
print("First five observations (source indices shown; not a representative sample):")
print(tweets.head().to_string())

# %%
# isna() flags missing cells; summing down each column counts True as 1, False as 0.
# missing_counts is a Series indexed by column name, with one missing-value total each.
missing_counts = tweets.isna().sum()

# strip() temporarily trims text ends; eq("") flags rows containing only whitespace.
# Summing these flags counts blank texts without changing the source column.
blank_text_count = tweets["text"].str.strip().eq("").sum()

# isin checks membership in allowed labels. ~ reverses each flag: True now means invalid.
# This is a row count, including missing labels, not a count of distinct bad label values.
invalid_label_count = (~tweets["target"].isin([0, 1])).sum()
print("Missing values by column:")
print(missing_counts.to_string())
print("Blank text rows:", blank_text_count)
print("Missing or invalid target rows:", invalid_label_count)

# Stop if required inputs are unusable; subsequent grouping assumes valid text and labels.
if missing_counts.any() or blank_text_count or invalid_label_count:
    raise ValueError("Unexpected missing, blank, or invalid values; inspect before continuing.")

# Count observations per label; sort_index orders the label keys as 0, then 1.
# target_summary has one row per class, with the original row count as its denominator.
target_counts = tweets["target"].value_counts().sort_index()
target_summary = target_counts.to_frame(name="rows")
target_summary["percent"] = (target_counts / len(tweets) * 100).round(2)
print("Original label distribution:")
print(target_summary.to_string())

# %% [markdown]
# ### ✅ Evidence & Interpretation
# The supplied table has 7,613 rows and two columns, with no missing values,
# blank texts, or invalid labels. Class `0` contains 4,342 rows (57.03%);
# class `1` contains 3,271 (42.97%). Non-disaster annotations are moderately more common.
#
# ### 🧠 Learning → Better Questions
# Usable fields and known class proportions establish a starting population.
# They do not establish that each row adds a distinct example or a reliable answer.
# **What changes when we count texts rather than rows, and inspect label agreement?**

# %% [markdown]
# ## 2. Distinguish repeated texts from conflicting annotations
#
# ### ❓ Main Question
# How much repetition and label disagreement does exact text equality reveal?
#
# **🎯 Goal & Approach — What & Why:** Group identical strings to distinguish
# repeated observations from distinct inputs. Count different labels within each
# group to expose inconsistent supervision without guessing a correct annotation.
#
# **🔎 Guiding Questions:** How many groups repeat, and how many rows do they contain?
# Do all observations with the same exact text carry the same label?
#
# **🗺️ Mental Model:** Three observations with text A form one text group.
# They contribute three participating rows, two extra occurrences, and either one
# or two distinct labels. These counts answer different questions.
#
# Optional reference: DataCamp's [Daily marketing reach by channel](https://campus.datacamp.com/courses/analyzing-marketing-campaigns-with-pandas/pandas-1?ex=9)
# uses grouping and `nunique()` to count distinct values within groups.

# %%
# text_counts maps each exact string (its index) to the number of source rows using it.
# Exact comparison still distinguishes case, whitespace, and punctuation.
text_counts = tweets["text"].value_counts()

# Keep only groups with more than one observation: len counts groups, sum counts rows.
repeated_text_counts = text_counts[text_counts > 1]

# duplicated(keep="first") flags later occurrences True and each first occurrence False.
# Checking the whole table additionally requires the target to match; neither call deletes rows.
extra_text_rows = tweets["text"].duplicated(keep="first").sum()
extra_identical_rows = tweets.duplicated(keep="first").sum()
print("Unique exact texts:", len(text_counts))
print("Repeated exact-text groups:", len(repeated_text_counts))
print("Rows belonging to repeated groups:", repeated_text_counts.sum())
print("Additional exact-text occurrences:", extra_text_rows)
print("Additional identical (text, target) rows:", extra_identical_rows)

# Each grouped result holds the number of distinct target values, not observations.
# Its index contains text strings, not the original integer source-row indices.
labels_per_text = tweets.groupby("text")["target"].nunique()

# Select counts above one and extract their index: the strings whose groups have both labels.
conflicting_texts = labels_per_text[labels_per_text > 1].index

# Map the conflicting group keys back to one Boolean flag per original source row.
# isin is True for any row whose text belongs to those groups, including every label.
exact_conflict_mask = tweets["text"].isin(conflicting_texts)
conflicting_row_count = exact_conflict_mask.sum()
print("Exact-text groups with both labels:", len(conflicting_texts))
print("Rows belonging to conflicting groups:", conflicting_row_count)

# %% [markdown]
# ### ✅ Evidence & Interpretation
# There are 7,503 distinct exact texts. The 69 repeated groups contain 179 rows,
# including 110 occurrences beyond the first. Only 92 extra rows also repeat the label.
# Eighteen text groups have both labels, involving 55 observations.
#
# Repetition and conflict are different: repeating one label adds no disagreement;
# recording two labels for one input does. These counts do not identify which label
# is correct or explain why the annotations differ.
#
# ### 🧠 Learning → Better Questions
# The initial row total overstates the number of distinct exact inputs. A table
# can have complete fields while offering contradictory supervision.
# Before deciding which examples survive, **does the intended preparation make
# previously different strings count as the same input?**

# %% [markdown]
# ## 3. Understand what normalization changes
#
# ### ❓ Main Question
# Which observations become equivalent under our chosen text preparation?
#
# **🎯 Goal & Approach — What & Why:** Compare the original text with a copy
# lowercased, reduced to single whitespace separators, and trimmed at both ends.
# Use this prepared string as both the duplicate key and the eventual model input.
# Otherwise, differently written inputs could bypass the duplicate policy.
#
# **🔎 Guiding Questions:** Do distinct original strings merge? Does that change
# which source rows belong to conflicting groups?
#
# This fixed rule learns no vocabulary or statistics from the corpus. It keeps
# words, punctuation, hashtags, mentions, and URLs, though their case may change.
# Lowercasing loses emphasis and case distinctions; retaining original text makes
# those losses inspectable. No stemming, stop-word removal, or encoding repair is used.
#
# The school example motivates visible preparation steps; its broader movie-review
# cleaning is not adopted. DataCamp's [Normalizing reviews](https://campus.datacamp.com/courses/regular-expressions-in-python/basic-concepts-of-string-manipulation?ex=6)
# illustrates string operations; here `.str` applies them to every text in a pandas column.

# %%
# prepared copies every source row, preserving its index, original text, and target.
# The new prepared_text column holds the comparable input without overwriting the original.
prepared = tweets.copy()
prepared["prepared_text"] = prepared["text"].str.lower()

# The regular expression \s+ matches one or more whitespace characters, including newlines.
# Replace each run with one space, then strip only leading and trailing whitespace.
prepared["prepared_text"] = prepared["prepared_text"].str.replace(r"\s+", " ", regex=True)
prepared["prepared_text"] = prepared["prepared_text"].str.strip()

# Both Series are indexed by normalized strings: count distinct labels or original forms.
# A normalized group can now collect source rows that had different exact strings.
labels_per_prepared_text = prepared.groupby("prepared_text")["target"].nunique()
original_forms_per_key = prepared.groupby("prepared_text")["text"].nunique()

# collision_keys lists normalized strings merging originals; conflicting_keys lists label disputes.
# These are group keys, not source-row numbers; more than one form does not imply a conflict.
collision_keys = original_forms_per_key[original_forms_per_key > 1].index
conflicting_keys = labels_per_prepared_text[labels_per_prepared_text > 1].index

# collision_mask marks each source row involved in a merge. .loc keeps True rows and all columns.
# The displayed integer index still identifies where each observation came from in the CSV.
collision_mask = prepared["prepared_text"].isin(collision_keys)
print("Distinct normalized texts before exclusions:", len(labels_per_prepared_text))
print("Normalized keys merging distinct original texts:", len(collision_keys))
print("Original/prepared example:")
print(prepared[["text", "prepared_text", "target"]].head(1).to_string())
print("Rows in normalization collisions (source indices retained):")
print(prepared.loc[collision_mask].to_string())

# %% [markdown]
# ### ✅ Evidence & Interpretation
# Normalization leaves 7,501 distinct keys, merging two pairs of original strings.
# Source rows 3340 and 7604 differ in whitespace; row 6119 adds a case variant
# to an existing conflicting group. The source indices are zero-based row positions.
#
# ### 🧠 Learning → Better Questions
# Equality depends on the representation used. Inspecting only original strings
# would miss a row that becomes equivalent to conflicting inputs after preparation.
# **How does applying the agreed policy to these normalized groups change the
# population we will evaluate?**

# %% [markdown]
# ## 4. Apply the retention policy and account for the population
#
# ### ❓ Main Question
# Which observations remain under the agreed policy, and what does exclusion cost?
#
# **🎯 Goal & Approach — What & Why:** Exclude every row in a normalized group
# containing both labels, then keep the first source occurrence of each consistent
# group. Count these two exclusion reasons separately to make the change traceable.
# This is the project's approved policy, not a prescribed school rule or label repair.
#
# **🔎 Guiding Questions:** How many rows and labels does each step remove?
# Does the resulting table still contain repeated inputs or conflicting groups?
#
# A group's first source occurrence is a reproducible choice, not evidence that
# it is better annotated. Excluding disagreements changes the evaluation population
# and may make the task easier; it does not settle those disagreements.
#
# Optional references: [Driving right (1)](https://campus.datacamp.com/courses/intermediate-python/logic-control-flow-and-filtering?ex=15)
# separates a Boolean mask from its selection; [Dropping duplicates](https://campus.datacamp.com/courses/data-manipulation-with-pandas/aggregating-dataframes?ex=7)
# illustrates how the chosen columns define repetition.

# %%
# Create one flag per source row: True means its normalized group has both labels.
# .loc selects whole rows; ~ flips the flags so consistent_rows keeps every other group.
conflict_mask = prepared["prepared_text"].isin(conflicting_keys)
conflicting_rows = prepared.loc[conflict_mask]
consistent_rows = prepared.loc[~conflict_mask]

# Among consistent rows, True marks occurrences after the first matching prepared_text.
# Filtering preserves source order; the first occurrence has the earliest source index.
repeat_mask = consistent_rows.duplicated(subset="prepared_text", keep="first")
redundant_rows = consistent_rows.loc[repeat_mask]

# model_tweets keeps one row per surviving normalized input, with original text and label.
# Its source indices can have gaps; copying does not reset or renumber them.
model_tweets = consistent_rows.loc[~repeat_mask].copy()

# Build one summary record per stage; each dictionary names that stage's counts.
# stage_rows is the corresponding table; eq(label).sum() counts observations of that class.
stage_records = []
for stage, stage_rows in [
    ("original", tweets),
    ("conflicts excluded", conflicting_rows),
    ("repeats excluded", redundant_rows),
    ("prepared", model_tweets),
]:
    stage_records.append({
        "stage": stage,
        "label_0": stage_rows["target"].eq(0).sum(),
        "label_1": stage_rows["target"].eq(1).sum(),
        "total": len(stage_rows),
    })

# Convert the records to a table indexed by stage names, not tweets or source positions.
preparation_counts = pd.DataFrame(stage_records).set_index("stage")
print("Rows by stage and target label:")
print(preparation_counts.to_string())
print("Conflicting normalized groups:", len(conflicting_keys))

# Subtract the disjoint removed and retained populations; zero means every row is accounted for.
unaccounted_rows = len(tweets) - len(conflicting_rows) - len(redundant_rows) - len(model_tweets)

# Count any later repeated input and any surviving group with more than one distinct label.
# remaining_label_counts again has normalized-text keys; each value counts labels, not rows.
remaining_duplicates = model_tweets["prepared_text"].duplicated().sum()
remaining_label_counts = model_tweets.groupby("prepared_text")["target"].nunique()
remaining_conflicts = (remaining_label_counts > 1).sum()
print("Unaccounted source rows:", unaccounted_rows)
print("Remaining extra occurrences / conflicting groups:", remaining_duplicates, remaining_conflicts)

# %% [markdown]
# ### ✅ Evidence & Interpretation
# The 18 conflicting normalized groups contain 56 rows: 29 labeled `0`, 27 labeled `1`.
# Removing them, then 74 redundant consistent occurrences (16 / 58 by label),
# leaves 7,483 rows: 4,297 labeled `0`, 3,186 labeled `1`.
# The accounting difference, remaining duplicates, and remaining conflicts are all zero.
#
# Class `1` now represents 42.58%, compared with 42.97% originally. A small shift
# in class proportions does not make the exclusions neutral: ambiguous examples
# are absent from the population on which future evaluation will be based.
#
# ### 🧠 Learning → Better Questions
# A group-level decision must be mapped back to all its source rows. Otherwise,
# removing only a disputed occurrence could silently retain an arbitrary answer.
# Preparation therefore defines the population a future score describes, not just
# the input format. **How can we separate this population while preserving label
# alignment, and what kind of independence can we actually verify?**

# %% [markdown]
# ## 5. Separate training and holdout observations
#
# ### ❓ Main Question
# What does our training/holdout separation preserve, and what remains uncertain?
#
# **🎯 Goal & Approach — What & Why:** Reserve 20% of the prepared rows for
# holdout evaluation using stratification and `random_state=42`. Split whole rows
# first, so original text, prepared text, labels, and source indices travel together.
#
# **🔎 Guiding Questions:** Are class proportions similar in both partitions?
# Do they share any identical original or prepared text?
#
# Stratification keeps label proportions close, subject to whole-row rounding.
# The seed makes selection reproducible for the same inputs, order, and environment.
# It cannot establish representativeness; changing inputs or order can change the split.
# Neither mechanism prevents repeated texts or related events from crossing partitions.
# The preceding retention policy addresses matching prepared text before this split.
#
# DataCamp's [Train/test split + computing accuracy](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/classification-1?ex=8)
# illustrates stratification and a fixed seed. Only that separation concept is
# adapted here; no model, score, or learned text transformation is produced.

# %%
import sklearn
from sklearn.model_selection import train_test_split

# Randomly partition whole observations; each returned table retains its source indices.
# stratify uses target counts, not text groups; neither table is sorted back into source order.
print("scikit-learn:", sklearn.__version__)
train_rows, holdout_rows = train_test_split(
    model_tweets, test_size=0.20, random_state=42, stratify=model_tweets["target"]
)

# X holds prepared input strings; y holds recorded labels, each as a one-dimensional Series.
# Selecting from the same partition preserves matching source indices and element order.
X_train = train_rows["prepared_text"]
y_train = train_rows["target"]
X_holdout = holdout_rows["prepared_text"]
y_holdout = holdout_rows["target"]

# Summarize each partition separately: one row per label, percent of that partition's rows.
# counts and summary are recalculated for each table, so the denominator changes with it.
for name, frame in [("training", train_rows), ("holdout", holdout_rows)]:
    counts = frame["target"].value_counts().sort_index()
    summary = counts.to_frame(name="rows")
    summary["percent"] = (counts / len(frame) * 100).round(2)
    print(f"\n{name}: {len(frame)} rows")
    print(summary.to_string())

# %%
# Sets contain distinct strings rather than rows; & finds strings present on both sides.
# These overlap objects therefore hold shared text values, not source indices or labels.
original_overlap = set(train_rows["text"]) & set(holdout_rows["text"])
prepared_overlap = set(X_train) & set(X_holdout)
print("Partition rows / prepared rows:", len(train_rows) + len(holdout_rows), len(model_tweets))
print("Shared original texts:", len(original_overlap))
print("Shared prepared texts:", len(prepared_overlap))

# %% [markdown]
# ### ✅ Evidence & Interpretation
# Results from the supplied data and existing environment:
#
# | Population | Rows | Label 0 | Label 1 | Label 1 share |
# | --- | ---: | ---: | ---: | ---: |
# | Original | 7,613 | 4,342 | 3,271 | 42.97% |
# | Prepared | 7,483 | 4,297 | 3,186 | 42.58% |
# | Training | 5,986 | 3,437 | 2,549 | 42.58% |
# | Holdout | 1,497 | 860 | 637 | 42.55% |
#
# The partitions contain all 7,483 prepared rows and closely preserve their class mix.
# Both original-text and prepared-text overlap counts are zero. This establishes
# separation under the compared text definitions, not model performance.
#
# ### 🧠 Learning → Better Questions
# The starting row count suggested how much data was available. Grouping and
# normalization showed why distinct inputs, consistent labels, and retained rows
# are different quantities. Filtering then changed which population can be evaluated.
#
# Stratification controls the class mix; removing repeated inputs addresses a
# different risk. Zero shared strings still cannot establish independence between
# reports about the same event or the truth of retained annotations.
#
# **Better question:** What additional evidence would justify extending a future
# score on these retained tweets to other events or ambiguous original examples?
# This analysis cannot answer that from matching text alone. Keep any future claim
# limited to the retained population until broader evidence supports it.
#
# Any later learned text transformation must be fitted on training data only.
# The holdout is reserved for evaluation, not tuning; no model is fitted in this file.
#
# ### 🧑‍💼 Recruiter / Transfer Check
# A different dataset has balanced labels and no exact text matches across a split.
# Is that enough to claim its test observations are independent? Explain the gap.
#
# <details><summary>💡 Show answer</summary>
#
# No. Label balance and exact-string separation answer only two specific questions.
# Text normalization could merge variants, and different wording can describe the
# same event. Independence needs evidence about those relationships; balanced class
# counts do not supply it. This prepared answer does not assess a learner's understanding.
#
# </details>
