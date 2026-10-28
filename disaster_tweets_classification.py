# %% [markdown]
# # Disaster Tweets — From Recorded Rows to a Measured Baseline
#
# **Project Question:** Can tweet text help distinguish disaster from non-disaster
# examples, and what would make evidence of that ability trustworthy?
#
# ### 📋 Context, Source & Task
# MasterSchool DA112 Sprint 3 asks us to classify disaster tweets using the supplied
# `DATA/train.csv`: text and target (`1` disaster, `0` non-disaster).
# These recorded annotations are not independently verified facts about events.
# The Project Map documents this teaching copy; its exact transformation history is unknown.
#
# This assistant-authored reconstruction retrospectively explains the dataset investigation,
# preparation, and baseline evaluation. Its questions organize the reasoning without
# claiming a learner's past thoughts or independent understanding.
# The separate post-baseline section revisits guided cross-validation and model comparison;
# it does not replace the published baseline or provide a new final holdout evaluation.
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
# Optional reference: DataCamp's [Daily marketing reach by channel](https://campus.datacamp.com/courses/analyzing-marketing-campaigns-with-pandas/pandas-1?ex=9)
# uses grouping and `nunique()` to count distinct values within groups.
#
# %% [markdown]
# ### 🗺️ Mental Model — Rows become a group
#
# ```text
# A/0   A/0   A/1
#  \     |     /
#   +----+----+
#        |
#        v
# One text group
# 3 rows; 2 extra
# 2 labels -> disagreement
# ```

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
# Of 7,503 distinct exact texts, 69 repeated groups contain 179 rows, including 110
# occurrences beyond the first. Only 92 extra rows also repeat the label.
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
# This fixed rule learns no corpus statistics and keeps words, punctuation,
# hashtags, mentions, and URLs, though their case may change.
# Lowercasing loses emphasis and case distinctions; preserved original text makes those losses inspectable.
# No stemming, stop-word removal, or encoding repair is used.
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
# Conflict membership therefore increases from 55 to 56 rows, while the number of
# conflicting groups remains 18.
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
# in class proportions does not make the exclusions neutral.
# Examples with conflicting recorded labels are excluded;
# other ambiguous or incorrectly labeled texts may remain.
#
# ### 🧠 Learning → Better Questions
# Mapping group decisions back to every source row avoids retaining an arbitrary
# label from a disputed group. Preparation defines the population a score describes,
# not just the input format. **Which tweets will our result describe, and which
# were excluded before evaluation?**
# It will describe held-out examples drawn from the 7,483 retained rows.
# The 56 conflicting rows and 74 redundant occurrences are excluded before that evaluation;
# the result cannot establish performance on those removed observations.
#
# **⚖️ Ethical Takeaway:** Disclose exclusions so readers do not mistake results
# on retained tweets for evidence covering every original example.

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
# Stratification keeps label proportions close, subject to whole-row rounding;
# the seed makes selection reproducible for the same inputs, order, and environment.
# Neither establishes representativeness or prevents related events from crossing partitions.
# The preceding retention policy addresses matching prepared text before this split.
#
# DataCamp's [Train/test split + computing accuracy](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/classification-1?ex=8)
# illustrates stratification and a fixed seed. Only that separation concept is
# adapted in this preparation section; it produces no model or learned text transformation.

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
# Generalization to those populations is not established by this text-only table.
# It would require event identifiers or verified links between reports, annotation review,
# and separately collected examples representative of the intended use.
# Keep claims limited to the retained population until that evidence exists.
#
# Any later learned text transformation must be fitted on training data only.
# The holdout is reserved for evaluation, not tuning; no model is fitted in this preparation section.
#
# ### 🧑‍💼 Recruiter / Transfer Check
# A different dataset has balanced labels and no exact text matches across a split.
# Is that enough to claim its test observations are independent? Explain the gap.
#
# <details><summary>💡 Show answer</summary>
#
# No: label balance and exact-string separation alone do not establish independence.
# Normalization can merge variants, while different wording can describe the same event;
# independence needs evidence about those relationships.
# This prepared answer supports reasoning without assessing a learner's understanding.
#
# </details>

# %% [markdown]
# ## Work block 2 — Baseline model and evaluation
#
# ### ❓ Main Question
# How well does the TF-IDF + Logistic Regression baseline classify the held-out
# tweets, and where does it fail?
#
# **📋 Context, Source & Task:** Use the unchanged prepared training and holdout
# objects above. The outcome is one measured baseline for the retained population,
# not evidence of reliability on every disaster report.
#
# The school `guided.ipynb` demonstrates fitting TF-IDF on training text, transforming
# held-out text, and fitting/evaluating Logistic Regression. Here those steps form
# one Pipeline. We retain its sound sequence, not its movie-review cleaning,
# `sag` solver choice, warning suppression, or example scores.
#
# DataCamp sources inspected for this adaptation:
# - [TF-IDF vectors for TED talks](https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/tf-idf-and-similarity-scores?ex=3): instructions and starter code for numerical text features.
# - [Your first TfIdf](https://campus.datacamp.com/courses/sentiment-analysis-in-python/more-on-numeric-vectors-transforming-tweets?ex=14): separate learning and transformation.
# - [Pipeline for song genre prediction I](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=6)
#   and [II](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=7): named steps, fitting, and prediction.
# - [Assessing a diabetes prediction classifier](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/fine-tuning-your-model-3?ex=3): classification report and confusion matrix.
#
# These are inspected instructions and available starter code, not claimed completed
# official solutions. Our tweet-specific code and explanations are adaptations.
# Installed scikit-learn 1.5.2 documentation clarifies defaults and API behavior;
# it supplies no additional modeling method.
#
# ### From the guided manual sequence to our Pipeline
# The inspected `guided.ipynb` cells below use zero-based notebook positions.
# This table explains correspondence; it does not run a second version of our baseline.
#
# | Guided operation | What our Pipeline does |
# | --- | --- |
# | Cell 9: vectorizer `fit_transform(X_train)` | During `fit`, learn training vocabulary/IDF and produce training features. |
# | Cell 10: classifier `fit(training_features, y_train)` | During the same `fit`, learn classifier coefficients from those features and labels. |
# | Cell 11: vectorizer `transform(X_test)` | During `predict`, transform held-out text using the already learned vocabulary/IDF. |
# | Cell 12: classifier `predict(test_features)` | During the same `predict`, use the fitted classifier to return labels. |
#
# Pipeline preserves this order and passes each step's output to the next.
# It is equivalent to manual chaining when inputs and settings match; our defaults
# differ from the guided `sag` solver, so this is not a claim of identical models.
#
# One predictive Pipeline has one final classifier here.
# Guided cell 15 compares two classifiers in a loop; they are alternatives, not
# two simultaneous final predictors inside one Pipeline.
#
# **⚠️ Source check:** Guided cell 1's positive/negative variable names disagree with
# the loaded file suffixes; names alone cannot establish label meaning.
# Cell 9 prints the sample count twice, so its displayed feature count is unreliable.
# We inspect actual inputs and matrix dimensions rather than copying those displays.
#
# ### 6. A — How does text become numerical evidence?
#
# **🎯 Goal & Approach — What & Why:** Translate each prepared tweet into numerical
# features so a classifier can learn associations with its recorded label.
# Learn that representation only from training text to keep evaluation separate.
#
# **🔎 Guiding Questions:** What occupies a matrix row or column? Which information
# becomes a feature, and which information is lost?
#
# Each matrix row represents one document (a tweet); each column represents one
# vocabulary term. The vectorizer's rule selects text units called tokens;
# the vocabulary contains the terms found during training.
#
# Term frequency counts occurrences within a document; inverse document frequency
# gives more weight to terms appearing in fewer training documents.
# Defaults multiply counts by smoothed IDF, then scale each nonzero row so its
# squared weights sum to one. These weights are not probabilities.
#
# Most vocabulary terms are absent from a particular tweet, giving that row many
# zero weights. A sparse matrix stores nonzero entries efficiently instead of a
# large dense table.
#
# **⚠️ Watch Out:** The default `(?u)\b\w\w+\b` rule selects runs of at least two
# Unicode word characters: letters, digits, and underscores.
# Punctuation separates tokens: `#fire` becomes `fire`, while the one-character `t`
# in a URL is dropped. URLs and mentions contribute fragments; single-token features
# lose word order and the context needed to interpret negation, even when `not` remains.
#
# Fixed TF-IDF defaults use word unigrams, no vocabulary cap or stop-word removal,
# smoothed IDF, and L2 row normalization.
# Logistic Regression uses `lbfgs`, L2 regularization (`C=1`), an intercept,
# equal class weights, and at most 100 solver iterations.
# Regularization discourages large coefficients; these baseline settings were fixed
# before inspecting holdout scores.
#
# API checks: [TfidfVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
# and [LogisticRegression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html).

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# This unfitted vectorizer defines the token/weight rules; it has not learned vocabulary or IDF.
# The unfitted classifier will learn coefficients for those numerical features plus an intercept.
tfidf_vectorizer = TfidfVectorizer()
logistic_classifier = LogisticRegression()

# Each tuple gives a step a name and its object. List order is the data-processing order.
# Creating the Pipeline connects the objects; it does not yet learn from any tweet.
baseline_steps = [("tfidf", tfidf_vectorizer), ("classifier", logistic_classifier)]
baseline_pipeline = Pipeline(baseline_steps)

# %% [markdown]
# ### 🧠 Learning → Better Questions
# A prepared string and a model feature are different objects. Tokenization and
# weighting determine which distinctions the classifier can use; this representation
# cannot retain every distinction visible to a human reader.
# **Which quantities must be learned, and how can prediction reuse them without learning from the holdout?**
#
# ### 7. B — What does the classifier learn, and how does it predict?
#
# **🎯 Goal & Approach — What & Why:** Fit the single Pipeline on the 5,986 training
# observations, then reuse its learned transformation and classifier for the holdout.
# This keeps feature columns consistent between learning and prediction.
#
# **🔎 Guiding Questions:** Which step uses text alone, which uses labels,
# and what changes between `fit` and `predict`?
#
# %% [markdown]
# ### 🗺️ Mental Model — Learn once, reuse for prediction
#
# ```text
# FIT
# Training text
#       |
#       v
# Learn vocabulary + IDF
#       |
#       v
# Features + training labels
#       |
#       v
# Learn coefficients + intercept
#
# PREDICT
# Holdout text
#       |
#       v
# Reuse vocabulary + IDF
#       |
#       v
# Features -> reuse classifier
#       |
#       v
# Predicted labels
# ```
#
# %% [markdown]
# Logistic Regression learns one coefficient per term and an intercept from
# numerical features and recorded labels.
# Their weighted combination becomes a class-1 probability estimate; `predict`
# returns class 1 above 0.5 and class 0 otherwise.
# These learned associations do not explain what caused a real event.
#
# Prediction ignores words absent from the learned vocabulary, without adding
# columns or recalculating IDF. With no known tokens, the feature row is zero,
# leaving only the classifier's intercept.
#
# **What does Pipeline guarantee about fit/predict ordering, and what remains unknown
# about label quality or event independence?** It applies configured steps consistently,
# reusing learned transformations before prediction.
# Correct training inputs, annotation quality, and event independence still require
# separate evidence.
# See the installed-version [Pipeline contract](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html).

# %%
# fit first learns vocabulary and IDF from X_train and transforms that same training text.
# It then fits classifier coefficients/intercept using the resulting matrix and y_train.
# TF-IDF does not use the labels; the classifier does. No holdout input participates here.
baseline_pipeline.fit(X_train, y_train)

# Inspect one training document using the SAME fitted vectorizer, without another fit.
# Its single sparse row uses the entire learned feature axis; .shape means (documents, terms).
training_example_text = X_train.head(1)
training_example_features = tfidf_vectorizer.transform(training_example_text)
print("Training example (source index retained):")
print(training_example_text.to_string())
print("One training document's feature-matrix shape:", training_example_features.shape)

# %% [markdown]
# ### ✅ Evidence & Interpretation
# The fitted representation has 18,494 term columns; the inspected training document
# occupies one row on that axis.
# The classifier's training matrix has 5,986 rows, one per training observation.
# Training converged within the default limit without warnings or parameter adjustments.
#
# TF-IDF learned vocabulary and IDF from text alone; Logistic Regression used those
# features and the recorded training labels to learn coefficients and an intercept.
# Prediction reuses all of these quantities without updating them from holdout text or labels.
#
# ### 🧠 Learning → Better Questions
# Fitting has established a representation and model, not their performance on unseen
# observations. **How much agreement does reusing these learned quantities produce
# with the recorded holdout labels, and which errors does that agreement hide?**

# %%
# predict transforms X_holdout with the learned vocabulary/IDF, then applies the fitted classifier.
# The returned array contains one 0/1 per input position, in X_holdout order; it has no pandas index.
holdout_predictions = baseline_pipeline.predict(X_holdout)
print("Holdout predictions:", len(holdout_predictions))

# %% [markdown]
# ### 8. C — What do the scores and mistakes tell us?
#
# **🎯 Goal & Approach — What & Why:** Compare predictions with the 1,497 recorded
# holdout labels. Use positive-class F1 as agreed, then inspect its component errors.
# The holdout is now evaluation evidence; its results must not guide changes to this baseline.
#
# **🔎 Guiding Questions:** How precise and complete are predictions of class 1?
# Which disagreements are hidden by the overall score, and what can examples explain?
#
# With `target=1` positive, a true positive (TP) agrees with that label;
# a false positive (FP) predicts disaster for label 0.
# A false negative (FN) misses a recorded disaster; a true negative (TN) agrees on class 0.
# These names describe agreement with annotations, not independently checked events.
#
# | Metric | Question and denominator |
# | --- | --- |
# | Precision, class 1 | Of predicted disasters, how many agree with label 1? TP / (TP + FP). |
# | Recall, class 1 | Of recorded disasters, how many were found? TP / (TP + FN). |
# | F1, class 1 | Harmonic balance of precision and recall: 2TP / (2TP + FP + FN). |
# | Accuracy | What fraction of all predictions match recorded labels? (TP + TN) / all rows. |
# | Support | How many evaluated observations carry each recorded label? |
#
# F1 weights FP and FN equally in this formula; it does not establish their real-world
# costs. Class averages in the report are supplementary; the row labeled `1` supplies
# the primary F1. Accuracy alone can conceal an uneven ability to recognize the two classes.
#
# **✅ Notes for Students:** The inspected DataCamp Twitter-metrics exercise describes
# confusion-matrix axes in reverse. The actual scikit-learn convention is recorded labels
# in rows, predictions in columns; use that convention and explicit `[0, 1]` order here.
# See [Twitter lesson](https://campus.datacamp.com/courses/sentiment-analysis-in-python/lets-predict-the-sentiment?ex=6)
# and the [1.5.2 API definition](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.confusion_matrix.html).

# %%
from sklearn.metrics import classification_report, confusion_matrix

# This formatted report compares pairs in matching holdout order, not by pandas index lookup.
# Each class row contains precision, recall, F1, and support; digits controls display rounding only.
holdout_report = classification_report(
    y_holdout, holdout_predictions, labels=[0, 1], digits=4
)
print("Holdout classification report (primary metric: F1 in row 1):")
print(holdout_report)

# Counts are arranged [[TN, FP], [FN, TP]] because row = recorded label, column = prediction.
# The DataFrame adds readable axis labels; it does not change the counts or their order.
confusion_counts = confusion_matrix(y_holdout, holdout_predictions, labels=[0, 1])
confusion_table = pd.DataFrame(
    confusion_counts,
    index=["Recorded 0", "Recorded 1"],
    columns=["Predicted 0", "Predicted 1"],
)
print("Confusion matrix (counts):")
print(confusion_table.to_string())

# %% [markdown]
# ### ✅ Evidence & Interpretation
# On the unchanged 1,497-row holdout, class-1 F1 is **0.7308**, precision **0.7989**,
# recall **0.6735**, and overall accuracy **0.7889**. Support is 860 for class 0 and
# 637 for class 1. These values describe this retained population and fixed experiment.
#
# The matrix contains 752 TN, 108 FP, 208 FN, and 429 TP.
# Thus 429 of 537 predicted disasters match label 1, while 429 of 637 recorded
# disasters are found.
#
# **What do the 208 missed examples among 637 recorded positives reveal that
# overall accuracy does not?** The model misses 32.65% of recorded positives,
# despite matching 1,181 of 1,497 labels overall.
# Accuracy combines both classes and hides this uneven recognition.
#
# Scores count disagreements but do not explain them. Inspect at most three examples
# per error type, taking the lowest source indices so the selection is reproducible.
# These are illustrative examples, not a representative error sample or evidence of prevalence.
#
# Keep recorded labels, predictions, visible wording, and hypotheses separate.
# An annotation is the evaluation reference, not proof of what happened; do not relabel
# an example or alter the pipeline after reading these errors.
#
# Selection uses DataCamp's [Subsetting rows](https://campus.datacamp.com/courses/data-manipulation-with-pandas/transforming-dataframes?ex=7)
# (including available official solutions) and [Sorting by index values](https://campus.datacamp.com/courses/data-manipulation-with-pandas/slicing-and-indexing-dataframes?ex=5)
# (instructions and starter code). Adapt their filtering/sorting patterns to source-row indices.

# %%
# Copy original text and recorded target in the SAME holdout order used for prediction.
# The separate table preserves source indices; none of the eight preparation objects is changed.
holdout_results = holdout_rows[["text", "target"]].copy()

# An unlabeled array is assigned by position: prediction i belongs to input row i.
# This new column now has holdout_results' source indices, not an invented 0..1496 index.
# Attach predictions before sorting; a default-index Series could silently misalign them.
holdout_results["prediction"] = holdout_predictions

# Each comparison returns one Boolean per source row; & requires both conditions to hold.
# FP selects recorded 0 / predicted 1; FN selects recorded 1 / predicted 0.
false_positive_mask = (holdout_results["target"] == 0) & (holdout_results["prediction"] == 1)
false_negative_mask = (holdout_results["target"] == 1) & (holdout_results["prediction"] == 0)

# .loc keeps matching records; sort_index orders their source-row keys, not model scores.
# head(3) caps each illustrative table at three records without changing the evaluation set.
false_positives = holdout_results.loc[false_positive_mask].sort_index().head(3)
false_negatives = holdout_results.loc[false_negative_mask].sort_index().head(3)
print("False positives: first three source indices (recorded 0, predicted 1):")
print(false_positives.to_string())
print("False negatives: first three source indices (recorded 1, predicted 0):")
print(false_negatives.to_string())

# %% [markdown]
# ### ✅ Evidence & Interpretation — selected disagreements
# Source indices below identify the printed records. The last column contains
# possible explanations, not measured feature contributions or verified causes.
#
# | Source row | Recorded / predicted | Visible text evidence | Hypothesis, not established cause |
# | --- | --- | --- | --- |
# | 26 | 0 / 1 (FP) | A short statement about being in NYC last week; no explicit disaster claim. | Learned associations with location or other tokens may outweigh the lack of an event description. |
# | 61 | 0 / 1 (FP) | Students' pens are described as ablaze in a publication announcement. | A single-word representation may confuse figurative fire language with an event. |
# | 71 | 0 / 1 (FP) | Accident-related hashtags, a general teen-accident headline, and links appear together. | Topic vocabulary may dominate the difference between general discussion and reporting a specific event. |
# | 12 | 1 / 0 (FN) | Rain/flooding and location hashtags describe many days of weather. | Sparse local wording or learned token associations may be insufficient for a positive decision. |
# | 51 | 1 / 0 (FN) | A headline describes imprisonment for setting a relative ablaze. | Brief crime-report wording may receive associations different from disaster reporting. |
# | 56 | 1 / 0 (FN) | A truck-fire report includes an address, abbreviations, and a link. | Abbreviations and fragmented location context may contribute to a missed report. |
#
# Fire-related wording appears in both error directions, challenging the assumption
# that disaster keywords alone settle the label.
# These examples establish neither feature contributions, correct annotations, nor
# the prevalence of a proposed error pattern.
# No example was relabeled or used to tune settings.
#
# ### 🧠 Learning → Better Questions
# The analysis now measures one model's agreement with labels in a defined evaluation
# population. Precision exceeds recall for class 1: predicted positives often match
# annotations, but almost one-third of recorded positives are missed.
# F1 summarizes that tradeoff; the matrix and examples reveal what the summary hides.
#
# The retained population excludes conflicting recorded labels, not all ambiguous or
# incorrectly labeled texts. Zero matching strings across partitions does not prove
# that reports concern independent events. A single holdout provides neither a
# deployment-readiness judgment nor evidence of emergency-response benefit.
#
# **Better question:** What independent evidence about event overlap and annotation
# context would be needed before extending this result beyond the retained tweets?
# Those relationships are not established by the supplied text-only table.
# Event identifiers or reviewed report links would reveal shared events;
# annotation guidelines and independent review would help assess label disagreements.
# Separately collected, representative examples would be needed to evaluate a broader claim.
#
# **Answer to the Project Question:** Text supports measured agreement with recorded
# labels in this retained holdout: class-1 F1 is 0.7308, with 208 missed recorded positives.
# Trust beyond this experiment also requires evidence about annotation quality,
# event separation, and the intended evaluation population; those conditions remain unestablished.
#
# The written reasoning is a learning resource. Data's independent Learning PASS
# remains **UNASSESSED**; successful execution does not demonstrate understanding.

# %% [markdown]
# ## Post-baseline learning — Guided cross-validation and model comparison
#
# ### ❓ Main Question
# How do classifier choice and adjacent token pairs change training-fold F1,
# and how consistent are these comparisons across folds?
#
# **📋 Context, Source & Task:** This is a retrospective educational extension,
# written after the published baseline holdout was inspected.
# It uses only the existing 5,986 training strings and labels; the 1,497-row holdout
# and its original baseline results remain separate.
#
# **🎯 Goal & Approach — What & Why:** Compare four configurations on the same
# ten validation folds to separate classifier choice from adding short token sequences.
# Keep all learned text transformations inside each fold so its validation text
# cannot influence vocabulary or IDF.
#
# **🔎 Guiding Questions:** What does one fold score describe?
# Do bigrams improve both classifiers, and does the highest mean establish a final winner?
#
# ### 9. What we transfer from the actual guided notebook
# Guided cell 14 computes ten F1 scores, their mean, and their standard deviation.
# Cell 15 compares Logistic Regression and LinearSVC with `(1, n)` n-grams for
# `n = 1, 2, 3, 4`; it does not vary Logistic Regression's `C`.
#
# **⚠️ Watch Out — fold leakage:** Both guided cells fit TF-IDF on all training
# text before CV, so each validation fold has already influenced vocabulary and IDF.
# We retain the comparison idea but pass raw prepared strings and the entire
# Pipeline into CV; each fold learns its own representation.
#
# Guided `cv=10` already gives this binary classifier stratified folds without shuffling.
# We explicitly shuffle within classes with seed 42, retaining approximate class proportions.
# Reusing the same integer-seeded splitter and input order gives every candidate the same folds.
#
# Each validation fold contains 598 or 599 observations; fitting uses the other
# 5,388 or 5,387 training observations.
# Each training observation is validated once per configuration, but fitting subsets overlap.
# These are not ten independent experiments or ten new external datasets.
#
# The inspected DataCamp instructions and starter code support:
# - [Cross-validation for R-squared](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-55bf0d67-0cc2-4212-a6e5-bda0d07d3411?ex=9): splitter, seed, and per-fold scores; we adapt the mechanism to stratified classification and F1.
# - [Analyzing cross-validation metrics](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-55bf0d67-0cc2-4212-a6e5-bda0d07d3411?ex=10): mean and standard deviation; fold quantiles alone are not a confidence interval for generalization performance.
# - [Visualizing classification model performance](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=14): loop over alternative models; here learned transformations belong inside the evaluated Pipeline.
#
# These are inspected teaching examples, not official tweet solutions or imported scores.
# The following code is our adaptation; the source notebooks remain unchanged.

# %% [markdown]
# ### 🗺️ Mental Model — One of ten training-only folds
#
# ```text
# Existing 5,986 training rows
#            |
#       +----+----+
#       |         |
#   Nine parts  One part
#       |       validation
#       v         |
#   Fit TF-IDF    |
#       |         |
#   Fit one       |
#   classifier    |
#       |         |
#       +----+----+
#            |
#   Reuse fitted Pipeline
#   on validation text
#            |
#    One class-1 F1 score
#
# Repeat for all ten parts
# Holdout stays outside
# ```

# %% [markdown]
# ### 10. Keep the comparison small and explain the alternatives
# `(1, 1)` uses individual tokens; `(1, 2)` adds adjacent pairs, such as `not safe`,
# while retaining individual tokens.
# Bigrams preserve limited local order, but do not guarantee better scores or understanding of a sentence.
#
# Two n-gram choices isolate this small representation change without the guided
# notebook's longer three- and four-token sequences.
# Both choices keep the existing tokenization, normalization, stop-word policy, and other TF-IDF defaults.
#
# Logistic Regression learns coefficients using a probability-based loss.
# LinearSVC learns a linear separating boundary using a margin-based loss;
# its decision scores are not native probability estimates.
# A margin measures signed separation from the decision boundary, rather than the probability that a recorded label is correct.
#
# We keep `C=1` and each estimator's remaining defaults; equal `C` does not make
# their different objectives equivalent.
# Logistic Regression keeps `lbfgs`; LinearSVC uses its default squared-hinge loss,
# L2 penalty, and `dual="auto"` optimizer selection, with seed 42 for reproducibility.
#
# Ten folds times four configurations means 40 fresh Pipeline fits.
# The unigram Logistic Regression configuration is evaluated once in this comparison;
# we do not add a duplicate standalone CV run or refit the selected candidate afterwards.
#
# API references for the installed version:
# [cross_val_score](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.cross_val_score.html),
# [StratifiedKFold](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.StratifiedKFold.html),
# and [LinearSVC](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html).

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

# cv_splitter defines ten training/validation partitions within X_train and y_train.
# The integer seed recreates identical memberships for each candidate; no holdout enters it.
cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Each tuple names an unfitted classifier option; neither is the fitted baseline object.
# ngram_options lists the smallest and largest token-sequence lengths included as features.
cv_classifier_options = [
    ("Logistic Regression", LogisticRegression()),
    ("LinearSVC", LinearSVC(random_state=42)),
]
ngram_options = [(1, 1), (1, 2)]

# Each dictionary entry will map a configuration name to its ten validation F1 scores.
# Each summary record will hold one configuration's descriptive statistics, not tweet rows.
fold_scores_by_configuration = {}
cv_summary_records = []

for classifier_name, classifier in cv_classifier_options:
    for ngram_range in ngram_options:
        # This candidate couples a fresh vectorizer with one classifier option.
        # cross_val_score clones the entire candidate for every fold, leaving it unfitted here.
        cv_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=ngram_range)),
            ("classifier", classifier),
        ])

        # One array element is class-1 F1 from that fold's withheld training observations.
        # Each clone fits vocabulary, IDF, and classifier only on the other nine parts.
        # Run folds sequentially; raise fit errors rather than replacing failures with missing scores.
        fold_f1_scores = cross_val_score(
            cv_pipeline, X_train, y_train,
            cv=cv_splitter, scoring="f1", n_jobs=1, error_score="raise",
        )

        # A readable configuration key identifies the model and its included n-gram lengths.
        # Store unrounded scores; mean averages ten folds, and std(ddof=0) describes their spread.
        configuration = f"{classifier_name} {ngram_range}"
        fold_scores_by_configuration[configuration] = fold_f1_scores
        cv_summary_records.append({
            "configuration": configuration,
            "mean_f1": fold_f1_scores.mean(),
            "std_f1": fold_f1_scores.std(ddof=0),
            "min_f1": fold_f1_scores.min(),
            "max_f1": fold_f1_scores.max(),
        })
        print(f"Completed ten training-only folds: {configuration}")

# Rows represent fold numbers 1..10, not source-row indices; columns identify configurations.
# Matching rows use matching validation observations because all candidates share the splitter.
cv_fold_scores = pd.DataFrame(fold_scores_by_configuration, index=range(1, 11))
cv_fold_scores.index.name = "fold"
print("\nClass-1 validation F1 by training fold:")
print(cv_fold_scores.round(4).to_string())

# One row per configuration summarizes all ten scores; the index contains configuration names.
# min/max show observed extremes; rounding changes printed values only, not stored results.
cv_summary = pd.DataFrame(cv_summary_records).set_index("configuration")
print("\nRetrospective training-only CV summary (not final holdout performance):")
print(cv_summary.round(4).to_string())

# %% [markdown]
# ### 11. Read the fold results without overstating them
# ### ✅ Evidence & Interpretation
# The existing Python 3.11.11 / scikit-learn 1.5.2 environment produced the results
# below with the fixed ten-fold splitter; all 40 fits completed without warnings.
# These are newly measured training-fold results, not school scores or replacements for the baseline holdout evidence.
#
# | Classifier | N-grams | Mean class-1 F1 | Fold standard deviation | Minimum | Maximum |
# | --- | --- | ---: | ---: | ---: | ---: |
# | Logistic Regression | `(1, 1)` | 0.7506 | 0.0197 | 0.7161 | 0.7907 |
# | Logistic Regression | `(1, 2)` | 0.7293 | 0.0181 | 0.6931 | 0.7667 |
# | LinearSVC | `(1, 1)` | 0.7554 | 0.0246 | 0.7184 | 0.7895 |
# | LinearSVC | `(1, 2)` | 0.7529 | 0.0170 | 0.7265 | 0.7821 |
#
# Unigram LinearSVC has the highest mean, about 0.0048 above unigram Logistic Regression,
# but exceeds it in only four of ten matching folds.
# Its fold spread is also larger: 0.0246 versus 0.0197.
# This observed ranking does not show a consistent advantage across every validation subset.
#
# Adding bigrams lowers Logistic Regression's mean from 0.7506 to 0.7293;
# all ten of its fold scores decrease.
# LinearSVC's mean also falls, from 0.7554 to 0.7529, although individual folds move in both directions.
# Bigrams therefore improve neither average here; these scores alone do not establish the cause or predict another dataset's outcome.
#
# Each score measures agreement with recorded class-1 labels in one withheld part
# of the original training set, after fitting on the other nine parts.
# The mean averages ten separate F1 values; it is not F1 calculated from pooled predictions.
#
# Standard deviation describes variation among these ten scores, using divisor ten (`ddof=0`).
# It is neither a confidence interval nor a significance test, and overlapping fitting
# subsets prevent treating the scores as independent experiments.
# LinearSVC with bigrams has the smallest observed spread (0.0170), but that does not
# establish robustness to other partitions, new events, or different annotations.
# The original 0.7308 holdout F1 evaluates a different set and training size;
# subtracting it from a CV average would not measure an improvement on the same evaluation.
#
# **⚠️ Watch Out — retrospective selection:** The highest CV mean identifies only
# the highest observed average among these four configurations on this partition.
# Using the same results to choose a configuration and advertise its performance introduces selection optimism.
# Our already inspected holdout cannot become a new independent final evaluation.
#
# Guided cell 16 refits its chosen configuration and evaluates its test set.
# We deliberately stop at training-only comparison, keeping the original baseline model,
# predictions, and published holdout metrics unchanged.
#
# ### 🧠 Learning → Better Questions
# Cross-validation checks sensitivity to training/validation membership within the retained
# training population; it does not repair annotation quality or establish event independence.
# **What evidence would justify treating a preferred configuration as more reliable outside these folds?**
# That claim is not established here: it needs reviewed labels, event relationships,
# and fresh representative evaluation data kept separate from model choice.
#
# **⚖️ Ethical Takeaway:** Report these results as retrospective training-only comparisons,
# so a favorable CV average is not mistaken for verified disaster-detection reliability.
#
# Hugging Face remains a separate learning task: inspect the actual model card,
# task labels, resources, and dependency needs before any approved setup or inference.
# Its pretrained-model `pipeline` serves a different purpose from fitting our scikit-learn
# Pipeline; a sentiment prediction is not a verified disaster annotation.
