# %% [markdown]
# # Disaster Tweets — Project Overview
#
# **Problem:** Can tweet text distinguish recorded disaster and non-disaster examples?
# **Goal:** Build an understandable classifier and evaluate how trustworthy its predictions are.
# **Context:** MasterSchool DA112; an educational analysis of labeled tweets, not an emergency-response system.
#
# ## Explore the five project phases
# Open any summary for the question, approach, evidence, and limitations.
# These summaries describe previously verified results; they do not execute the analysis.
# Detailed teaching cells follow in the same phase order.
# Look for **⚠️ Watch Out** blocks beside the evidence: they highlight consequential interpretation traps.
#
# <details>
# <summary><strong>1. Data Understanding, Quality & Preparation</strong></summary>
#
# **Question:** Which observations can support a fair evaluation?
#
# **Approach:** Audit labels and repeated texts, apply the agreed retention policy, then create the fixed stratified split.
#
# **Evidence:** 7,613 source rows become 7,483 retained rows: 5,986 training and 1,497 holdout.
#
# **Limit / next step:** No shared original or prepared strings across the split does not prove independence between events.
#
# </details>
#
# <details>
# <summary><strong>2. Exploratory Data Analysis & Visualization</strong></summary>
#
# **Question:** What patterns appear before modeling?
#
# **Approach:** Compare training-text distributions, then inspect tokenization, lemmatization, and information loss in five training examples.
#
# **Evidence:** Median character lengths are 101 and 114 for classes 0 and 1; both have median 15 whitespace-separated parts.
# The separate language lab changes 15 of 97 tokens; its results do not change the classifier inputs.
#
# **Limit / next step:** Distributions overlap. These descriptive differences establish neither causes nor predictive performance.
#
# </details>
#
# <details>
# <summary><strong>3. Predictive Modeling, Comparison & Evaluation</strong></summary>
#
# **Question:** How well does the classifier agree with unseen recorded labels?
#
# **Approach:** Evaluate the TF-IDF + Logistic Regression baseline; separately compare six full Pipelines in ten training-only CV folds.
#
# **Evidence:** Baseline holdout F1 is 0.7308, precision 0.7989, and recall 0.6735; 208 recorded positives are missed.
#
# **Limit / next step:** CV is a post-baseline learning comparison. It provides no new independent holdout test or deployment guarantee.
#
# </details>
#
# <details>
# <summary><strong>4. Hugging Face: Pretrained Transformer Mini-Lab</strong></summary>
#
# **Question:** What does a pretrained Transformer inference demonstrate?
#
# **Approach:** Run one local English sentiment prediction and contrast Hugging Face pipeline with scikit-learn Pipeline.
#
# **Evidence:** The movie-review example returns POSITIVE with a score of 0.999887.
#
# **Limit / next step:** This score is not measured accuracy. Sentiment inference supplies no evidence of disaster-classification performance.
#
# </details>
#
# <details>
# <summary><strong>5. Key Insights, Limitations & Recommendations</strong></summary>
#
# **Question:** What conclusions and next steps does the evidence justify?
#
# **Approach:** Connect observed scores, error examples, population limits, and the separate sentiment demonstration.
#
# **Evidence:** Text supports measured agreement with retained annotations; real-world disaster-detection reliability remains unestablished.
#
# **Limit / next step:** Review annotations and event relationships, then obtain fresh representative evaluation data before making broader claims.
#
# </details>
#
# **Reading and running:** Opening a summary does not hide or run the detailed cells below.
# In the Python editor, use **Fold Marker Regions** to fold the five complete phases; the overview stays visible.
# For execution, run cells from top to bottom: later phases depend on objects created earlier.
#
# <!--
# region 1. Data Understanding, Quality & Preparation
# -->

# %% [markdown]
# ## 1. Data Understanding, Quality & Preparation
#
# ### Context and source notes
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
# ### 1.1 Understand the observations and labels
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
# This file uses pandas, scikit-learn, Plotly, and NLTK; ipykernel supports the Interactive Window.
# Graph setup: `plotly==7.1.0` and `nbformat==5.11.1` in the existing environment.
# nbformat supports inline display; no Matplotlib, Seaborn, Dash, or web account is needed.
# The training-text lab below documents its separate, one-time NLTK resource setup.
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
# ### Visual evidence — recorded class balance
# A bar chart makes the count difference visible; printed percentages keep its size explicit.
# These are annotations in this supplied table, not disaster prevalence in the world.
# DataCamp's [categorical bar-chart exercise](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=2)
# motivates aggregation before plotting; the implementation below is our Plotly adaptation.

# %%
import plotly.express as px

# reset_index makes the label index a plotting column; the source table stays unchanged.
eda_class_counts = target_summary.reset_index()
eda_class_counts["class"] = eda_class_counts["target"].map({0: "0 — Non-disaster", 1: "1 — Disaster"})
eda_class_figure = px.bar(
    eda_class_counts, x="class", y="rows", text="rows",
    hover_data={"percent": ":.2f", "target": False},
    labels={"class": "Recorded class", "rows": "Source rows", "percent": "Percent of 7,613 rows"},
    title="Recorded labels in the original dataset (n = 7,613)",
    height=360, template="plotly_white",
)
eda_class_figure.update_traces(marker_color=["#0072B2", "#D55E00"], textposition="outside")
eda_class_figure.update_layout(yaxis_range=[0, 5000], margin=dict(t=65, b=55))
# show renders an interactive figure; it does not publish or upload the dataset.
eda_class_figure.show()

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
# ### 1.2 Distinguish repeated texts from conflicting annotations
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
# <!-- Reserve diagram space before Mermaid finishes rendering in the Interactive Window. -->
# <div style="height: 300px; overflow: auto;">
#
# ```mermaid
# flowchart LR
#     R1["A / 0"] --> G["One text group<br/>3 rows; 2 extra"]
#     R2["A / 0"] --> G
#     R3["A / 1"] --> G
#     G --> L["2 labels<br/>Disagreement"]
# ```
#
# </div>

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
# ### 1.3 Understand what normalization changes
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
# ### 1.4 Apply the retention policy and account for the population
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
# ### Visual evidence — what the retention policy removes
# Compare retained and excluded counts in separate panels because their scales differ greatly.
# Exclusion reasons are disjoint; stacked colors show each reason's recorded label composition.
# Compare counts using each panel's axis, not bar heights across panels.

# %%
# melt turns the two label columns into rows so Plotly can color the stacked bars.
eda_retention = preparation_counts.loc[["prepared", "conflicts excluded", "repeats excluded"], ["label_0", "label_1"]]
eda_retention = eda_retention.reset_index().melt(id_vars="stage", var_name="class", value_name="rows")
eda_retention["class"] = eda_retention["class"].map({"label_0": "0 — Non-disaster", "label_1": "1 — Disaster"})
eda_retention["panel"] = eda_retention["stage"].map({
    "prepared": "Retained", "conflicts excluded": "Excluded", "repeats excluded": "Excluded",
})
eda_retention_figure = px.bar(
    eda_retention, x="stage", y="rows", color="class", text="rows", facet_col="panel",
    color_discrete_sequence=["#0072B2", "#D55E00"],
    labels={"stage": "Policy outcome", "rows": "Source rows", "class": "Recorded class"},
    title="Retention accounting: 7,483 retained + 56 conflicts + 74 repeats = 7,613",
    height=400, template="plotly_white",
)
# Independent axes prevent the small exclusion counts from disappearing beside retained rows.
eda_retention_figure.update_yaxes(matches=None, showticklabels=True, rangemode="tozero")
eda_retention_figure.update_xaxes(matches=None)
eda_retention_figure.update_layout(legend=dict(orientation="h", y=-0.3), margin=dict(b=100))
eda_retention_figure.show()

# %% [markdown]
# ### ✅ Evidence & Interpretation
# The 18 conflicting normalized groups contain 56 rows: 29 labeled `0`, 27 labeled `1`.
# Removing them, then 74 redundant consistent occurrences (16 / 58 by label),
# leaves 7,483 rows: 4,297 labeled `0`, 3,186 labeled `1`.
# The accounting difference, remaining duplicates, and remaining conflicts are all zero.
#
# Class `1` now represents 42.58%, compared with 42.97% originally.
#
# > **⚠️ Watch Out — Excluding conflicts changes who the score describes**
# >
# > The policy removes 56 conflicting rows and 74 repeats before evaluation; a small change in class balance does not make exclusions neutral.
# > Agreement among retained labels does not establish that those labels are correct or that excluded cases would be classified reliably.
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
# ### 1.5 Separate training and holdout observations
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
# > **⚠️ Watch Out — Different texts can describe the same event**
# >
# > Zero shared strings rules out exact overlap under our comparisons; it does not establish event independence or annotation accuracy.
# > Stratification preserves class proportions but cannot prevent related reports from crossing partitions.
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

# <!--
# endregion
# region 2. Exploratory Data Analysis & Visualization
# -->

# %% [markdown]
# ## 2. Exploratory Data Analysis & Visualization
#
# Earlier data-audit cells show class balance and retention accounting where those decisions occur.
# This phase explores text lengths and language transformations after the split, using training rows only.
#
# ### Training-text distributions
#
# ### ❓ Main Question
# How do text-length distributions compare between recorded classes in the training subset?
#
# **🎯 Goal & Approach — What & Why:** Compare characters and whitespace-separated parts to distinguish two meanings of length.
# Use the existing training rows only; this retrospective exploration changes no model features or preparation decisions.
# **🔎 Guiding Questions:** Do typical lengths differ, and would a difference alone establish useful classification?
#
# **HOW:** Count characters in original text, including spaces and punctuation; split on whitespace for a simple second measure.
# These parts are not linguistic words or the tokens used by either TF-IDF or the Transformer.
# DataCamp's [grouped histograms](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=5)
# and [EDA histograms](https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/getting-to-know-a-dataset?ex=4)
# motivate distribution comparisons; this is an original adaptation to our training tweets.

# %%
# Work on a separate table: train_rows, model inputs, and source indices remain unchanged.
eda_lengths = train_rows[["target"]].copy()
eda_lengths["class"] = eda_lengths["target"].map({0: "0 — Non-disaster", 1: "1 — Disaster"})
eda_lengths["characters"] = train_rows["text"].str.len()
eda_lengths["whitespace_parts"] = train_rows["text"].str.split().str.len()
eda_length_summary = eda_lengths.groupby("class")[["characters", "whitespace_parts"]].agg(["count", "min", "median", "max"])
print("Training-only original-text lengths:")
print(eda_length_summary.to_string())

# %% [markdown]
# ### Character length — shape and overlap
# Both classes use the same 10-character intervals and axes, covering every observed length.
# Percentages use each class's own row count, preventing the larger class from dominating through sample size alone.
# Bins summarize a distribution; their boundaries affect its appearance and are not learned classification thresholds.

# %%
eda_character_figure = px.histogram(
    eda_lengths, x="characters", color="class", facet_col="class", histnorm="percent",
    category_orders={"class": ["0 — Non-disaster", "1 — Disaster"]},
    color_discrete_sequence=["#0072B2", "#D55E00"],
    labels={"characters": "Original-text characters", "class": "Recorded class"},
    title="Training-text character lengths (n = 3,437 / 2,549)",
    height=370, template="plotly_white",
)
# Explicit bin edges make the comparison reproducible; no observed values are clipped.
eda_character_figure.update_traces(xbins=dict(start=0, end=160, size=10))
eda_character_figure.update_yaxes(title_text="Percent within class")
eda_character_figure.update_layout(showlegend=False, bargap=0.05)
eda_character_figure.show()

# %% [markdown]
# ### Whitespace-separated parts — a complementary distribution
# A box plot shows the median and middle half of observations, with unusual values beyond its whiskers.
# Whiskers reach observed values within 1.5 times the interquartile range beyond each box edge.
# The interquartile range is the width of the middle half; points beyond whiskers are not automatically errors.
# This plot removes no rows.
# We use the linear quartile convention explicitly so the displayed summary has a defined calculation.

# %%
eda_parts_figure = px.box(
    eda_lengths, x="class", y="whitespace_parts", color="class", points="outliers",
    category_orders={"class": ["0 — Non-disaster", "1 — Disaster"]},
    color_discrete_sequence=["#0072B2", "#D55E00"],
    labels={"class": "Recorded class", "whitespace_parts": "Whitespace-separated parts per tweet"},
    title="Training-text length measured in whitespace-separated parts",
    height=360, template="plotly_white",
)
eda_parts_figure.update_traces(quartilemethod="linear")
eda_parts_figure.update_layout(showlegend=False)
eda_parts_figure.show()

# %% [markdown]
# ### ✅ Evidence & Interpretation
# Class 0 has 3,437 training rows: median 101 characters, range 7–157; class 1 has 2,549: median 114, range 14–150.
# Both classes have median 15 whitespace-separated parts; their ranges are 1–31 and 2–30, respectively.
# Typical character length differs, while typical part count does not; the distributions overlap and neither observation establishes classification performance.
#
# > **⚠️ Watch Out — Different distributions do not establish predictive value**
# >
# > The distributions overlap; their descriptive differences alone establish neither classification performance nor the causes of those differences.
# > Recorded labels also remain annotations, not independently verified events.
#
# The original table has no missing fields, so a missingness chart would add little beyond the existing zero-count check.
#
# ### 🧠 Learning → Better Questions
# Length depends on its definition; inspecting only an average or median would hide variation within each class.
# **Better question:** What accounts for the character-length difference when the median part count is identical?
# **Current answer:** Not established; examining part lengths, URLs, and annotation context in training data could investigate it, without assuming a cause.
# **⚖️ Ethical Takeaway:** Longer text is not evidence that a report describes a real emergency.

# %% [markdown]
# ### Text Preprocessing: Tokenization & Lemmatization
#
# ### ❓ Main Question
# How do tokenization and lemmatization change selected training tweets, and what information could cleaning erase?
#
# **🎯 Goal & Approach — What & Why:** Make language transformations visible before deciding whether they belong in a classifier.
# Inspect five reproducibly sampled training tweets in a separate table; keep the established model inputs and evaluation unchanged.
# **🔎 Guiding Questions:** Which units become tokens, which forms change, and does a smaller vocabulary establish better predictions?
#
# **📋 Context, Source & Task:** This is an original retrospective adaptation of school methods and DataCamp teaching examples.
# Source locations below use one-based notebook cell numbers and PDF page numbers; saved school outputs are not our results.
#
# | Teaching reference | What we transfer or adapt |
# | --- | --- |
# | MasterSchool `GMT20250210-Introduction to NLP.pdf`, pp. 20–29 | Tokenization, stemming, and lemmatization; cleaning choices depend on the task. |
# | MasterSchool `3.2 Campus Project Preparation.pptx.pdf`, pp. 11–15 | Inspect tokens, grammatical tags, lemmas, and reconstructed text. Its code examples use 20 Newsgroups. |
# | MasterSchool `GMT20250212-fetch_20newsgroups.ipynb`, cells 10–22 | WordNet with POS tags; here tag whole sequences before filtering, rather than each word in isolation. |
# | MasterSchool `guided_campus.ipynb`, cell 4 | An explicit preparation sequence; its source matches our local `guided.ipynb`. |
# | DataCamp [tokenization](https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/text-processing-fundamentals?ex=2), [stemming](https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/text-processing-fundamentals?ex=9), and [lemmatization](https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/text-processing-fundamentals?ex=10) | Inspect intermediate representations, using our own examples and NLTK. |
# | DataCamp [before/after lemmas](https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/text-preprocessing-pos-tagging-and-ner?ex=4) and [POS tagging](https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/text-preprocessing-pos-tagging-and-ner?ex=9) | Transfer the comparison and grammatical context; those exercises use spaCy, which this lab does not require. |
#
# ### WHY → HOW: distinguish the operations
# Tokenization chooses text units; it does not yet assign TF-IDF weights or learn disaster labels.
# Stemming uses shortening rules, while lemmatization looks up possible base forms using a dictionary and a specified part of speech.
# POS tagging estimates grammatical roles from a token sequence; WordNet itself does not read the surrounding sentence.
#
# School and DataCamp examples use `word_tokenize`; we compare it with NLTK's `TweetTokenizer` on one sampled tweet.
# The latter preserves hashtags and the URL in this example, making the adaptation to social text inspectable.
# This is a representation comparison, not evidence that either tokenizer improves classification.
#
# **Setup used:** `nltk==3.10.4`, with `punkt_tab`, `wordnet`, `averaged_perceptron_tagger_eng`, and `stopwords` under `~/nltk_data`.
# One-time setup in an approved environment:
# `.venv/bin/python -m pip install --only-binary=:all: "nltk==3.10.4"`
# then `.venv/bin/python -m nltk.downloader -d ~/nltk_data punkt_tab wordnet averaged_perceptron_tagger_eng stopwords`.
# These commands are documentation, not automatic downloads during analysis; restart the kernel after installation.
# See [NLTK data setup](https://www.nltk.org/data.html), [TweetTokenizer](https://www.nltk.org/api/nltk.tokenize.casual.html),
# [POS tagging](https://www.nltk.org/api/nltk.tag.html), and [WordNet lemmatization](https://www.nltk.org/api/nltk.stem.wordnet.html).

# %% [markdown]
# ### 🗺️ Mental Model — Inspect language transformations
#
# <div style="height: 180px; overflow: auto;">
#
# ```mermaid
# flowchart LR
#     T["Five training<br/>tweets"] --> K["Tokenize<br/>Keep context"]
#     K --> P["Estimate POS<br/>Look up lemmas"]
#     P --> E["Inspect changes<br/>and information loss"]
# ```
#
# </div>
# This teaching branch does not replace the baseline inputs.

# %%
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, word_tokenize

# Sample without looking at labels or model errors; retain CSV source indices for traceability.
# Five observations illustrate behavior but cannot estimate its prevalence across the dataset.
nlp_examples = train_rows[["text", "prepared_text"]].sample(n=5, random_state=42).sort_index()
print("\nLanguage lab: NLTK", nltk.__version__)
print("Five training examples, indexed by source row:")
print(nlp_examples[["text"]].to_string())

# Preserve case for grammatical tagging and keep handles/repeated characters as configured.
# The example is the second sampled row in source-index order, not a chosen model error.
nlp_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
nlp_example_text = nlp_examples["text"].iloc[1]
print("\nTokenization example, source row:", nlp_examples.index[1])
print("School word_tokenize:", word_tokenize(nlp_example_text))
print("TweetTokenizer:", nlp_tokenizer.tokenize(nlp_example_text))

# A short original illustration supplies known grammatical roles: noun, verb, and adjective.
# The default WordNet call assumes a noun; the explicit POS call can produce a different lemma.
nlp_lemmatizer = WordNetLemmatizer()
nlp_stemmer = PorterStemmer()
nlp_method_records = []
for word, wordnet_pos in [("floods", "n"), ("running", "v"), ("better", "a")]:
    nlp_method_records.append({
        "word": word, "Porter stem": nlp_stemmer.stem(word),
        "noun-default lemma": nlp_lemmatizer.lemmatize(word),
        "specified POS": wordnet_pos, "POS lemma": nlp_lemmatizer.lemmatize(word, pos=wordnet_pos),
    })
print("\nIllustrative words, not dataset measurements:")
print(pd.DataFrame(nlp_method_records).to_string(index=False))

# %% [markdown]
# ### Apply dictionary lookup with grammatical context
# Tag each complete token sequence before considering removal, retaining surrounding words as context.
# Map Treebank tag prefixes to WordNet roles: adjective, noun, verb, and adverb.
# Lemmatize alphabetic tokens with those roles, while leaving estimated proper names and other token types unchanged apart from lowercasing.
#
# > **⚠️ Watch Out — Grammatical tags are estimates**
# >
# > POS tags can be unreliable for abbreviations, informal text, and names; inspect the token/tag/lemma pairs before accepting changes.
# > Preserving negation and special tokens does not make this demonstration a validated cleaning policy or a lossless reconstruction.

# %%
# This dictionary translates two tag conventions; it does not learn anything from our labels.
# Each record below represents one token position in one sampled training tweet.
nlp_pos_map = {"J": "a", "N": "n", "V": "v", "R": "r"}
nlp_token_records = []
for source_row, original_text in nlp_examples["text"].items():
    nlp_tokens = nlp_tokenizer.tokenize(original_text)
    nlp_tagged_tokens = nltk.pos_tag(nlp_tokens)
    for position, (token, tag) in enumerate(nlp_tagged_tokens):
        # Lowercase only after tagging; compare lemmas with this same lowercased form.
        # NNP/NNPS denote estimated proper names; unknown tag prefixes return None.
        lower_token = token.lower()
        wordnet_pos = nlp_pos_map.get(tag[0])
        eligible = token.isalpha() and not tag.startswith("NNP") and wordnet_pos is not None
        lemma = nlp_lemmatizer.lemmatize(lower_token, pos=wordnet_pos) if eligible else lower_token
        nlp_token_records.append({
            "source_row": source_row, "position": position, "token": token,
            "lowercase": lower_token, "POS": tag, "lemma": lemma,
            "changed": lemma != lower_token,
        })

# Token-table indices are record positions; source_row and position retain the original mapping.
# The changed-only view excludes case changes, so it isolates the dictionary-lookup effect.
nlp_token_table = pd.DataFrame(nlp_token_records)
nlp_changed_tokens = nlp_token_table.loc[nlp_token_table["changed"], ["source_row", "token", "POS", "lemma"]]
print("\nAll changed tokens in the five-example lab:")
print(nlp_changed_tokens.to_string(index=False))
print("Token occurrences / changed occurrences:", len(nlp_token_table), len(nlp_changed_tokens))
print("Distinct lowercased forms / lemmas:", nlp_token_table["lowercase"].nunique(), nlp_token_table["lemma"].nunique())

# Grouping preserves token order within each tweet; joining is for inspection, not model input.
# These strings may contain new spaces around punctuation and are not exact original text.
nlp_reconstructed = nlp_token_table.groupby("source_row", sort=False)["lemma"].agg(" ".join)
print("\nReconstructed lemma strings (teaching output only):")
print(nlp_reconstructed.to_string())

# %% [markdown]
# ### Visual evidence — count changes within each sampled tweet
# Stack changed and unchanged token occurrences so each bar totals that tweet's token count.
# The denominator includes punctuation, URLs, and hashtags retained by this tokenizer; this is not a count of linguistic words.

# %%
# A status describes each token occurrence; grouping counts occurrences, not distinct vocabulary entries.
# Source-row numbers become strings so the chart treats them as five categories, not a continuous scale.
nlp_change_counts = nlp_token_table.assign(
    status=nlp_token_table["changed"].map({False: "Unchanged", True: "Changed by lemmatization"})
).groupby(["source_row", "status"]).size().reset_index(name="tokens")
nlp_change_counts["source_row"] = nlp_change_counts["source_row"].astype(str)
nlp_change_figure = px.bar(
    nlp_change_counts, x="source_row", y="tokens", color="status", text="tokens",
    category_orders={"status": ["Unchanged", "Changed by lemmatization"]},
    color_discrete_sequence=["#0072B2", "#D55E00"],
    labels={"source_row": "Sampled training source row", "tokens": "Token occurrences", "status": "Lookup result"},
    title="Lemmatization changes 15 of 97 tokens in five training examples",
    height=360, template="plotly_white",
)
nlp_change_figure.update_layout(legend=dict(orientation="h", y=-0.25), margin=dict(b=90))
nlp_change_figure.show()

# %% [markdown]
# ### Inspect information loss before calling text “clean”
# DataCamp's [stop-word exercise](https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/text-processing-fundamentals?ex=5)
# and the school notebooks filter common words using an English list.
# We test that operation on an original illustrative sentence because its negation has a clear role.

# %%
# This sentence is invented for a mechanism check; it is not a tweet or an evaluation observation.
# The stop-word set is a supplied language resource, not a list learned from our training data.
nlp_negation_example = "Residents are not trapped by the floods."
nlp_stop_words = set(stopwords.words("english"))
nlp_negation_tokens = [word.lower() for word in word_tokenize(nlp_negation_example) if word.isalpha()]
nlp_filtered_tokens = [word for word in nlp_negation_tokens if word not in nlp_stop_words]
nlp_removed_tokens = [word for word in nlp_negation_tokens if word in nlp_stop_words]
print("\nIllustrative sentence:", nlp_negation_example)
print("Removed stop words:", nlp_removed_tokens)
print("After that removal:", " ".join(nlp_filtered_tokens))

# %% [markdown]
# > **⚠️ Watch Out — Removing `not` erases the denial**
# >
# > **Before:** Residents are **not** trapped by the floods.
# >
# > **After stop-word removal:** `residents trapped floods`
# >
# > **A common word can carry essential meaning.** Removing `not` discards the explicit denial, so stop-word removal needs task-specific justification.
# > This constructed example demonstrates information loss, not a measured classification error.

# %% [markdown]
# ### ✅ Evidence & Interpretation
# In source row 2879, `word_tokenize` separates hashtag markers; `TweetTokenizer` keeps hashtag units and the URL together.
# Token definitions therefore change the units available to later processing, even for the same text.
# Both are different from the baseline vectorizer's default token rule, which we retain and inspect in the next phase.
#
# Across the five sampled tweets, 15 of 97 token occurrences change beyond lowercasing; distinct forms decrease from 89 to 87.
# Examples include `driven → drive`, `spreading → spread`, and `tracks → track`; these observations describe this sample only.
# The controlled word table also shows why a noun-default call leaves `running` unchanged, while specifying a verb produces `run`.
#
# No filtered or lemmatized string replaces `X_train`, `X_holdout`, or the duplicate-policy key.
#
# ### 🧠 Learning → Better Questions
# **Data → Information:** Original strings become inspectable token/tag/lemma records, with counted changes and traceable source rows.
# **Knowledge → Wisdom:** Fewer distinct forms can coexist with lost meaning; inspect both before adopting a transformation.
#
# > **⚠️ Watch Out — Fewer word forms do not establish better F1**
# >
# > The five-example sample measures transformations, not classifier performance.
# > Its vocabulary reduction establishes neither improved predictions nor better understanding of disaster reports.
#
# **Better question:** Does lemmatization improve class-1 F1 when all other modeling choices stay fixed?
# **Current answer:** Not established; this lab evaluates transformations, not a new classifier.
# Answering it would need a separately agreed training-only comparison with identical folds and learned features fitted inside each fold.
# **⚖️ Ethical Takeaway:** Keep the original text available so preprocessing cannot silently turn a denial into an apparent emergency claim.

# <!--
# endregion
# region 3. Predictive Modeling, Comparison & Evaluation
# -->

# %% [markdown]
# ## 3. Predictive Modeling, Comparison & Evaluation
#
# ### Baseline model and evaluation
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
# The newly inspected `GMT20250226 Pipeline II.ipynb`, cells 20–25, also puts text vectorization inside a model-selection Pipeline.
# We retain that separation principle; its saved source contains a stray `L` in cell 20, so it is not copied verbatim.
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
# > **⚠️ Watch Out — Verify school outputs against actual inputs**
# >
# > Guided cell 1's positive/negative variable names disagree with the loaded file suffixes; names alone cannot establish label meaning.
# > Cell 9 prints the sample count twice, so its displayed feature count is unreliable.
# > Check actual inputs and matrix dimensions before reusing those displays.
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
# The default `(?u)\b\w\w+\b` rule selects runs of at least two
# Unicode word characters: letters, digits, and underscores.
#
# > **⚠️ Watch Out — Keeping `not` does not preserve its context**
# >
# > Unigrams discard word order, so the baseline cannot directly represent which words `not` modifies.
# > Tokenization also changes visible text: `#fire` becomes `fire`, one-character tokens disappear, and URLs or mentions contribute fragments.
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
# <!-- Reserve diagram space before Mermaid finishes rendering in the Interactive Window. -->
# <div style="height: 240px; overflow: auto;">
#
# ```mermaid
# flowchart LR
#     T["FIT<br/>Training text"] --> V["Learn<br/>vocabulary + IDF"]
#     V --> F["Features +<br/>training labels"]
#     F --> C["Learn coefficients<br/>+ intercept"]
#     H["PREDICT<br/>Holdout text"] --> R["Reuse<br/>vocabulary + IDF"]
#     R --> P["Features<br/>Reuse classifier"]
#     P --> Y["Predicted labels"]
# ```
#
# </div>
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
# %% [markdown]
# ### From tokens to visible TF-IDF features
#
# **Main Question:** Which terms and weights does our existing fitted vectorizer produce for the inspected training tweet?
# **Goal:** Connect the language lab to the actual baseline representation, without fitting another vectorizer or changing predictions.
# MasterSchool's `GMT20250213-MovieLens 100K Dataset.ipynb`, cells 7–11, displays count and TF-IDF tables for small texts.
# DataCamp's [TF-IDF representation of product feedback](https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/feature-extraction-from-text?ex=7)
# supplies a similar table pattern; we inspect only one sparse row instead of making the entire training matrix dense.
#
# **HOW:** The fitted vectorizer's analyzer supplies its actual tokens; counting them gives this document's term frequencies.
# Stored feature indices connect nonzero weights to vocabulary names and learned IDF values.
# The graph ranks TF-IDF weights, which are neither classifier coefficients nor evidence that a term caused a prediction.

# %%
from collections import Counter

# Reuse the already transformed training example; these operations perform no fit.
# Counter maps each actual baseline token to its occurrence count within this one tweet.
nlp_baseline_tokens = tfidf_vectorizer.build_analyzer()(training_example_text.iloc[0])
nlp_term_counts = Counter(nlp_baseline_tokens)
nlp_sparse_row = training_example_features.getrow(0)
nlp_feature_indices = nlp_sparse_row.indices
nlp_feature_terms = tfidf_vectorizer.get_feature_names_out()[nlp_feature_indices]

# One row represents one nonzero feature, aligned by the sparse row's column indices.
# IDF comes from all training documents; TF-IDF also includes the configured L2 row normalization.
nlp_feature_table = pd.DataFrame({
    "term": nlp_feature_terms,
    "count_in_tweet": [nlp_term_counts[term] for term in nlp_feature_terms],
    "training_idf": tfidf_vectorizer.idf_[nlp_feature_indices],
    "tfidf_weight": nlp_sparse_row.data,
}).sort_values(["tfidf_weight", "term"], ascending=[False, True])
print("\nActual baseline tokens, source row", training_example_text.index[0], ":")
print(nlp_baseline_tokens)
print("Nonzero features / fitted vocabulary:", len(nlp_feature_table), len(tfidf_vectorizer.vocabulary_))
print("Ten largest TF-IDF weights for this training example:")
print(nlp_feature_table.head(10).round(4).to_string(index=False))

# A small display table keeps the plot readable; other nonzero features remain in nlp_feature_table.
nlp_feature_plot = nlp_feature_table.head(10).sort_values("tfidf_weight", kind="stable")
nlp_feature_figure = px.bar(
    nlp_feature_plot, x="tfidf_weight", y="term", orientation="h",
    hover_data={"count_in_tweet": True, "training_idf": ":.3f", "tfidf_weight": ":.4f"},
    labels={"tfidf_weight": "TF-IDF weight after L2 normalization", "term": "Vocabulary term",
            "count_in_tweet": "Occurrences in tweet", "training_idf": "Training IDF"},
    title=f"Ten largest feature weights: training source row {training_example_text.index[0]}",
    height=390, template="plotly_white",
)
nlp_feature_figure.update_traces(marker_color="#0072B2")
nlp_feature_figure.show()

# %% [markdown]
# ### ✅ Evidence & Interpretation
# The printed tokens and table answer the feature question using the unchanged fitted baseline, not the language lab's lemmas.
# Each displayed term has a document count, a training-derived IDF, and a normalized weight; the chart shows only ten nonzero entries.
# Vocabulary terms absent from the full sparse row have zero weight for this tweet.
# For equal counts within this document, larger IDF gives larger TF-IDF weight, without establishing greater predictive usefulness.
#
# **Better question:** Does a large TF-IDF weight imply that a term supports class 1?
#
# > **⚠️ Watch Out — TF-IDF weight is not a class contribution**
# >
# > **Answer:** No; whether a term supports class 1 also depends on the classifier's signed coefficient.
# > The chart displays input feature weights, not coefficients or their contributions to the decision.
# > A feature representation does not explain what caused a real disaster.
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
# ### Visual evidence — the same holdout disagreements
# MasterSchool's `GMT20250219 load_dataset('imdb').ipynb`, cells 22 and 26, displays confusion matrices as heatmaps.
# This Plotly adaptation visualizes our existing counts: rows are recorded labels and columns are predictions.
# It adds no evaluation observations or new model result; sentiment class names from the lesson do not apply here.

# %%
nlp_confusion_figure = px.imshow(
    confusion_counts, text_auto=True,
    x=["0 — Non-disaster", "1 — Disaster"],
    y=["0 — Non-disaster", "1 — Disaster"],
    labels={"x": "Predicted label", "y": "Recorded label", "color": "Tweets"},
    color_continuous_scale="Blues", zmin=0, zmax=int(confusion_counts.max()),
    title="Original baseline holdout: 108 false positives and 208 false negatives",
    height=380, template="plotly_white",
)
nlp_confusion_figure.update_xaxes(side="bottom")
nlp_confusion_figure.show()

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
# overall accuracy does not?**
#
# > **⚠️ Watch Out — Accuracy hides missed disaster labels**
# >
# > **Answer:** At 78.89% accuracy, the model still misses 208 of 637 recorded positives (32.65%).
# > The 1,181 matches among 1,497 labels combine both classes and hide this uneven recognition.
# > Read class-1 recall and the confusion matrix alongside accuracy.
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
# ### Post-baseline learning — Guided cross-validation and model comparison
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
# **🎯 Goal & Approach — What & Why:** Compare six configurations on the same
# ten validation folds to separate classifier choice from adding short token sequences.
# Keep all learned text transformations inside each fold so its validation text
# cannot influence vocabulary or IDF.
#
# **🔎 Guiding Questions:** What does one fold score describe?
# Do bigrams improve all three classifiers, and does the highest mean establish a final winner?
# What does Naive Bayes assume, and why can we use it with these TF-IDF features?
#
# ### 9. What we transfer from the actual guided notebook
# Guided cell 14 computes ten F1 scores, their mean, and their standard deviation.
# Cell 15 compares Logistic Regression and LinearSVC with `(1, n)` n-grams for
# `n = 1, 2, 3, 4`; it does not vary Logistic Regression's `C`.
#
# > **⚠️ Watch Out — Learn TF-IDF inside every CV fold**
# >
# > Both guided cells fit TF-IDF before CV, allowing each validation fold to influence vocabulary and IDF.
# > Passing raw prepared strings and the entire Pipeline into CV makes each fold learn its representation from that fold's training rows only.
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
#
# Sprint 3 suggests Logistic Regression or Naive Bayes; adding the latter here is
# a post-baseline learning choice, not a missing required baseline step.
# DataCamp's [Predicting the sentiment of a movie review](https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/n-gram-models?ex=8)
# supplies inspected instructions and starter code for fitting MultinomialNB to bag-of-words features.
# Our adaptation uses tweet TF-IDF, class-1 F1, and training-only CV instead of review accuracy; no course solution or score is copied.

# %% [markdown]
# ### 🗺️ Mental Model — One of ten training-only folds
#
# <!-- Reserve diagram space before Mermaid finishes rendering in the Interactive Window. -->
# <div style="height: 260px; overflow: auto;">
#
# ```mermaid
# flowchart LR
#     T["Nine training parts"] --> F["Fit Pipeline<br/>TF-IDF, then classifier"]
#     F --> P["Reuse fitted Pipeline<br/>Predict validation labels"]
#     V["One validation part"] -->|text| P
#     P --> S["One class-1<br/>F1 score"]
#     V -->|recorded labels| S
# ```
#
# </div>
# Split the existing 5,986 training rows; repeat for all ten validation parts. Holdout stays outside.

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
# ### Why add Multinomial Naive Bayes?
# It offers a different learning assumption while keeping the representation and evaluation fixed.
# Logistic Regression directly learns class probabilities through its loss; LinearSVC learns a margin-based boundary.
# Naive Bayes instead estimates class prevalence and term distributions within each class, then combines them using Bayes' rule.
#
# "Naive" means treating feature evidence as conditionally independent given the class.
# Related words and overlapping unigrams/bigrams violate that simplification; neither existing classifier requires this independence assumption.
# A useful classification score would not prove the assumption true or establish calibrated probabilities.
#
# The multinomial model describes token counts distributed across a vocabulary within each class.
# Our TF-IDF values are nonnegative fractional weights, which scikit-learn accepts as feature mass.
# They are neither literal counts nor probabilities, so this practical application does not establish that the count model holds literally.
#
# During fitting, MultinomialNB sums each feature's weight over training documents belonging to each class.
# Default `alpha=1.0` adds one to every class-feature total before normalization across that class's vocabulary.
# This Laplace smoothing prevents zero term probabilities for vocabulary terms absent from one class; alpha stays fixed, without tuning.
#
# Default `fit_prior=True` estimates class priors from that fold's label proportions.
# Prediction adds the log prior to the sum of feature weights times log smoothed term probabilities, choosing the higher-scoring class.
# Logs turn multiplication into addition; unknown tokens remain ignored by that fold's fitted vectorizer.
#
# > **⚠️ Watch Out — Compatible inputs do not validate Naive Bayes assumptions**
# >
# > Nonnegative TF-IDF permits this calculation; it does not prove independent language features, correct labels, or trustworthy probability estimates.
# > We compare predicted labels through F1, without interpreting probabilities as certainty about a real disaster.
#
# See the installed-version [Naive Bayes explanation](https://scikit-learn.org/1.5/modules/naive_bayes.html#multinomial-naive-bayes)
# and [MultinomialNB API](https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

# %% [markdown]
# ### 🗺️ Mental Model — Naive Bayes inside one fold
#
# <!-- Reserve diagram space before Mermaid finishes rendering in the Interactive Window. -->
# <div style="height: 380px; overflow: auto;">
#
# ```mermaid
# flowchart LR
#     T["Fold training<br/>TF-IDF + labels"] --> P["Class proportions<br/>Class priors"]
#     T --> W["Term weight totals per class<br/>Add alpha; normalize"]
#     P --> S["Class log scores<br/>Choose class 0 or 1"]
#     W --> S
#     V["Validation TF-IDF"] --> S
#     S --> F["Compare labels<br/>F1"]
#     L["Recorded<br/>validation labels"] --> F
# ```
#
# </div>

# %% [markdown]
# Ten folds times six configurations means 60 fresh Pipeline fits: the original 40 plus 20 Naive Bayes fits.
# The unigram Logistic Regression configuration is evaluated once in this comparison;
# we do not add a duplicate standalone CV run or refit the selected candidate afterwards.
#
# API references for the installed version:
# [cross_val_score](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.cross_val_score.html),
# [StratifiedKFold](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.StratifiedKFold.html),
# and [LinearSVC](https://scikit-learn.org/1.5/modules/generated/sklearn.svm.LinearSVC.html).

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# cv_splitter defines ten training/validation partitions within X_train and y_train.
# The integer seed recreates identical memberships for each candidate; no holdout enters it.
cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Each tuple names an unfitted classifier option; none is the fitted baseline object.
# ngram_options lists the smallest and largest token-sequence lengths included as features.
cv_classifier_options = [
    ("Logistic Regression", LogisticRegression()),
    ("LinearSVC", LinearSVC(random_state=42)),
    # NB sums nonnegative feature weights per class, smooths with alpha=1.0, then normalizes.
    # Each fold's label shares supply priors; prediction combines these with term evidence to choose 0 or 1.
    ("MultinomialNB", MultinomialNB()),
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
# below with the fixed ten-fold splitter; all 60 fits completed without warnings.
# The original four configurations reproduce their published means, standard deviations, minima, and maxima at four decimals.
# Their 40 printed fold scores also match the pre-extension rerun; the added 20 scores measure Naive Bayes on those same folds.
#
# | Classifier | N-grams | Mean class-1 F1 | Fold standard deviation | Minimum | Maximum |
# | --- | --- | ---: | ---: | ---: | ---: |
# | Logistic Regression | `(1, 1)` | 0.7506 | 0.0197 | 0.7161 | 0.7907 |
# | Logistic Regression | `(1, 2)` | 0.7293 | 0.0181 | 0.6931 | 0.7667 |
# | LinearSVC | `(1, 1)` | 0.7554 | 0.0246 | 0.7184 | 0.7895 |
# | LinearSVC | `(1, 2)` | 0.7529 | 0.0170 | 0.7265 | 0.7821 |
# | MultinomialNB | `(1, 1)` | 0.7219 | 0.0147 | 0.6991 | 0.7414 |
# | MultinomialNB | `(1, 2)` | 0.7113 | 0.0156 | 0.6905 | 0.7465 |
#
# The executable table prints every candidate's ten fold scores; these are the newly added Naive Bayes observations, rounded only for display.
# Matching fold numbers identify the same validation rows across configurations.
#
# | Fold | MultinomialNB `(1, 1)` | MultinomialNB `(1, 2)` |
# | ---: | ---: | ---: |
# | 1 | 0.7414 | 0.7000 |
# | 2 | 0.7040 | 0.7017 |
# | 3 | 0.7172 | 0.7031 |
# | 4 | 0.6991 | 0.6981 |
# | 5 | 0.7051 | 0.6905 |
# | 6 | 0.7390 | 0.7119 |
# | 7 | 0.7349 | 0.7465 |
# | 8 | 0.7339 | 0.7181 |
# | 9 | 0.7252 | 0.7264 |
# | 10 | 0.7189 | 0.7166 |
#
# Unigram LinearSVC has the highest mean, about 0.0048 above unigram Logistic Regression,
# but exceeds it in only four of ten matching folds.
# Its fold spread is also larger: 0.0246 versus 0.0197.
# This observed ranking does not show a consistent advantage across every validation subset.
#
# Adding bigrams lowers Logistic Regression's mean from 0.7506 to 0.7293;
# all ten of its fold scores decrease.
# LinearSVC's mean also falls, from 0.7554 to 0.7529, although individual folds move in both directions.
# These scores alone do not establish the cause or predict another dataset's outcome.
#
# Naive Bayes has lower means than both existing classifiers at each matching n-gram setting.
# Its unigram scores exceed unigram Logistic Regression in two folds and unigram LinearSVC in one, so the ordering is not universal.
# Adding bigrams lowers its mean from 0.7219 to 0.7113, with decreases in eight folds; bigrams improve none of the three averages.
#
# Naive Bayes supplies a valid contrasting calculation, but these results do not establish why its F1 is lower.
# Feature dependence, weighting, and smoothing could affect performance; this comparison does not isolate their effects or validate the model's assumptions.
#
# Each score measures agreement with recorded class-1 labels in one withheld part
# of the original training set, after fitting on the other nine parts.
# The mean averages ten separate F1 values; it is not F1 calculated from pooled predictions.
#
# Standard deviation describes variation among these ten scores, using divisor ten (`ddof=0`).
# It is neither a confidence interval nor a significance test, and overlapping fitting
# subsets prevent treating the scores as independent experiments.
#
# Unigram Naive Bayes has the smallest observed spread (0.0147), despite its lower mean than either existing unigram classifier.
# LinearSVC with bigrams still has the smallest spread among the original four (0.0170); neither observation establishes external robustness.
#
# The original 0.7308 holdout F1 evaluates a different set and training size;
# subtracting it from a CV average would not measure an improvement on the same evaluation.
#
# > **⚠️ Watch Out — The best CV average is not fresh test performance**
# >
# > The highest mean identifies only the highest observed average among these six configurations on this partition.
# > Choosing a configuration and advertising its performance from the same results introduces selection optimism.
# > Our already inspected holdout cannot become a new independent final evaluation.
#
# Guided cell 16 refits its chosen configuration and evaluates its test set.
# We deliberately stop at training-only comparison, keeping the original baseline model,
# predictions, and published holdout metrics unchanged.
#
# ### 🧠 Learning → Better Questions
# Naive Bayes illustrates that a smaller fold spread can accompany lower agreement with recorded positive labels.
# Spread describes consistency across these folds; it cannot by itself justify preferring a classifier or trusting its probabilities.
#
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

# <!--
# endregion
# region 4. Hugging Face: Pretrained Transformer Mini-Lab
# -->

# %% [markdown]
# ## 4. Hugging Face: Pretrained Transformer Mini-Lab
#
# ### One pretrained Transformer inference
#
# ### ❓ Main Question
# What does one pretrained sentiment prediction demonstrate, and how does its
# pipeline differ from our fitted scikit-learn Pipeline?
#
# **🎯 Goal & Approach — What & Why:** Learn to interpret a pretrained model's
# output within its actual task. Use one original English movie-review sentence,
# separate from our tweets, to observe local inference without training or model comparison.
#
# **🔎 Guiding Questions:** What was learned before this lab? What do the returned
# label and score mean, and can they answer our disaster-classification question?
#
# **📋 Context, Source & Task:** The closing MasterSchool Transformers lesson
# introduces `pipeline("sentiment-analysis")`, then calls it on text and prints the result.
# This assistant-authored adaptation names the checkpoint, revision, and CPU explicitly,
# instead of relying on an implicit default model and download.
#
# DataCamp's [Using AutoClasses](https://campus.datacamp.com/courses/working-with-hugging-face/building-pipelines-with-hugging-face?ex=11)
# provides inspected instructions and starter code combining this model, its tokenizer,
# and a sentiment pipeline. We use the direct pipeline interface; neither source supplies our runtime result.
# The inspected [sentiment lesson in Natural Language Processing (NLP) in Python](https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/text-classification-with-hugging-face?ex=2)
# uses the same DistilBERT checkpoint family; it reinforces this lab without requiring another model download or inference task.

# %% [markdown]
# ### 12. Choose a model whose output we can explain
# DistilBERT is a smaller Transformer derived from BERT through knowledge distillation:
# a smaller model learns from a larger model's behavior during pretraining.
# The authors then fine-tuned this checkpoint on English SST-2 sentiment examples;
# our lab only reuses those already learned weights.
#
# The selected model has approximately 67 million parameters, six Transformer layers,
# and a 268 MB float32 safetensors weight file. Its model card declares Apache-2.0.
# Its configuration maps `0` to `NEGATIVE` and `1` to `POSITIVE`, with a 512-position input limit.
# These label numbers describe sentiment and have no connection to our disaster targets.
#
# The tokenizer converts text into vocabulary IDs, including word pieces and special tokens.
# Attention lets each token's representation depend on other tokens in the input.
# A classification head converts the resulting representation into two sentiment scores;
# this computation does not verify facts or events.
#
# | Interface | What happens in this project? |
# | --- | --- |
# | scikit-learn `Pipeline.fit` | Learns TF-IDF and a classifier from our training text and labels; CV repeats fitting inside each fold. |
# | scikit-learn `Pipeline.predict` | Reuses those learned steps to classify new text. |
# | Hugging Face `pipeline(...)` | Loads a matching pretrained model and tokenizer, then packages tokenization, inference, and output processing. |
# | Calling the HF pipeline on text | Reuses existing weights; this mini-lab performs no fitting or fine-tuning. |
#
# > **⚠️ Watch Out — Two sentiment labels leave no neutral option**
# >
# > This checkpoint returns either positive or negative sentiment, even when neither label captures the input well.
# > Its model card reports country-dependent sentiment bias; one successful example cannot evaluate that bias or broader reliability.
#
# The card's generic "Direct Use" paragraph does not consistently describe this checkpoint;
# its actual configuration and SST-2 training description establish the task used here.
#
# Sources: [model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english),
# [pinned configuration](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/714eb0fa89d2f80546fda750413ed43d93601a13/config.json),
# [DistilBERT documentation](https://huggingface.co/docs/transformers/model_doc/distilbert),
# and [pipeline API](https://huggingface.co/docs/transformers/main_classes/pipelines).

# %% [markdown]
# ### 🗺️ Mental Model — Reuse learned weights
#
# <!-- Reserve diagram space before Mermaid finishes rendering in the Interactive Window. -->
# <div style="height: 140px; overflow: auto;">
#
# ```mermaid
# flowchart LR
#     T["Original<br/>English sentence"] --> K["Tokenizer"]
#     K --> M["Pretrained<br/>Transformer"]
#     M --> S["Sentiment<br/>label + score"]
# ```
#
# </div>

# %% [markdown]
# ### 13. Run one local inference
# This section additionally requires PyTorch and Transformers in the existing `.venv`.
# The approved setup uses Python 3.11.11, `torch==2.15.0`, and `transformers==5.21.0`.
# Restart the Interactive Window kernel after installation before running cells in order.
#
# Setup command, after separate installation approval:
# `.venv/bin/python -m pip install --only-binary=:all: -c requirements.txt "torch==2.15.0" "transformers==5.21.0"`
# Here `-c` constrains versions without installing every package in `requirements.txt`.
#
# First loading downloads the selected model files into the Hugging Face cache;
# the default location is `~/.cache/huggingface/hub`, outside this repository.
# Prediction runs locally on CPU; the demonstration sentence is not sent to a hosted inference service.
# See [installation and caching](https://huggingface.co/docs/transformers/installation).

# %%
import torch
import transformers

print("\nHugging Face mini-lab (sentiment demonstration only):")
print("PyTorch:", torch.__version__, "Transformers:", transformers.__version__)

# These strings identify the model repository and an immutable revision of its files.
# Pinning the revision keeps model/tokenizer selection stable across later runs.
hf_model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
hf_model_revision = "714eb0fa89d2f80546fda750413ed43d93601a13"

# Load matching pretrained weights and tokenizer; no tweet data or labels enter this operation.
# CPU is explicit; model_kwargs requests safetensors rather than another weight-file format.
sentiment_pipeline = transformers.pipeline(
    task="sentiment-analysis",
    model=hf_model_id,
    revision=hf_model_revision,
    device="cpu",
    model_kwargs={"use_safetensors": True},
)
print("Model:", hf_model_id)
print("Revision:", hf_model_revision)
# The loaded model's configuration maps numerical output positions to sentiment names.
print("Sentiment labels:", sentiment_pipeline.model.config.id2label)

# %%
# One original demonstration sentence, outside the project dataset and evaluation partitions.
# Keep natural text: this model uses its own tokenizer, not our TF-IDF preparation.
demo_text = "The film was delightful and beautifully acted."

# Calling the pipeline tokenizes the sentence, runs the fixed model, and processes its scores.
# The result is a list with one dictionary for this input: its top label and softmax score.
# Softmax turns two raw class scores into positive values summing to one; weights are not updated.
sentiment_result = sentiment_pipeline(demo_text)
print("Demonstration text:", demo_text)
print("Observed sentiment result:", sentiment_result)

# %% [markdown]
# ### ✅ Evidence & Interpretation
# The complete script ran successfully on CPU on 2026-10-29 with the versions and
# checkpoint revision printed above. The observed result for
# `The film was delightful and beautifully acted.` was `POSITIVE`, score **0.999887** (rounded).
#
# > **⚠️ Watch Out — A high sentiment score does not establish disaster detection**
# >
# > `POSITIVE` names a sentiment label; `0.999887` is its softmax score, not measured accuracy or a calibrated guarantee of correctness.
# > Neither available label answers whether a disaster occurred.
#
# Pretraining and SST-2 fine-tuning happened before this lab; this call reused the
# tokenizer and model without learning from our sentence or tweets.
# Unlike our scikit-learn `fit` calls, inference changed no model weights.
# This verifies one local prediction, not performance across a population.
#
# The full rerun reproduced the previous baseline and all 60 printed CV fold scores
# exactly, including their summaries. The holdout F1 remains 0.7308;
# the sentiment score is a different quantity and must not be compared with that F1.

# %% [markdown]
# ### 🧠 Learning → Better Questions
# The sentence and observed output are **Data**; the checkpoint, English sentiment
# task, and label definitions give them context as **Information**.
# **Knowledge:** a working inference path can answer its configured task while
# providing no evidence for another task, even when its score is high.
#
# **Wisdom:** restrict this result to a sentiment demonstration.
# The task mismatch makes checking label meaning more useful than treating a large score as success on our project question.
#
# **Better Question:** What evidence would justify using a pretrained model for disaster classification?
# **Current answer: not established here.** It would require task-aligned labels and
# independent, representative disaster evaluation with reviewed annotations and event relationships.
# This demonstration supplies neither; the question does not authorize another experiment.
#
# **⚖️ Ethical Takeaway:** Do not present sentiment confidence as evidence that a disaster occurred.

# <!--
# endregion
# region 5. Key Insights, Limitations & Recommendations
# -->

# %% [markdown]
# ## 5. Key Insights, Limitations & Recommendations
#
# **Main Question:** What can we responsibly conclude and recommend from this project?
# **Goal:** Separate measured classification results from claims that require new evidence.
#
# **Evidence and answer:** The baseline reaches class-1 F1 0.7308 on the retained holdout,
# while missing 208 of 637 recorded positives. Training-only CV compares six configurations;
# it does not establish reliability on new events. The Transformer lab demonstrates sentiment inference only.
#
# **Learning:** Data quality and evaluation boundaries determine what a successful run can establish.
# Labels remain annotations, excluded conflicts change the population, and different wording can describe the same event.
# The training-text lab also shows that reducing word forms and preserving meaning are different goals; no lemmatized classifier was evaluated.
#
# **Recommendation:** Preserve this measured baseline and report its limitations alongside its scores.
# Before broader use, review annotations and event relationships, then evaluate on fresh representative examples.
# No operational benefit or emergency-response readiness has been demonstrated.
#
# **Better Question:** Would performance hold on independently reviewed tweets from separate events?
# **Current answer:** Not established; this requires new evaluation evidence beyond the present experiment.
# **Ethical takeaway:** A model prediction should not be presented as verification that a disaster occurred.
# <!--
# endregion
# -->
