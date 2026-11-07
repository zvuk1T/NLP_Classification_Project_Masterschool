# %% [markdown]
# # Disaster Tweets — Project Overview
#
# **Problem:** Can tweet text distinguish recorded disaster and non-disaster examples?
#
# **🎯 Goal & Approach — WHY:** Build an understandable classifier and evaluate how trustworthy its predictions are.
#
# **📋 Source & Context:** MasterSchool DA112; an educational analysis of labeled tweets, not an emergency-response system.
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
# **What is the Question ❓** Which observations can support a fair evaluation?
#
# **🎯 Goal & Approach — WHY:** Audit labels and repeated texts, apply the agreed retention policy, then create the fixed stratified split.
#
# **✅ Evidence, Answer & Limits:** 7,613 source rows become 7,483 retained rows: 5,986 training and 1,497 holdout.
#
# **Limit / next step:** No shared original or prepared strings across the split does not prove independence between events.
#
# </details>
#
# <details>
# <summary><strong>2. Exploratory Data Analysis & Visualization</strong></summary>
#
# **What is the Question ❓** What patterns appear before modeling?
#
# **🎯 Goal & Approach — WHY:** Compare training-text distributions, then inspect tokenization, lemmatization, and information loss in five training examples.
#
# **✅ Evidence, Answer & Limits:** Median character lengths are 101 and 114 for classes 0 and 1; both have median 15 whitespace-separated parts.
# The separate language lab changes 15 of 97 tokens; its results do not change the classifier inputs.
#
# **Limit / next step:** Distributions overlap. These descriptive differences establish neither causes nor predictive performance.
#
# </details>
#
# <details>
# <summary><strong>3. Predictive Modeling, Comparison & Evaluation</strong></summary>
#
# **What is the Question ❓** How well does the classifier agree with unseen recorded labels?
#
# **🎯 Goal & Approach — WHY:** Evaluate the original baseline; separately compare six model configurations and three regularization settings in training-only CV.
#
# **✅ Evidence, Answer & Limits:** Baseline holdout F1 is 0.7308, precision 0.7989, and recall 0.6735; 208 recorded positives are missed.
#
# **Limit / next step:** CV is a post-baseline learning comparison. It provides no new independent holdout test or deployment guarantee.
#
# </details>
#
# <details>
# <summary><strong>4. Hugging Face: Pretrained Transformer Mini-Lab</strong></summary>
#
# **What is the Question ❓** What does a pretrained Transformer inference demonstrate?
#
# **🎯 Goal & Approach — WHY:** Run one local English sentiment prediction and contrast Hugging Face pipeline with scikit-learn Pipeline.
#
# **✅ Evidence, Answer & Limits:** The movie-review example returns POSITIVE with a score of 0.999887.
#
# **Limit / next step:** This score is not measured accuracy. Sentiment inference supplies no evidence of disaster-classification performance.
#
# </details>
#
# <details>
# <summary><strong>5. Key Insights, Limitations & Recommendations</strong></summary>
#
# **What is the Question ❓** What conclusions and next steps does the evidence justify?
#
# **🎯 Goal & Approach — WHY:** Connect observed scores, error examples, population limits, and the separate sentiment demonstration.
#
# **✅ Evidence, Answer & Limits:** Text supports measured agreement with retained annotations; real-world disaster-detection reliability remains unestablished.
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
# **What is the Question ❓** Can tweet text help distinguish disaster from non-disaster
# examples, and what would make evidence of that ability trustworthy?
#
# ### 📋 Source & Context
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
# ### What is the Question ❓
# What do the dataset structure and target distribution tell us before modeling?
#
# **🎯 Goal & Approach — WHY:** Inspect the source rows, check usable text and labels, then count each recorded class.
# Label proportions describe this population; they do not measure predictive ability.
#
# **✅ Notes for Students:** Open this repository, select `.venv/bin/python`, restart the kernel, and run cells top-to-bottom.
# README explains the installed packages and required local resources; missing resources stop execution without downloading replacements.
#
# **Step 1 — Confirm the runtime.** Check which Python executable and pandas version will read the data.

# %%
from pathlib import Path
import sys

import pandas as pd

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
print("pandas:", pd.__version__)

# %% [markdown]
# **Step 2 — Locate the input.** The next cell produces a file path, not a data table.
# A script uses its own folder; Interactive Window execution uses the current working folder.

# %%
# __file__ exists during script execution; resolve makes that script path absolute.
if "__file__" in globals():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
else:
    project_root = Path.cwd()

# Build the CSV path and stop if it is missing before attempting to read it.
data_path = project_root / "DATA" / "train.csv"
if not data_path.is_file():
    raise FileNotFoundError(
        f"Cannot find {data_path}. Start the Interactive Window in the repository root."
    )

print("CSV:", data_path)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `globals()` checks whether a script filename is available; `parent` selects that file's containing folder.
# `/` joins path components, while `is_file()` checks the resulting location.
# If the file is missing, inspect the working folder before continuing.
#
# </details>

# %% [markdown]
# **Step 3 — Read and preview the table.** Use the verified path to load one labeled tweet per row.

# %%
# Read one labeled observation per row; its index is the zero-based source position.
# Keep tweets unchanged so later subsets can be traced back to these original rows.
tweets = pd.read_csv(data_path)
print("Rows and columns:", tweets.shape)
print("Column types:")
print(tweets.dtypes.to_string())
print("First five observations (source indices shown; not a representative sample):")
print(tweets.head().to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `read_csv()` creates `tweets`; its default integer index records zero-based CSV row positions.
# `shape`, `dtypes`, and `head()` show structure and five examples, without establishing representativeness or data quality.
#
# </details>

# %% [markdown]
# **Step 4 — Inspect missing cells.** Count absence before interpreting or changing any values.

# %%
# isna marks individual missing cells; summing down each column counts them.
missing_cells = tweets.isna()
missing_counts = missing_cells.sum()
print("Missing values by column:")
print(missing_counts.to_string())

# %% [markdown]
# **Step 5 — Check empty text.** A non-missing string can still contain only spaces.

# %%
# Work on a temporary Series; the original text column remains unchanged.
trimmed_text = tweets["text"].str.strip()
blank_text_mask = trimmed_text.eq("")
blank_text_count = blank_text_mask.sum()
print("Blank text rows:", blank_text_count)

# %% [markdown]
# **Step 6 — Check label values.** Each observation must have recorded label 0 or 1 before grouping or modeling.

# %%
# Each flag describes one row; ~ reverses membership so invalid or missing labels become True.
valid_label_mask = tweets["target"].isin([0, 1])
invalid_label_mask = ~valid_label_mask
invalid_label_count = invalid_label_mask.sum()
print("Missing or invalid target rows:", invalid_label_count)

# %% [markdown]
# **Step 7 — Decide whether analysis can continue.** The checks above provide the evidence for one input safeguard.

# %%
# Name the three reasons for stopping before combining them into one decision.
has_missing_values = missing_counts.any()
has_blank_text = blank_text_count > 0
has_invalid_labels = invalid_label_count > 0
unusable_inputs = has_missing_values or has_blank_text or has_invalid_labels
if unusable_inputs:
    raise ValueError("Unexpected missing, blank, or invalid values; inspect before continuing.")

# %% [markdown]
# **Step 8 — Summarize recorded classes.** With usable fields, count each label and calculate its share of the original rows.

# %%
# Count labels, then order the Series index as 0 and 1.
target_counts = tweets["target"].value_counts()
target_counts = target_counts.sort_index()

# The summary has one row per class; divide each count by the full source population.
target_summary = target_counts.to_frame(name="rows")
target_shares = target_counts / len(tweets)
target_percentages = target_shares * 100
target_summary["percent"] = target_percentages.round(2)
print("Original label distribution:")
print(target_summary.to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `value_counts()` counts observations; `to_frame()` makes those counts a named table column.
# Each percentage uses all original rows as its denominator; rounding changes only the reported precision.
# The earlier checks establish usable fields under our rules, not correct annotations or distinct examples.
#
# </details>

# %% [markdown]
# ### Visual evidence — recorded class balance
# A bar chart makes the count difference visible; printed percentages keep its size explicit.
# These are annotations in this supplied table, not disaster prevalence in the world.
# DataCamp's [categorical bar-chart exercise](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=2)
# motivates aggregation before plotting; the implementation below is our Plotly adaptation.

# %%
import plotly.express as px

# Turn the class index into a plotting column; each row still summarizes one recorded class.
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
# <details><summary>💡 Explain this code</summary>
#
# `reset_index()` exposes class labels as a column; `map()` gives them readable names.
# `px.bar()` displays the existing counts, with percentages in hover details.
# The denominator is the supplied dataset, not all real-world disaster reports.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# The supplied table has 7,613 rows and two columns, with no missing values,
# blank texts, or invalid labels. Class `0` contains 4,342 rows (57.03%);
# class `1` contains 3,271 (42.97%). Non-disaster annotations are moderately more common.
#
# ### 🧠 What We Learned
# Usable fields and known class proportions establish a starting population.
# They do not establish that each row adds a distinct example or a reliable answer.
#
# **What changes when we count texts rather than rows, and inspect label agreement?**

# %% [markdown]
# ### 1.2 Distinguish repeated texts from conflicting annotations
#
# ### What is the Question ❓
# How much repetition and label disagreement does exact text equality reveal?
#
# **🎯 Goal & Approach — WHY:** Group identical strings to distinguish
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
# Count source rows per exact string; the index contains texts, not source-row positions.
text_counts = tweets["text"].value_counts()
repeated_text_counts = text_counts[text_counts > 1]

# Later occurrences repeat text; whole-row duplicates additionally require the same label.
extra_text_rows = tweets["text"].duplicated(keep="first").sum()
extra_identical_rows = tweets.duplicated(keep="first").sum()
print("Unique exact texts:", len(text_counts))
print("Repeated exact-text groups:", len(repeated_text_counts))
print("Rows belonging to repeated groups:", repeated_text_counts.sum())
print("Additional exact-text occurrences:", extra_text_rows)
print("Additional identical (text, target) rows:", extra_identical_rows)

# %% [markdown]
# The 69 repeated groups contain 179 rows, including 110 occurrences beyond the first.
# Only 92 extra rows also repeat the label; text repetition and whole-row repetition therefore differ.
# Next, count distinct labels within each text group to locate disagreements.

# %%
# One value counts distinct labels for one exact text; a count above one means disagreement.
labels_per_text = tweets.groupby("text")["target"].nunique()
conflicting_texts = labels_per_text[labels_per_text > 1].index

# Map those text keys back to one flag per source row, including every conflicting group member.
exact_conflict_mask = tweets["text"].isin(conflicting_texts)
conflicting_row_count = exact_conflict_mask.sum()
print("Exact-text groups with both labels:", len(conflicting_texts))
print("Rows belonging to conflicting groups:", conflicting_row_count)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Grouped `nunique()` counts labels per text, rather than rows per text.
# `.index` extracts conflicting text keys; `isin()` returns a membership flag for every original row.
# These checks identify disagreements without deleting rows or choosing a correct annotation.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# Eighteen exact-text groups contain both labels, involving 55 observations.
# Identical inputs can therefore provide contradictory supervision; these counts cannot identify the correct label or explain the disagreement.
#
# ### 🧠 What We Learned
# Complete fields do not establish distinct inputs or consistent labels.
# Before applying the retention policy, check whether normalization makes previously different strings equal.

#
# ### 🤔 Question Yourself
# In the earlier three-row example, identical texts have labels `0, 0, 1`.
# How many additional text occurrences and distinct labels are there? How does that differ from identical whole-row duplicates?
#
# <details><summary>💡 Show answer</summary>
#
# There are three rows, two additional text occurrences, and two distinct labels.
# Only one additional `(text, target)` row is identical, because the third label differs.
# The group has conflicting annotations; these counts do not determine the correct label.
#
# </details>

# %% [markdown]
# ### 1.3 Understand what normalization changes
#
# ### What is the Question ❓
# Which observations become equivalent under our chosen text preparation?
#
# **🎯 Goal & Approach — WHY:** Compare the original text with a copy
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
# Preserve original text and source indices; the new column defines the model input and duplicate key.
prepared = tweets.copy()
prepared["prepared_text"] = prepared["text"].str.lower()
# \s+ matches a run of whitespace; replace it with one space, then trim both ends.
prepared["prepared_text"] = prepared["prepared_text"].str.replace(r"\s+", " ", regex=True)
prepared["prepared_text"] = prepared["prepared_text"].str.strip()
print("Original/prepared example (source index retained):")
print(prepared[["text", "prepared_text", "target"]].head(1).to_string())

# %% [markdown]
# `prepared_text` changes case and whitespace while retaining the original text beside it.
# We now group by that new key, because the classifier will receive those same prepared strings.

# %%
# Each Series uses normalized text keys: values count distinct labels or distinct original forms.
labels_per_prepared_text = prepared.groupby("prepared_text")["target"].nunique()
original_forms_per_key = prepared.groupby("prepared_text")["text"].nunique()
collision_keys = original_forms_per_key[original_forms_per_key > 1].index
conflicting_keys = labels_per_prepared_text[labels_per_prepared_text > 1].index
print("Distinct normalized texts before exclusions:", len(labels_per_prepared_text))
print("Normalized keys merging distinct original texts:", len(collision_keys))
print("Normalized groups with both labels:", len(conflicting_keys))

# %% [markdown]
# Normalization leaves 7,501 distinct keys and merges two pairs of original strings.
# A merge need not create a label conflict; inspect the source rows before using these group keys for exclusion.

# %%
# isin maps group keys back to source rows; loc keeps every matching row and its original index.
collision_mask = prepared["prepared_text"].isin(collision_keys)
print("Rows in normalization collisions (source indices retained):")
print(prepared.loc[collision_mask].to_string())

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# Rows 3340 and 7604 differ in whitespace; row 6119 adds a case variant to an existing conflicting group.
# Conflict membership increases from 55 to 56 rows, while the number of conflicting groups remains 18.
# Source indices identify zero-based CSV row positions.
#
# ### 🧠 What We Learned
# Equality depends on the representation used.
# The retention policy must use `conflicting_keys` from prepared text, so it also excludes the newly equivalent group member.

#
# ### 🤔 Question Yourself
# Two different original strings become equal after normalization.
# Why is checking duplicates and label conflicts only in the original strings insufficient?
#
# <details><summary>💡 Show answer</summary>
#
# The classifier receives prepared strings, so different originals can become identical model inputs.
# Grouping by `prepared_text` can reveal repetition or conflicts that exact original-string comparisons miss.
# The retention policy therefore uses prepared keys while preserving original text for inspection.
#
# </details>

# %% [markdown]
# ### 1.4 Apply the retention policy and account for the population
#
# ### What is the Question ❓
# Which observations remain under the agreed policy, and what does exclusion cost?
#
# **🎯 Goal & Approach — WHY:** Exclude every row in a normalized group
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
# Map conflicting text keys to source rows; ~ keeps every row outside those groups.
conflict_mask = prepared["prepared_text"].isin(conflicting_keys)
conflicting_rows = prepared.loc[conflict_mask]
consistent_rows = prepared.loc[~conflict_mask]
print("Excluded conflicting rows:", len(conflicting_rows))
print("Rows available for repeat checking:", len(consistent_rows))

# %% [markdown]
# Excluding all 56 conflicting rows leaves 7,557 rows with consistent labels within each normalized group.
# Those groups can still repeat, so use `consistent_rows` for the next check.

# %%
# True marks occurrences after each first prepared_text; filtering preserves source order.
repeat_mask = consistent_rows.duplicated(subset="prepared_text", keep="first")
redundant_rows = consistent_rows.loc[repeat_mask]
# Keep one row per surviving input, retaining original text, label and source index.
model_tweets = consistent_rows.loc[~repeat_mask].copy()
print("Excluded later repeats:", len(redundant_rows))
print("Retained rows:", len(model_tweets))
print(model_tweets[["prepared_text", "target"]].head(3).to_string())

# %% [markdown]
# Removing 74 later repeats leaves 7,483 rows; index gaps retain the link to their original CSV positions.
# Before splitting `model_tweets`, account for all removed and retained rows by recorded label.

# %%
# One record describes one disjoint outcome, except original which supplies the starting total.
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
preparation_counts = pd.DataFrame(stage_records).set_index("stage")
print("Rows by stage and target label:")
print(preparation_counts.to_string())

# %% [markdown]
# Conflicts remove 29 label-0 and 27 label-1 rows; repeats remove another 16 and 58, respectively.
# The prepared row contains 4,297 label-0 and 3,186 label-1 observations.
# Check that these outcomes account for every source row and leave no repeated or conflicting inputs.

# %%
# Subtract the disjoint removed and retained populations; zero means every source row is accounted for.
unaccounted_rows = len(tweets) - len(conflicting_rows) - len(redundant_rows) - len(model_tweets)
remaining_duplicates = model_tweets["prepared_text"].duplicated().sum()
# Group-level counts check surviving label disagreement separately from repeated rows.
remaining_label_counts = model_tweets.groupby("prepared_text")["target"].nunique()
remaining_conflicts = (remaining_label_counts > 1).sum()
print("Unaccounted source rows:", unaccounted_rows)
print("Remaining extra occurrences / conflicting groups:", remaining_duplicates, remaining_conflicts)

# %% [markdown]
# All three checks return zero, so the retained table satisfies the agreed accounting and equality rules.
# The following chart uses `preparation_counts`; the later split uses `model_tweets`.

# %% [markdown]
# ### Visual evidence — what the retention policy removes
# Compare retained and excluded counts in separate panels because their scales differ greatly.
# Exclusion reasons are disjoint; stacked colors show each reason's recorded label composition.
# Compare counts using each panel's axis, not bar heights across panels.

# %%
# Reshape label columns into plotting rows: one stage/class pair and its count per row.
eda_retention = preparation_counts.loc[["prepared", "conflicts excluded", "repeats excluded"], ["label_0", "label_1"]]
eda_retention = eda_retention.reset_index()
eda_retention = eda_retention.melt(id_vars="stage", var_name="class", value_name="rows")
eda_retention["class"] = eda_retention["class"].map({"label_0": "0 — Non-disaster", "label_1": "1 — Disaster"})
eda_retention["panel"] = eda_retention["stage"].map({
    "prepared": "Retained", "conflicts excluded": "Excluded", "repeats excluded": "Excluded",
})
print("Chart input: one row per outcome and recorded class:")
print(eda_retention.to_string(index=False))

# %% [markdown]
# Six rows represent three policy outcomes crossed with two recorded classes.
# Plot these existing counts in separate retained/excluded panels so the smaller exclusions remain visible.

# %%
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
# <details><summary>💡 Explain this code</summary>
#
# `reset_index()` exposes stage names; `melt()` converts the two label columns into stage/class/count rows for plotting.
# Separate panels keep small exclusion counts visible beside the retained population.
# Each panel has its own vertical scale, so equal bar heights across panels do not mean equal counts.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# The retained population contains 7,483 rows, with class 1 representing 42.58%, compared with 42.97% originally.
# The preceding checks account for all rows and find no remaining duplicate inputs or conflicting groups.
#
# > **⚠️ Watch Out — Excluding conflicts changes who the score describes**
# >
# > The policy removes 56 conflicting rows and 74 repeats before evaluation; a small change in class balance does not make exclusions neutral.
# > Agreement among retained labels does not establish that those labels are correct or that excluded cases would be classified reliably.
#
# ### 🧠 What We Learned
# Mapping group decisions back to all their rows avoids choosing an arbitrary label from a disputed group.
# The later score describes held-out examples drawn from the retained population; performance on excluded observations remains unestablished.
# Use this retained table for the fixed split below.
#
# **⚖️ Ethical Takeaway:** Disclose exclusions so readers do not mistake results
# on retained tweets for evidence covering every original example.

# %% [markdown]
# ### 1.5 Separate training and holdout observations
#
# ### What is the Question ❓
# What does our training/holdout separation preserve, and what remains uncertain?
#
# **🎯 Goal & Approach — WHY:** Reserve 20% of the prepared rows for
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

# Partition whole observations before fitting; returned tables keep source indices in split order.
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

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `train_test_split()` reserves 20% of the retained rows and approximately preserves label proportions using the fixed seed.
# Selecting text and labels from each returned table keeps positions and source indices aligned; partition percentages use their own denominators.
# Returned rows are not sorted by source index, and stratification does not keep related events together.
#
# </details>

# %%
# Sets keep distinct text values; & finds shared strings rather than shared source indices.
original_overlap = set(train_rows["text"]) & set(holdout_rows["text"])
prepared_overlap = set(X_train) & set(X_holdout)
print("Partition rows / prepared rows:", len(train_rows) + len(holdout_rows), len(model_tweets))
print("Shared original texts:", len(original_overlap))
print("Shared prepared texts:", len(prepared_overlap))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `set()` keeps distinct strings; `&` finds strings present in both training and holdout, and `len()` counts that overlap.
# The two comparisons check original text and the prepared model input separately.
# Zero overlap rules out identical strings under those definitions, not related events or near-duplicates.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
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
# ### 🧠 What We Learned
# Row counts, distinct inputs and consistently labeled inputs measure different things.
# Excluding rows changes the population our scores can describe.
# Generalization to new events or excluded examples is not established; it needs
# event relationships, annotation review and fresh representative examples.
#
# > **⚠️ Watch Out — Different texts can describe the same event**
# >
# > Zero shared strings rules out exact overlap under our comparisons; it does not establish event independence or annotation accuracy.
# > Stratification preserves class proportions but cannot prevent related reports from crossing partitions.
#
# Any later learned text transformation must be fitted on training data only.
# The holdout is reserved for evaluation, not tuning; no model is fitted in this preparation section.
#
# ### 🤔 Question Yourself
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
# ### What is the Question ❓
# How do text-length distributions compare between recorded classes in the training subset?
#
# **🎯 Goal & Approach — WHY:** Compare characters and whitespace-separated parts to distinguish two meanings of length.
# Use the existing training rows only; this retrospective exploration changes no model features or preparation decisions.
#
# **🔎 Guiding Questions:** Do typical lengths differ, and would a difference alone establish useful classification?
#
# **💻 Code / Method — HOW:** Count characters in original text, including spaces and punctuation; split on whitespace for a simple second measure.
# These parts are not linguistic words or the tokens used by either TF-IDF or the Transformer.
# DataCamp's [grouped histograms](https://campus.datacamp.com/courses/data-manipulation-with-pandas/creating-and-visualizing-dataframes?ex=5)
# and [EDA histograms](https://campus.datacamp.com/courses/exploratory-data-analysis-in-python/getting-to-know-a-dataset?ex=4)
# motivate distribution comparisons; this is an original adaptation to our training tweets.

# %%
# Keep training source indices in a separate descriptive table; these lengths do not become model features.
eda_lengths = train_rows[["target"]].copy()
eda_lengths["class"] = eda_lengths["target"].map({0: "0 — Non-disaster", 1: "1 — Disaster"})
eda_lengths["characters"] = train_rows["text"].str.len()
eda_lengths["whitespace_parts"] = train_rows["text"].str.split().str.len()
print("Length measures for three training observations:")
print(eda_lengths.head(3).to_string())

# %% [markdown]
# Each source-indexed row now has two measures: characters and whitespace-separated parts.
# Group these same rows by recorded class to compare their typical lengths and ranges.

# %%

# Select the two length measures, then summarize each class separately.
# The grouped object holds rows by class; agg creates the final summary table.
eda_length_columns = ["characters", "whitespace_parts"]
eda_lengths_by_class = eda_lengths.groupby("class")[eda_length_columns]
eda_length_summary = eda_lengths_by_class.agg(["count", "min", "median", "max"])
print("Training-only original-text lengths:")
print(eda_length_summary.to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `groupby()` and `agg()` summarize each class; two-level column names identify the measure and statistic.
# Median character length is 101 for class 0 and 114 for class 1; both have median 15 whitespace-separated parts.
# The charts below show the variation those medians hide.
#
# </details>

# %% [markdown]
# ### Character length — shape and overlap
# Both classes use the same 10-character intervals and axes, covering every observed length.
# Percentages use each class's own row count, preventing the larger class from dominating through sample size alone.
# Bins summarize a distribution; their boundaries affect its appearance and are not learned classification thresholds.

# %%
# Normalize each class trace separately so different sample sizes do not dominate the histogram.
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
# <details><summary>💡 Explain this code</summary>
#
# `histnorm="percent"` expresses each bin as a share of its class trace; separate panels and shared axes support comparison.
# Fixed 10-character bins make the display reproducible.
# The vertical axis is a percentage, and bin edges are display choices rather than learned classification thresholds.
#
# </details>

# %% [markdown]
# ### Whitespace-separated parts — a complementary distribution
# A box plot shows the median and middle half of observations, with unusual values beyond its whiskers.
# Whiskers reach observed values within 1.5 times the interquartile range beyond each box edge.
# The interquartile range is the width of the middle half; points beyond whiskers are not automatically errors.
# This plot removes no rows.
# We use the linear quartile convention explicitly so the displayed summary has a defined calculation.

# %%
# Summarize per-tweet part counts by class; displayed outliers remain included in the data.
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
# <details><summary>💡 Explain this code</summary>
#
# `px.box()` groups training-text lengths by class; `quartilemethod="linear"` fixes quartile interpolation.
# `points="outliers"` shows observations beyond the whiskers without removing them.
# An outside point is not automatically an error, and the chart changes no model inputs.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
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
# ### 🧠 What We Learned
# Length depends on its definition; inspecting only an average or median would hide variation within each class.
# The cause of the character-length difference is not established. Inspecting part lengths,
# URLs and annotation context in training data could investigate it.
#
# **⚖️ Ethical Takeaway:** Longer text is not evidence that a report describes a real emergency.

#
# ### 🤔 Question Yourself
# The recorded classes contain different numbers of training examples.
# Why do the character-length histograms use percentages within each class rather than raw counts?
#
# <details><summary>💡 Show answer</summary>
#
# Each bin is divided by its own class total, so bar heights compare the share of each class at those lengths.
# Raw counts would also reflect the different class sizes.
# Shared bins and axes make these distribution shapes comparable; they do not establish predictive performance.
#
# </details>

# %% [markdown]
# ### Text Preprocessing: Tokenization & Lemmatization
#
# ### What is the Question ❓
# How do tokenization and lemmatization change selected training tweets, and what information could cleaning erase?
#
# **🎯 Goal & Approach — WHY:** Make language transformations visible before deciding whether they belong in a classifier.
# Inspect five reproducibly sampled training tweets in a separate table; keep the established model inputs and evaluation unchanged.
#
# **🔎 Guiding Questions:** Which units become tokens, which forms change, and does a smaller vocabulary establish better predictions?
#
# **📋 Source & Context:** This is an original retrospective adaptation of school methods and DataCamp teaching examples.
# Source locations below use one-based notebook cell numbers and PDF page numbers; saved school outputs are not our results.
#
# <details><summary>📋 Teaching sources and adaptations</summary>
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
# </details>
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
# **Setup:** `nltk==3.10.4` reads English `punkt_tab`, `wordnet`, `averaged_perceptron_tagger_eng`, and `stopwords` from `.cache/nltk_data/`.
# These separately prepared resources are data files, not Python packages; the analysis never downloads them.
# Restart the Interactive Window kernel before running all cells, so an earlier session cannot retain globally loaded corpora.
# README describes the project-local layout and the separate setup boundary when resources are missing.
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

# Restrict NLTK lookup to existing project resources; this language lab leaves model inputs unchanged.
nltk_data_path = project_root / ".cache" / "nltk_data"
nltk.data.path = [str(nltk_data_path)]
nltk_resources = [
    "tokenizers/punkt_tab/english/", "corpora/wordnet.zip",
    "taggers/averaged_perceptron_tagger_eng/", "corpora/stopwords/english",
]
for resource in nltk_resources:
    try:
        nltk.data.find(resource)
    except LookupError as error:
        raise FileNotFoundError(
            f"Missing NLTK resource {resource} in {nltk_data_path}. "
            "See README for separate resource setup; nothing was downloaded."
        ) from error
print("NLTK resource directory:", nltk_data_path)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `nltk.data.path` restricts lookup to the project's existing resource directory.
# The loop checks each required resource; a missing resource stops execution without downloading a replacement.
# These files support tokenization and dictionary lookup, not disaster-label learning.
#
# </details>

# %% [markdown]
# Select five training examples before inspecting any transformations.
# Their original CSV indices let us trace every later token back to its tweet.

# %%
# Sample without looking at labels or model errors; retain CSV source indices for traceability.
# Five observations illustrate behavior but cannot estimate its prevalence across the dataset.
nlp_example_columns = ["text", "prepared_text"]
nlp_example_candidates = train_rows[nlp_example_columns]
nlp_examples = nlp_example_candidates.sample(n=5, random_state=42)
nlp_examples = nlp_examples.sort_index()
print("\nLanguage lab: NLTK", nltk.__version__)
print("Five training examples, indexed by source row:")
print(nlp_examples[["text"]].to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `sample()` chooses five rows reproducibly; `sort_index()` then orders their source indices without changing membership.
# Each row is still a training tweet with its original and prepared text.
# This small sample illustrates behavior rather than measuring dataset-wide prevalence.
#
# </details>

# %% [markdown]
# Compare two tokenizers on one of those tweets before processing the whole sample.

# %%
# Preserve case for grammatical tagging and keep handles/repeated characters as configured.
# The example is the second sampled row in source-index order, not a chosen model error.
nlp_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
nlp_example_text = nlp_examples["text"].iloc[1]
nlp_example_tokens = nlp_tokenizer.tokenize(nlp_example_text)
print("\nTokenization example, source row:", nlp_examples.index[1])
print("School word_tokenize:", word_tokenize(nlp_example_text))
print("TweetTokenizer:", nlp_example_tokens)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `.iloc[1]` selects the second sampled row by position; its source index remains available separately.
# `nlp_example_tokens` is an ordered list of text units, including special tokens retained by `TweetTokenizer`.
# Here `word_tokenize` separates hashtag markers; `TweetTokenizer` retains hashtag units and the URL for the teaching steps below.
#
# </details>

# %% [markdown]
# Compare three transformations on the same illustrative words, adding one result column at a time.
# These examples explain the methods; they are not dataset measurements.

# %%
# Create the two tools and a three-row input table; the supplied roles mean noun, verb, and adjective.
nlp_lemmatizer = WordNetLemmatizer()
nlp_stemmer = PorterStemmer()
nlp_method_table = pd.DataFrame({
    "word": ["floods", "running", "better"],
    "specified POS": ["n", "v", "a"],
})
print(nlp_method_table.to_string(index=False))

# %% [markdown]
# **Stemming:** Apply shortening rules to each word, then inspect the new column before dictionary lookup.

# %%
# Collect one stem per input word in the same row order.
nlp_stems = []
for word in nlp_method_table["word"]:
    stem = nlp_stemmer.stem(word)
    nlp_stems.append(stem)
nlp_method_table["Porter stem"] = nlp_stems
print(nlp_method_table[["word", "Porter stem"]].to_string(index=False))

# %% [markdown]
# `floods` becomes `flood`, and `running` becomes `run`; `better` stays unchanged.
# These shortening rules use no grammatical role. Compare them with dictionary lookup next.
#
# **Default lemmatization:** WordNet assumes a noun when no grammatical role is supplied.
# Compare this result with the stem before changing the grammatical input.

# %%
# Keep one default lemma per word; this call uses no surrounding sentence.
nlp_default_lemmas = []
for word in nlp_method_table["word"]:
    lemma = nlp_lemmatizer.lemmatize(word)
    nlp_default_lemmas.append(lemma)
nlp_method_table["noun-default lemma"] = nlp_default_lemmas
print(nlp_method_table[["word", "noun-default lemma"]].to_string(index=False))

# %% [markdown]
# Default lookup changes `floods` to `flood` but leaves `running` unchanged.
# Supply the grammatical roles from the original input table to see whether that changes the result.
#
# **Lemmatization with a supplied role:** Pair each word with its noun, verb, or adjective code from the input table.
# DataCamp's [Loop over DataFrame](https://campus.datacamp.com/courses/intermediate-python/loops?ex=14) introduces this row-by-row pattern.

# %%
# iterrows supplies a row label and its values; _ means the table index is not needed here.
nlp_pos_lemmas = []
for _, example in nlp_method_table.iterrows():
    word = example["word"]
    wordnet_pos = example["specified POS"]
    lemma = nlp_lemmatizer.lemmatize(word, pos=wordnet_pos)
    nlp_pos_lemmas.append(lemma)
nlp_method_table["POS lemma"] = nlp_pos_lemmas

# Order the comparison columns; each row describes one illustrative word.
nlp_method_columns = [
    "word", "Porter stem", "noun-default lemma", "specified POS", "POS lemma",
]
nlp_method_table = nlp_method_table[nlp_method_columns]
print("\nIllustrative words, not dataset measurements:")
print(nlp_method_table.to_string(index=False))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# With supplied roles, verb `running` becomes `run` and adjective `better` becomes `good`.
# Supplying a role changes dictionary lookup; WordNet does not infer that role from a sentence.
# For real tweets, estimate roles from complete token sequences before applying lookup.
#
# </details>

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
# Tag the one-tweet example before lowercasing; each pair contains a token and its estimated grammatical role.
nlp_example_tagged_tokens = nltk.pos_tag(nlp_example_tokens)
print("\nToken / POS pairs for the same example:")
print(nlp_example_tagged_tokens)

# Translate Treebank prefixes into WordNet's adjective, noun, verb, and adverb codes.
# This fixed dictionary uses no dataset labels; unsupported prefixes have no mapped role.
nlp_pos_map = {"J": "a", "N": "n", "V": "v", "R": "r"}

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `pos_tag()` estimates a role from the whole token sequence, preserving case and surrounding words as context.
# The dictionary translates the first tag letter into a convention understood by WordNet; it does not estimate roles itself.
# Tags are estimates, so they support an inspectable illustration rather than a validated cleaning policy.
#
# </details>

# %% [markdown]
# **Step 1 — Tokenize the five tweets.** Keep each source index beside its ordered token list.

# %%
# This dictionary maps one CSV source index to one tweet's tokens; the next cell tags those same lists.
nlp_tokens_by_source = {}
for source_row, original_text in nlp_examples["text"].items():
    nlp_tokens = nlp_tokenizer.tokenize(original_text)
    nlp_tokens_by_source[source_row] = nlp_tokens
print("Example token list:")
print(nlp_tokens_by_source[nlp_examples.index[1]])

# %% [markdown]
# The preview retains hashtag units and a URL; these are tokens alongside ordinary words.
# Pass the same ordered lists to the tagger so surrounding tokens remain available as context.
#
# **Step 2 — Tag each complete sequence.** Reuse the token lists before lowercasing, preserving sentence context for POS estimation.

# %%
# Each dictionary value now contains ordered (token, estimated tag) pairs for one tweet.
nlp_tags_by_source = {}
for source_row, nlp_tokens in nlp_tokens_by_source.items():
    nlp_tagged_tokens = nltk.pos_tag(nlp_tokens)
    nlp_tags_by_source[source_row] = nlp_tagged_tokens
print("Example token / POS pairs:")
print(nlp_tags_by_source[nlp_examples.index[1]])

# %% [markdown]
# Each pair now contains the original token and an estimated grammatical tag.
# The next table keeps those pairs beside their tweet and token positions.
#
# **Step 3 — Make one row per token.** Convert the five tagged sequences into a table before applying any lemmatization rule.

# %%
# The outer loop selects a tweet; enumerate supplies each token's position inside that tweet.
nlp_tagged_records = []
for source_row, nlp_tagged_tokens in nlp_tags_by_source.items():
    for position, tagged_token in enumerate(nlp_tagged_tokens):
        token, tag = tagged_token
        nlp_tagged_records.append({
            "source_row": source_row,
            "position": position,
            "token": token,
            "POS": tag,
        })

# The table index counts token records; source_row still identifies the original CSV observation.
nlp_tagged_table = pd.DataFrame(nlp_tagged_records)
print(nlp_tagged_table.head(10).to_string(index=False))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The grain changes from one token sequence per tweet to one row per token occurrence.
# `source_row` and `position` preserve both the tweet link and the within-tweet order.
# The first ten rows are a preview; the full table remains available for the next steps.
#
# </details>

# %% [markdown]
# **Step 4 — Prepare dictionary inputs.** Keep the tagged table and add lowercased forms and translated WordNet roles in a working copy.

# %%
# Each new column still describes the same token row; map leaves unsupported tag prefixes missing.
nlp_lemma_steps = nlp_tagged_table.copy()
nlp_lemma_steps["lowercase"] = nlp_lemma_steps["token"].str.lower()
nlp_lemma_steps["tag_prefix"] = nlp_lemma_steps["POS"].str[0]
nlp_lemma_steps["wordnet_pos"] = nlp_lemma_steps["tag_prefix"].map(nlp_pos_map)
nlp_role_columns = ["token", "POS", "lowercase", "wordnet_pos"]
print(nlp_lemma_steps[nlp_role_columns].head(10).to_string(index=False))

# %% [markdown]
# `wordnet_pos` translates supported tag prefixes; punctuation and other unsupported tags have no mapped role.
# Keep those rows, then mark whether lookup is appropriate for each token.
#
# **Step 5 — Decide which tokens qualify.** Inspect three Boolean columns before combining them into one eligibility flag.

# %%
# Each column contains one True/False decision per token; NNP and NNPS indicate estimated proper names.
nlp_lemma_steps["is_alphabetic"] = nlp_lemma_steps["token"].str.isalpha()
nlp_lemma_steps["is_proper_name"] = nlp_lemma_steps["POS"].str.startswith("NNP")
nlp_lemma_steps["has_wordnet_pos"] = nlp_lemma_steps["wordnet_pos"].notna()
nlp_rule_columns = ["token", "is_alphabetic", "is_proper_name", "has_wordnet_pos"]
print(nlp_lemma_steps[nlp_rule_columns].head(10).to_string(index=False))

# %% [markdown]
# The three flags describe separate conditions; a supported role alone does not permit lookup for an estimated proper name.
# Combine the flags into one decision while keeping every token row.

# %%

# & requires all row conditions; ~ reverses the proper-name flag, preserving those names from dictionary lookup.
nlp_lemma_steps["eligible"] = (
    nlp_lemma_steps["is_alphabetic"]
    & ~nlp_lemma_steps["is_proper_name"]
    & nlp_lemma_steps["has_wordnet_pos"]
)
print(nlp_lemma_steps[["token", "eligible"]].head(10).to_string(index=False))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Unlike `and` for individual values, `&` combines whole Boolean columns row by row.
# A token qualifies only when it is alphabetic, is not tagged as a proper name, and has a supported WordNet role.
# These are the existing teaching rules; estimated tags can still be wrong.
#
# </details>

# %% [markdown]
# **Step 6 — Look up eligible lemmas.** Read the prepared rows and keep the lowercased form whenever lookup is not allowed.

# %%
# Collect one result per token in table order; no tweet is dropped or passed to the classifier.
nlp_lemma_values = []
for _, token_row in nlp_lemma_steps.iterrows():
    lower_token = token_row["lowercase"]
    lemma = lower_token
    if token_row["eligible"]:
        wordnet_pos = token_row["wordnet_pos"]
        lemma = nlp_lemmatizer.lemmatize(lower_token, pos=wordnet_pos)
    nlp_lemma_values.append(lemma)
nlp_lemma_steps["lemma"] = nlp_lemma_values

# Compare against the lowercased form, so case-only differences do not count as lemmatization changes.
nlp_lemma_steps["changed"] = nlp_lemma_steps["lemma"] != nlp_lemma_steps["lowercase"]
print(nlp_lemma_steps[["token", "lemma", "changed"]].head(10).to_string(index=False))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `iterrows()` provides one prepared token row at a time; its eligibility flag controls the dictionary call.
# Appending in row order keeps the lemma list aligned with the table when it becomes a column.
# `changed` measures dictionary-lookup effects beyond lowercasing, not classifier improvement.
#
# </details>

# %% [markdown]
# **Step 7 — Inspect the completed comparison.** Keep the original result columns, then select changed occurrences for review.

# %%
# Token-table indices are record positions; source_row and position retain the original mapping.
# The changed-only view excludes case changes, so it isolates the dictionary-lookup effect.
nlp_result_columns = [
    "source_row", "position", "token", "lowercase", "POS", "lemma", "changed",
]
nlp_token_table = nlp_lemma_steps[nlp_result_columns].copy()
nlp_changed_mask = nlp_token_table["changed"]
nlp_changed_columns = ["source_row", "token", "POS", "lemma"]
nlp_changed_tokens = nlp_token_table.loc[nlp_changed_mask, nlp_changed_columns]
print("\nAll changed tokens in the five-example lab:")
print(nlp_changed_tokens.to_string(index=False))
print("Token occurrences / changed occurrences:", len(nlp_token_table), len(nlp_changed_tokens))
print("Distinct lowercased forms / lemmas:", nlp_token_table["lowercase"].nunique(), nlp_token_table["lemma"].nunique())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Across 97 token occurrences, 15 change beyond lowercasing; distinct forms decrease from 89 to 87.
# The changed-only table includes `driven → drive`, `spreading → spread` and `tracks → track`, with source indices retained.
# Reuse the full token table to reconstruct each tweet and count changes by source row.
#
# </details>

# %% [markdown]
# Reconstruct one display string per tweet to inspect the transformation in context.

# %%
# Grouping preserves token order within each tweet; joining is for inspection, not model input.
# These strings may contain new spaces around punctuation and are not exact original text.
nlp_lemmas_by_source = nlp_token_table.groupby("source_row", sort=False)["lemma"]
nlp_reconstructed = nlp_lemmas_by_source.agg(" ".join)
print("\nReconstructed lemma strings (teaching output only):")
print(nlp_reconstructed.to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `groupby("source_row", sort=False)` collects lemmas while retaining tweet and token order.
# `" ".join` creates one display string per tweet, adding spaces between tokens.
# This does not recover original spacing or punctuation, and the strings are not classifier inputs.
#
# </details>

# %% [markdown]
# ### Visual evidence — count changes within each sampled tweet
# Stack changed and unchanged token occurrences so each bar totals that tweet's token count.
# The denominator includes punctuation, URLs, and hashtags retained by this tokenizer; this is not a count of linguistic words.

# %%
# Add readable labels in a separate plotting table, preserving the original token table.
nlp_plot_tokens = nlp_token_table.copy()
nlp_plot_tokens["status"] = nlp_plot_tokens["changed"].map({
    False: "Unchanged",
    True: "Changed by lemmatization",
})

# Group token occurrences, count each group, then turn its index into plotting columns.
nlp_tokens_by_status = nlp_plot_tokens.groupby(["source_row", "status"])
nlp_status_counts = nlp_tokens_by_status.size()
nlp_change_counts = nlp_status_counts.reset_index(name="tokens")
nlp_change_counts["source_row"] = nlp_change_counts["source_row"].astype(str)
print("\nToken counts by sampled tweet and lookup result:")
print(nlp_change_counts.to_string(index=False))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The copy adds readable status labels without changing the original token table.
# Grouping counts occurrences for each tweet/status pair; `reset_index()` exposes those group keys as columns.
# Source identifiers become strings so the chart treats them as categories, not a numerical scale.
#
# </details>

# %% [markdown]
# Plot the inspected counts; each stacked bar retains that tweet's full token-occurrence denominator.

# %%
# The chart displays the existing grouped counts; it does not reprocess text or learn a model.
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
# <details><summary>💡 Explain this code</summary>
#
# Each stacked bar counts changed and unchanged token occurrences for one sampled tweet.
# Its total includes retained punctuation and special tokens; case-only changes are excluded.
# These counts describe transformations, not vocabulary size or classifier improvement.
#
# </details>

# %% [markdown]
# ### Inspect information loss before calling text “clean”
# DataCamp's [stop-word exercise](https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/text-processing-fundamentals?ex=5)
# and the school notebooks filter common words using an English list.
# We test that operation on an original illustrative sentence because its negation has a clear role.

# %%
# Illustrative text only; training and holdout inputs stay unchanged.
nlp_negation_example = "Residents are not trapped by the floods."
# Supplied English stop words; alphabetic tokens retain sentence order.
nlp_stop_words = set(stopwords.words("english"))
nlp_negation_tokens = [word.lower() for word in word_tokenize(nlp_negation_example) if word.isalpha()]
# Partition tokens by membership, not meaning.
nlp_filtered_tokens = [word for word in nlp_negation_tokens if word not in nlp_stop_words]
nlp_removed_tokens = [word for word in nlp_negation_tokens if word in nlp_stop_words]
print("\nIllustrative sentence:", nlp_negation_example)
print("Removed stop words:", nlp_removed_tokens)
print("After that removal:", " ".join(nlp_filtered_tokens))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The supplied stop-word set is not learned from these tweets; the first comprehension keeps alphabetic tokens and lowercases them.
# The next two comprehensions partition those tokens by set membership, preserving order and repetitions; joining shows the retained wording.
# Membership does not measure meaning: removing `not` erases this sentence's denial without changing any baseline input.
#
# </details>

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
# ### ✅ Evidence, Answer & Limits
# The tokenizer comparison shows different hashtag and URL handling for the same tweet.
# The lemma tables show changed word forms; the stop-word example shows loss of an explicit denial.
# These outputs describe transformations, not a new classifier evaluation.
#
# ### 🧠 What We Learned
# Token definitions and grammatical roles change the representation, while removal can discard essential meaning.
# The baseline continues with its existing prepared strings; a lemmatization benefit for class-1 F1 remains unestablished.
# Testing that benefit would need a separately agreed training-only comparison with identical folds and fold-fitted features.
#
# > **⚠️ Watch Out — Fewer word forms do not establish better F1**
# >
# > The five-tweet example measures transformations; its vocabulary reduction does not establish improved predictions.
#
# **⚖️ Ethical Takeaway:** Keep original text available so preprocessing cannot silently turn a denial into an apparent emergency claim.

#
# ### 🤔 Question Yourself
# A preprocessing step reduces the vocabulary but removes `not` from a denial.
# What would you need before claiming it improves disaster classification?
#
# <details><summary>💡 Show answer</summary>
#
# Vocabulary reduction describes a transformation, not predictive performance.
# Compare task-aligned predictions with identical training folds and fold-fitted features, while checking whether important meaning survives.
# The language lab did not evaluate a lemmatized classifier.
#
# </details>

# <!--
# endregion
# region 3. Predictive Modeling, Comparison & Evaluation
# -->

# %% [markdown]
# ## 3. Predictive Modeling, Comparison & Evaluation
#
# ### Baseline model and evaluation
#
# ### What is the Question ❓
# How well does the TF-IDF + Logistic Regression baseline classify the held-out
# tweets, and where does it fail?
#
# **📋 Source & Context:** Use the unchanged prepared training and holdout
# objects above. The outcome is one measured baseline for the retained population,
# not evidence of reliability on every disaster report.
#
# The school `guided.ipynb` teaches training-only TF-IDF, then Logistic Regression.
# Our Pipeline connects that sequence; tweet preparation, defaults and measured scores are project-specific.
#
# <details><summary>📋 Teaching sources and adaptations</summary>
#
# `guided.ipynb` uses movie-review cleaning and the `sag` solver; neither is transferred here.
# `GMT20250226 Pipeline II.ipynb`, cells 20–25, also places vectorization inside model selection.
# Its stray `L` in cell 20 prevents copying it verbatim.
#
# DataCamp sources inspected for this adaptation:
# - [TF-IDF vectors for TED talks](https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/tf-idf-and-similarity-scores?ex=3): instructions and starter code for numerical text features.
# - [Your first TfIdf](https://campus.datacamp.com/courses/sentiment-analysis-in-python/more-on-numeric-vectors-transforming-tweets?ex=14): separate learning and transformation.
# - [Pipeline for song genre prediction I](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=6)
#   and [II](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=7): named steps, fitting, and prediction.
# - [Assessing a diabetes prediction classifier](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/fine-tuning-your-model-3?ex=3): classification report and confusion matrix.
#
# These inspected instructions and starter examples inform our adaptation; course solutions and scores are not reused.
# Installed scikit-learn 1.5.2 documentation supplies API details and defaults.
#
# </details>

# %% [markdown]
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

# %% [markdown]
#
# ### 6. A — How does text become numerical evidence?
#
# **🎯 Goal & Approach — WHY:** Turn prepared tweets into numerical features,
# learning the representation only from training text.
#
# **🔎 Guiding Questions:** What occupies a matrix row or column? Which information
# becomes a feature, and which information is lost?
#
# Each matrix row is a tweet; each column is a vocabulary term learned from training.
# Tokenization defines which text units can enter that vocabulary.
#
# Term frequency counts occurrences within a tweet. Inverse document frequency (IDF)
# gives more weight to terms appearing in fewer training tweets.
# Defaults multiply counts by smoothed IDF, then scale each nonzero row to unit length (L2 normalization).
# The resulting weights are not probabilities.
#
# Most entries are zero because a tweet contains few vocabulary terms.
# A sparse matrix saves space by storing nonzero entries and their positions.
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

# Define unfitted text and classifier steps; no vocabulary, IDF, or coefficients are learned yet.
tfidf_vectorizer = TfidfVectorizer()
logistic_classifier = LogisticRegression()

# Each tuple gives a step a name and its object. List order is the data-processing order.
# Creating the Pipeline connects the objects; it does not yet learn from any tweet.
baseline_steps = [("tfidf", tfidf_vectorizer), ("classifier", logistic_classifier)]
baseline_pipeline = Pipeline(baseline_steps)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The constructors define an unfitted vectorizer and classifier; each tuple names one step and supplies its object.
# `Pipeline(baseline_steps)` connects those steps, while the later `fit()` learns vocabulary, IDF, and classifier parameters.
# Consistent ordering still requires appropriate training inputs; creating a Pipeline does not prevent leakage by itself.
#
# </details>

# %% [markdown]
# ### 🧠 What We Learned
# A prepared string and a model feature are different objects. Tokenization and
# weighting determine which distinctions the classifier can use; this representation
# cannot retain every distinction visible to a human reader.
#
# **Which quantities must be learned, and how can prediction reuse them without learning from the holdout?**
#
# ### 7. B — What does the classifier learn, and how does it predict?
#
# **🎯 Goal & Approach — WHY:** Fit the single Pipeline on the 5,986 training
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
# Learn vocabulary, IDF and classifier parameters from the training partition only.
baseline_pipeline.fit(X_train, y_train)

# Inspect one training document using the SAME fitted vectorizer, without another fit.
# Its single sparse row uses the entire learned feature axis; .shape means (documents, terms).
training_example_text = X_train.head(1)
training_example_features = tfidf_vectorizer.transform(training_example_text)
print("Training example (source index retained):")
print(training_example_text.to_string())
print("One training document's feature-matrix shape:", training_example_features.shape)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `fit(X_train, y_train)` learns training vocabulary and IDF, builds a sparse tweet-by-term matrix, then fits classifier coefficients and an intercept.
# The vectorizer learns from text, while the classifier also uses the recorded labels; neither step sees holdout inputs during fitting.
# `head(1)` retains one training source index, and `transform()` reuses the fitted representation without learning new terms.
# This separation protects the evaluation workflow, but does not establish accurate annotations or independent events.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# The fitted representation has 18,494 term columns; the inspected training document
# occupies one row on that axis.
# The classifier's training matrix has 5,986 rows, one per training observation.
# Training converged within the default limit without warnings or parameter adjustments.
#
# TF-IDF learned vocabulary and IDF from text alone; Logistic Regression used those
# features and the recorded training labels to learn coefficients and an intercept.
# Prediction reuses all of these quantities without updating them from holdout text or labels.
#
#
# ### 🤔 Question Yourself
# A holdout tweet contains a word absent from the fitted training vocabulary.
# Does prediction add a feature column or learn a new IDF value for it?
#
# <details><summary>💡 Show answer</summary>
#
# No. Prediction reuses the training vocabulary and IDF values; the unknown term is ignored.
# If no known tokens remain, the feature row is zero and the classifier uses only its intercept.
# Refitting on holdout text would change the evaluation procedure.
#
# </details>

# %% [markdown]
# ### From tokens to visible TF-IDF features
#
# **What is the Question ❓** Which terms and weights does our existing fitted vectorizer produce for the inspected training tweet?
#
# **🎯 Goal & Approach — WHY:** Connect the language lab to the actual baseline representation, without fitting another vectorizer or changing predictions.
# MasterSchool's `GMT20250213-MovieLens 100K Dataset.ipynb`, cells 7–11, displays count and TF-IDF tables for small texts.
# DataCamp's [TF-IDF representation of product feedback](https://campus.datacamp.com/courses/natural-language-processing-nlp-in-python/feature-extraction-from-text?ex=7)
# supplies a similar table pattern; we inspect only one sparse row instead of making the entire training matrix dense.
#
# **💻 Code / Method — HOW:** The fitted vectorizer's analyzer supplies its actual tokens; counting them gives this document's term frequencies.
# Stored feature indices connect nonzero weights to vocabulary names and learned IDF values.
# The graph ranks TF-IDF weights, which are neither classifier coefficients nor evidence that a term caused a prediction.

# %%
from collections import Counter

# Reuse the fitted token rule; the Counter maps each term to its count in this one tweet.
nlp_baseline_analyzer = tfidf_vectorizer.build_analyzer()
nlp_baseline_tokens = nlp_baseline_analyzer(training_example_text.iloc[0])
nlp_term_counts = Counter(nlp_baseline_tokens)
print("Actual baseline tokens, source row", training_example_text.index[0], ":")
print(nlp_baseline_tokens)
print("Occurrences within this tweet:")
print(pd.Series(nlp_term_counts, name="count_in_tweet").to_string())

# %% [markdown]
# The analyzer returns the baseline's actual terms, including `fant4stic` and `fantasticfour`.
# Counts describe this tweet alone; TF-IDF also needs the vocabulary positions and IDF learned from all training text.

# %%
# Sparse column positions identify which learned vocabulary terms occur in this tweet.
nlp_sparse_row = training_example_features.getrow(0)
nlp_feature_indices = nlp_sparse_row.indices
nlp_all_feature_names = tfidf_vectorizer.get_feature_names_out()
nlp_feature_terms = nlp_all_feature_names[nlp_feature_indices]
print("Sparse row shape:", nlp_sparse_row.shape)
print("Stored nonzero features:", len(nlp_feature_indices))

# %% [markdown]
# One document occupies 18,494 possible term columns, but only 16 have nonzero weights.
# Use those stored column positions to align each term with its learned IDF and weight.

# %%
# Each row describes one present vocabulary term; all arrays use the same sparse column positions.
nlp_feature_table = pd.DataFrame({
    "term": nlp_feature_terms,
    "count_in_tweet": [nlp_term_counts[term] for term in nlp_feature_terms],
    "training_idf": tfidf_vectorizer.idf_[nlp_feature_indices],
    "tfidf_weight": nlp_sparse_row.data,
})
nlp_feature_table = nlp_feature_table.sort_values(
    ["tfidf_weight", "term"], ascending=[False, True]
)
print("All nonzero features for this training example:")
print(nlp_feature_table.round(4).to_string(index=False))

# %% [markdown]
# `fant4stic` and `fantasticfour` each occur once, with IDF about 9.0042 and normalized weight about 0.3378.
# For equal counts within this tweet, a higher IDF produces a higher TF-IDF weight.
# The next chart displays the ten largest weights from this same table.

# %%
# A small display table keeps the plot readable; other nonzero features remain in nlp_feature_table.
nlp_feature_plot = nlp_feature_table.head(10)
nlp_feature_plot = nlp_feature_plot.sort_values("tfidf_weight", kind="stable")
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
# <details><summary>💡 Explain this code</summary>
#
# `head(10)` selects the largest weights; the second sort arranges those bars without changing membership.
# Bar length shows normalized TF-IDF; hover adds the term count and training-derived IDF.
# The complete nonzero-feature table remains above. These weights are inputs, not classifier coefficients or causal explanations.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# The full 16-row feature table comes from the unchanged fitted baseline; the chart displays its ten largest weights.
# The other vocabulary terms have zero weight for this tweet.
#
# ### 🧠 What We Learned
# A large TF-IDF weight does not by itself imply support for class 1;
# the classifier's signed coefficient also matters.
# Fitting establishes a representation and model. Their agreement with recorded
# holdout labels, including the errors hidden by a single score, is evaluated next.
#
# > **⚠️ Watch Out — TF-IDF weight is not a class contribution**
# >
# > The chart displays input feature weights, not coefficients or their contributions to the decision.
# > A feature representation does not explain what caused a real disaster.

#
# ### 🤔 Question Yourself
# A term has the largest TF-IDF weight in the displayed tweet.
# Does that mean it provides the strongest support for the disaster class?
#
# <details><summary>💡 Show answer</summary>
#
# No. TF-IDF describes an input feature; its classifier coefficient determines the sign and size of its contribution.
# For this linear classifier, that contribution is the feature weight multiplied by its coefficient.
# The full decision combines all contributions with the intercept.
#
# </details>

# %%
# Reuse fitted steps; predictions follow X_holdout positions and have no pandas source index.
holdout_predictions = baseline_pipeline.predict(X_holdout)
print("Holdout predictions:", len(holdout_predictions))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `Pipeline.predict()` reuses the fitted vectorizer, then passes its features to the fitted classifier without refitting.
# Its 0/1 array follows holdout input order; `len()` counts predictions rather than correct answers.
# Preserve that positional pairing when comparing predictions with recorded labels.
#
# </details>

# %% [markdown]
# ### 8. C — What do the scores and mistakes tell us?
#
# **🎯 Goal & Approach — WHY:** Compare predictions with the 1,497 recorded
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

# Compare labels and predictions in matching positional order; digits rounds only the displayed report.
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
# <details><summary>💡 Explain this code</summary>
#
# `classification_report()` reports precision, recall, F1, and support; class 1 supplies the primary F1.
# `labels=[0, 1]` fixes the matrix as `[[TN, FP], [FN, TP]]`, with recorded-label rows and prediction columns.
# The DataFrame adds readable axes; all results measure annotation agreement rather than independently verified disaster events.
#
# </details>

# %% [markdown]
# ### Visual evidence — the same holdout disagreements
# MasterSchool's `GMT20250219 load_dataset('imdb').ipynb`, cells 22 and 26, displays confusion matrices as heatmaps.
# This Plotly adaptation visualizes our existing counts: rows are recorded labels and columns are predictions.
# It adds no evaluation observations or new model result; sentiment class names from the lesson do not apply here.

# %%
# Display existing confusion counts: rows are recorded labels, columns are predictions.
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
# <details><summary>💡 Explain this code</summary>
#
# `px.imshow()` displays the existing matrix; `text_auto=True` prints each count.
# Recorded-label rows and prediction columns place missed positives at lower left and false alarms at upper right.
# Color encodes counts, not rates; the two recorded classes have different denominators.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
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
# Copy holdout wording and labels in prediction order before selecting illustrative disagreements.
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
# <details><summary>💡 Explain this code</summary>
#
# Assign the prediction array by position before sorting, preserving its pairing with source-indexed holdout rows.
# The two masks select recorded 0/predicted 1 and recorded 1/predicted 0; `.loc`, `sort_index()`, and `head(3)` select reproducible examples.
# A default-index Series could misalign predictions, and three examples per error type cannot establish prevalence or causes.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits — selected disagreements
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
# ### 🧠 What We Learned
# Precision exceeds recall for class 1: almost one-third of recorded positives are missed.
# F1 summarizes agreement; the matrix and examples reveal the error trade-off.
# Excluding conflicting labels removes neither all ambiguity nor possible annotation errors.
# Event independence and emergency-response benefit remain unestablished.
# Broader claims need reviewed event links, annotation guidance and independent review,
# followed by evaluation on separately collected, representative examples.
#
# **Answer to the Project Question:** Text supports measured agreement with recorded
# labels in this retained holdout: class-1 F1 is 0.7308, with 208 missed recorded positives.
# Trust beyond this experiment also requires evidence about annotation quality,
# event separation, and the intended evaluation population; those conditions remain unestablished.
#
# The written reasoning is a learning resource. Data's independent Learning PASS
# remains **UNASSESSED**; successful execution does not demonstrate understanding.

#
# ### 🤔 Question Yourself
# A classifier has high overall accuracy. Can that alone tell you whether it finds most disaster-labelled tweets?
#
# <details><summary>💡 Show answer</summary>
#
# No. Accuracy includes both recorded classes, so correctly ignored negatives can conceal missed positives.
# Read class-1 recall and the confusion matrix to see how many recorded positives were found and missed.
#
# </details>

# %% [markdown]
# ### Post-baseline learning — Guided cross-validation and model comparison
#
# ### What is the Question ❓
# How do classifier choice and adjacent token pairs change training-fold F1, and how consistent are these comparisons?
#
# **📋 Source & Context:** This retrospective extension uses the existing 5,986 training rows after the original holdout was inspected.
# The original baseline and 1,497-row holdout remain separate.
#
# **🎯 Goal & Approach — WHY:** Compare three classifiers at two n-gram settings using the same ten stratified folds.
# Fit the complete text Pipeline inside each fold so validation text cannot influence vocabulary or IDF.
#
# **🔎 Guiding Questions:** Do bigrams improve all three classifiers, and does the highest mean establish a reliable winner?
#
# Each fold withholds 598 or 599 training observations and fits on the other nine parts.
# Every training observation is validated once per configuration; fitting subsets overlap, so these are not ten independent experiments.
#
# > **⚠️ Watch Out — Learn TF-IDF inside every CV fold**
# >
# > Guided cells 14–15 precompute TF-IDF before CV, allowing validation text to influence vocabulary and IDF.
# > Our CV receives raw prepared strings and the entire Pipeline, learning features separately inside each fold.
#
# <details><summary>📋 Teaching sources and adaptations</summary>
#
# Guided cell 14 summarizes ten F1 scores; cell 15 compares Logistic Regression and LinearSVC with one- to four-token sequences.
# Here we use only unigram/bigram choices and explicitly shuffle stratified folds with seed 42.
# Sprint 3 suggests Logistic Regression or Naive Bayes; their comparison here is additional learning, not a missing baseline requirement.
#
# Inspected DataCamp instructions and starter examples support:
# - [Cross-validation](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-55bf0d67-0cc2-4212-a6e5-bda0d07d3411?ex=9): split and score individual folds; we adapt regression to stratified F1.
# - [CV summaries](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/regression-55bf0d67-0cc2-4212-a6e5-bda0d07d3411?ex=10): summarize scores with a mean and spread.
# - [Model comparison](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=14): loop over alternatives, keeping learned transformations inside our Pipelines.
# - [Movie-review Naive Bayes](https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/n-gram-models?ex=8): fit MultinomialNB; our adaptation uses tweet TF-IDF and training-only F1.
#
# Course examples supply methods, not this project's scores; the source notebooks remain unchanged.
#
# </details>

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
#
# `(1, 1)` uses individual tokens; `(1, 2)` also includes adjacent pairs, such as `not safe`.
# Bigrams preserve limited local order. All other TF-IDF defaults stay fixed.
#
# | Classifier | What fitting learns | Main distinction |
# | --- | --- | --- |
# | Logistic Regression | Coefficients through a probability-based loss | Regularized linear classifier; `lbfgs`, `C=1`. |
# | LinearSVC | Coefficients through a margin-based loss | Margin-based decision scores, not native probabilities; `C=1`, seed 42. |
# | MultinomialNB | Class proportions and term weights within each class | Combines evidence using Bayes' rule; assumes features are conditionally independent given the class. |
#
# LinearSVC retains its default squared-hinge loss, L2 penalty and `dual="auto"` optimizer choice.
# Equal `C` does not make the two objectives equivalent.
#
# **Why Naive Bayes accepts these inputs:** TF-IDF supplies nonnegative feature weights.
# Scikit-learn accepts these as fractional feature mass, although the multinomial model describes counts.
# Related words and overlapping n-grams also violate its independence simplification.
#
# **How Naive Bayes calculates a prediction:** Sum each term's weights within each training class.
# Default `alpha=1.0` adds one before normalization, preventing zero term probabilities.
# Default `fit_prior=True` estimates class proportions from that fold's labels.
# Prediction combines the log prior with weighted log term probabilities, then chooses the higher-scoring class.
# Logs turn multiplication into addition; the fitted vectorizer ignores unknown tokens.
#
# > **⚠️ Watch Out — Compatible inputs do not validate Naive Bayes assumptions**
# >
# > Nonnegative TF-IDF permits the calculation; it does not prove independent language features, correct labels or trustworthy probabilities.
# > We compare predicted labels using F1.
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

# Define seeded training-only folds shared by candidates; each fold learns its own full text Pipeline.
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

# %% [markdown]
# **Step 1 — Build six named configurations.** Pair each classifier with each n-gram range before running cross-validation.

# %%
# Each dictionary entry holds one unfitted Pipeline; its name will identify the later fold scores.
cv_pipelines = {}
for classifier_name, classifier in cv_classifier_options:
    for ngram_range in ngram_options:
        cv_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        cv_steps = [
            ("tfidf", cv_vectorizer),
            ("classifier", classifier),
        ]
        cv_pipeline = Pipeline(cv_steps)
        configuration = f"{classifier_name} {ngram_range}"
        cv_pipelines[configuration] = cv_pipeline
print("Configurations to compare:")
print(list(cv_pipelines))

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The two loops combine three classifiers with two n-gram choices; the dictionary contains six named Pipelines.
# Creating those objects performs no fitting; the next cell evaluates each complete workflow on the same training folds.
#
# </details>

# %% [markdown]
# **Step 2 — Measure fold F1.** Reuse the named Pipelines; each configuration returns ten scores before any averaging.

# %%
# Reset measured scores when rerunning this cell; each array retains its ten validation F1 values.
fold_scores_by_configuration = {}
for configuration, cv_pipeline in cv_pipelines.items():
    fold_f1_scores = cross_val_score(
        cv_pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="f1",
        n_jobs=1,
        error_score="raise",
    )
    fold_scores_by_configuration[configuration] = fold_f1_scores
    print(f"Completed ten training-only folds: {configuration}")

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# `cross_val_score()` clones the complete Pipeline for each fold and returns ten validation F1 values.
# Each clone fits TF-IDF and its classifier only on that fold's fitting rows; the holdout is absent.
# Sequential execution and `error_score="raise"` expose fitting failures instead of silently replacing them with missing scores.
#
# </details>

# %% [markdown]
# **Step 3 — Inspect individual folds.** Turn the stored score arrays into a table before reducing them to summaries.

# %%
# Rows represent fold numbers 1..10, not source-row indices; columns identify configurations.
# Matching rows use matching validation observations because all candidates share the splitter.
cv_fold_scores = pd.DataFrame(fold_scores_by_configuration, index=range(1, 11))
cv_fold_scores.index.name = "fold"
print("\nClass-1 validation F1 by training fold:")
print(cv_fold_scores.round(4).to_string())

# %% [markdown]
# **Step 4 — Summarize each configuration.** Calculate the mean and spread from those same stored scores, without fitting again.

# %%
# Each record describes one configuration; ddof=0 preserves the existing descriptive standard-deviation convention.
cv_summary_records = []
for configuration, fold_f1_scores in fold_scores_by_configuration.items():
    mean_f1 = fold_f1_scores.mean()
    std_f1 = fold_f1_scores.std(ddof=0)
    min_f1 = fold_f1_scores.min()
    max_f1 = fold_f1_scores.max()
    cv_summary_records.append({
        "configuration": configuration,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "min_f1": min_f1,
        "max_f1": max_f1,
    })

# Each table row now summarizes ten scores; names become the index instead of ordinary numbered rows.
cv_summary = pd.DataFrame(cv_summary_records)
cv_summary = cv_summary.set_index("configuration")
print("\nRetrospective training-only CV summary (not final holdout performance):")
print(cv_summary.round(4).to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Each row summarizes the ten scores already inspected in the fold table; `round(4)` changes printed precision only.
# Fold spread is descriptive because the training subsets overlap; it is not an independent final test or confidence interval.
#
# </details>

# %% [markdown]
# ### 11. Read the fold results without overstating them
# ### ✅ Evidence, Answer & Limits
# The existing Python 3.11.11 / scikit-learn 1.5.2 environment produced these results with the fixed ten-fold splitter.
# Each score evaluates recorded class-1 labels in one withheld part of the training set.
# The preceding table prints all ten scores per configuration; matching fold numbers use matching validation rows.
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
# **Classifier comparison:** Unigram LinearSVC has the highest mean, about 0.0048 above unigram Logistic Regression,
# but leads it in only four of ten folds. Naive Bayes has lower means at both n-gram settings, though it occasionally leads within a fold.
#
# **Bigram comparison:** Adding pairs lowers all three means. Logistic Regression declines in all ten folds,
# Naive Bayes in eight, while LinearSVC changes in both directions. This comparison does not isolate the causes.
#
# **Spread:** Unigram Naive Bayes has the smallest standard deviation (0.0147), despite lower mean F1.
# Standard deviation uses `ddof=0` and describes these ten scores; it is not a confidence interval or significance test.
# The mean averages fold F1 values, which differs from F1 calculated on pooled predictions.
#
# > **⚠️ Watch Out — The best CV average is not fresh test performance**
# >
# > Selecting and reporting a configuration on these same folds can produce optimism.
# > The familiar holdout cannot become a new independent final test, and its F1 is not directly comparable with a CV average.
#
# Guided cell 16 refits its chosen model; this comparison stops before that step, preserving the original baseline and holdout predictions.
#
# ### 🧠 What We Learned
# Mean and spread answer different questions; neither alone establishes a reliable winner.
# Overlapping training subsets are not independent experiments, and CV cannot repair annotation quality or establish event independence.
# Reliability beyond these folds remains unestablished; it needs reviewed labels, event relationships and fresh representative evaluation kept separate from model choice.
#
# **⚖️ Ethical Takeaway:** Report the population and evaluation limits alongside scores; a favorable CV average does not verify disaster-detection reliability.

#
# ### 🤔 Question Yourself
# Suppose TF-IDF is fitted on all training texts before cross-validation, without using their labels.
# Can information still leak from a validation fold?
#
# <details><summary>💡 Show answer</summary>
#
# Yes. Validation texts would already influence the vocabulary and IDF, even without their labels.
# Fit the complete Pipeline separately on each fold's training subset, then transform its validation texts without refitting.
# This keeps validation text out of feature learning.
#
# </details>

# %% [markdown]
# ### 11.1 Beyond the baseline — a small regularization experiment
#
# **What is the Question ❓** Can changing one Logistic Regression setting reduce missed positives without concealing additional false alarms?
#
# **🎯 Goal & Approach — WHY:** Compare `C=0.1, 1, 10` using the existing training rows, unigram TF-IDF, and ten stratified folds.
# The same three settings were explored in a temporary training-only run; this section makes that comparison reproducible.
# It is an exploratory extension, not a new independent test or a replacement for the original baseline.
#
# **WHY C:** Regularization penalizes large learned coefficients, limiting how strongly individual features influence the model's decision.
# Smaller `C` means a stronger penalty; larger `C` permits more flexibility, which does not guarantee better predictions.
# We change only `C`, keeping the solver, preprocessing, class weights, and default prediction rule fixed.
#
# MasterSchool Sprint 3 suggests optional `C`/solver trials; see [Project Map, section 3.2](DA112_PROJECT_MAP.md#32-sprint-3--self-paced-project-our-principal-assignment).
# DataCamp's [Regularizing models with Twitter data](https://campus.datacamp.com/courses/sentiment-analysis-in-python/lets-predict-the-sentiment?ex=11)
# compares regularization on airline sentiment; this assistant-authored adaptation uses our disaster labels and leakage-safe folds.
# See the installed-version [LogisticRegression parameters](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html).
#
# **💻 Code / Method — HOW:** Out-of-fold predictions give each training row one prediction from a model fitted on the other nine parts.
# Each complete Pipeline learns its vocabulary, IDF, and classifier within that fold; the original holdout is excluded.
# Three settings require 30 fits, including a repeated `C=1` reference because the earlier comparison retained scores, not predictions.
#
# We pool these predictions to count errors and calculate recall, precision, and F1 across all training rows.
# Mean fold F1 remains separate: averaging ten F1 values is not the same calculation as F1 from pooled predictions.
# Neither summary establishes performance on new events; see [cross_val_predict](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.cross_val_predict.html).

# %%
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict

# Fix the C choices; prediction-table rows retain training source indices and columns identify settings.
c_values = [0.1, 1.0, 10.0]
c_oof_predictions = pd.DataFrame(index=y_train.index)

# %% [markdown]
# First collect out-of-fold predictions for each fixed C value; then evaluate the stored columns in a separate cell.

# %%
for c_value in c_values:
    # A fresh candidate leaves the already fitted baseline untouched; all other defaults stay unchanged.
    c_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(C=c_value)),
    ])

    # cross_val_predict clones the whole Pipeline inside each fold and returns labels in input order.
    # predict keeps Logistic Regression's default decision rule; no threshold is selected here.
    c_oof_predictions[c_value] = cross_val_predict(
        c_pipeline, X_train, y_train, cv=cv_splitter, method="predict", n_jobs=1,
    )
    print(f"Completed ten training-only folds for C={c_value:g}")

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Each prediction column retains the training source indices and contains one withheld-fold prediction per training row.
# A fresh Pipeline is fitted inside each fold, so the later metric calculation can reuse predictions without further fitting.
# The original fitted baseline and holdout remain separate from these C experiments.
#
# </details>

# %% [markdown]
# Evaluate one prediction column at a time, distinguishing pooled F1 from the mean of ten fold scores.

# %%
# First summarize all withheld predictions together: one record per C setting.
c_pooled_records = []
for c_value, c_predictions in c_oof_predictions.items():
    c_confusion_counts = confusion_matrix(y_train, c_predictions, labels=[0, 1])
    # Fixed matrix order is TN, FP, FN, TP; these counts use recorded dataset labels.
    c_tn, c_fp, c_fn, c_tp = c_confusion_counts.ravel()
    c_pooled_records.append({
        "C": c_value,
        "pooled_recall": recall_score(y_train, c_predictions, pos_label=1),
        "pooled_precision": precision_score(y_train, c_predictions, pos_label=1),
        "pooled_f1": f1_score(y_train, c_predictions, pos_label=1),
        "missed_positives": c_fn,
        "false_alarms": c_fp,
    })
c_pooled_summary = pd.DataFrame(c_pooled_records).set_index("C")
print("Pooled results from all withheld training predictions:")
print(c_pooled_summary.round(4).to_string())

# %% [markdown]
# Compared with `C=1`, `C=10` misses fewer recorded positives and produces more false alarms.
# These pooled scores combine all 5,986 predictions; next inspect F1 separately within each validation fold.

# %%
# Reuse stored predictions; positional selection pairs each fold's labels and predictions without fitting again.
c_fold_scores_by_value = {}
for c_value, c_predictions in c_oof_predictions.items():
    c_fold_scores = []
    for _, valid_positions in cv_splitter.split(X_train, y_train):
        c_fold_labels = y_train.iloc[valid_positions]
        c_fold_predictions = c_predictions.iloc[valid_positions]
        c_fold_f1 = f1_score(c_fold_labels, c_fold_predictions, pos_label=1)
        c_fold_scores.append(c_fold_f1)
    c_fold_scores_by_value[c_value] = c_fold_scores

# Rows identify the same validation folds; columns identify the three C settings.
c_fold_score_table = pd.DataFrame(c_fold_scores_by_value, index=range(1, 11))
c_fold_score_table.index.name = "fold"
print("F1 for each training validation fold:")
print(c_fold_score_table.round(4).to_string())

# %% [markdown]
# Each column contains ten F1 values, calculated on the same fold memberships across settings.
# Average each column separately, then place that mean beside its pooled metrics to keep the two calculations distinguishable.

# %%
# Preserve the original mean-of-folds calculation; Series indices align each mean with its C setting.
c_mean_fold_scores = {}
for c_value, c_fold_scores in c_fold_scores_by_value.items():
    c_mean_fold_scores[c_value] = pd.Series(c_fold_scores).mean()
c_mean_fold_f1 = pd.Series(c_mean_fold_scores)
c_summary = c_pooled_summary.copy()
c_summary.insert(3, "mean_fold_f1", c_mean_fold_f1)
print(f"\nTraining-only OOF comparison: {len(y_train)} rows; "
      f"{y_train.eq(1).sum()} recorded positives; {y_train.eq(0).sum()} recorded negatives.")
print(c_summary.round(4).to_string())

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The final table reuses pooled results and inserts each C setting's mean of ten fold F1 values.
# Index alignment matches the C keys, while `round(4)` changes only printed precision.
# Pooled F1 and mean fold F1 summarize different calculations; neither is a new holdout result.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
# The local Python 3.11.11 / scikit-learn 1.5.2 run completed all 30 fits without warnings.
# Each setting produced 5,986 withheld predictions: 2,549 recorded positives and 3,437 recorded negatives.
# The repeated `C=1` mean fold F1 reproduces the earlier 0.7506; no new holdout predictions were made.
#
# | C | Pooled recall | Pooled precision | Pooled F1 | Mean fold F1 | Missed positives | False alarms |
# | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
# | 0.1 | 0.5073 | 0.8507 | 0.6355 | 0.6350 | 1,256 | 227 |
# | 1.0 | 0.7018 | 0.8069 | 0.7507 | 0.7506 | 760 | 428 |
# | 10.0 | 0.7372 | 0.7704 | 0.7534 | 0.7534 | 670 | 560 |
#
# **Answer:** Compared with `C=1` on these same rows, `C=10` has 90 fewer missed positives but 132 additional false alarms.
# Pooled F1 rises only about 0.0027, while precision falls; `C=0.1` misses substantially more positives despite higher precision.
# This supports a small observed trade-off, not a reliably superior model or an explanation of why particular tweets changed predictions.
#
# > **⚠️ Watch Out — Comparing settings is not an independent final test**
# >
# > Selecting and reporting a preferred setting from these same results can produce optimism; no winner is refitted here.
# > The original 67.35% recall concerns different holdout rows and cannot serve as the starting point for this comparison.
#
# ### 🧠 What We Learned
# **Would the additional detections justify the additional false alarms?** That decision is not established by F1 alone.
# It requires an agreed purpose, acceptable review workload, and error costs; these educational labels do not establish operational benefits.
# Preserve the original baseline and treat `C=10` as a candidate for further investigation, with independent evidence needed for broader claims.

#
# ### 🤔 Question Yourself
# A setting has the highest F1 on the training folds. Does that justify selecting it using the holdout as a tie-breaker?
#
# <details><summary>💡 Show answer</summary>
#
# No. Using holdout scores to choose a setting turns that holdout into selection data.
# Keep selection within training validation; report exploratory differences with their error trade-offs and seek independent evidence for broader claims.
#
# </details>

# <!--
# endregion
# region 4. Hugging Face: Pretrained Transformer Mini-Lab
# -->

# %% [markdown]
# ## 4. Hugging Face: Pretrained Transformer Mini-Lab
#
# ### One pretrained Transformer inference
#
# ### What is the Question ❓
# What does one pretrained sentiment prediction demonstrate, and how does its
# pipeline differ from our fitted scikit-learn Pipeline?
#
# **🎯 Goal & Approach — WHY:** Learn to interpret a pretrained model's
# output within its actual task. Use one original English movie-review sentence,
# separate from our tweets, to observe local inference without training or model comparison.
#
# **🔎 Guiding Questions:** What was learned before this lab? What do the returned
# label and score mean, and can they answer our disaster-classification question?
#
# **📋 Source & Context:** The closing MasterSchool Transformers lesson
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
# This section uses the existing PyTorch, Transformers, and huggingface-hub versions recorded in requirements.txt.
# Restart the Interactive Window kernel and run cells in order using the existing Python 3.11.11 `.venv`.
#
# **WHY a local snapshot:** Explicit cache lookup keeps the model revision and file location inspectable without contacting the Hub.
# The required snapshot must already exist under `.cache/huggingface/hub/`; missing files stop execution instead of triggering a download.
# Prediction runs locally on CPU; the demonstration sentence is not sent to a hosted inference service.
# See README for setup and [Hugging Face cache layout](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

# %%
import torch
import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("\nHugging Face mini-lab (sentiment demonstration only):")
print("PyTorch:", torch.__version__, "Transformers:", transformers.__version__)

# Pin matching sentiment model/tokenizer files; loading the pipeline performs no training or prediction.
hf_model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
hf_model_revision = "714eb0fa89d2f80546fda750413ed43d93601a13"

# Despite its name, snapshot_download with local_files_only only locates cached files.
# An explicit cache_dir prevents silently reusing the user-wide cache.
hf_cache_dir = project_root / ".cache" / "huggingface" / "hub"
hf_model_files = ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"]
try:
    hf_model_path = Path(snapshot_download(
        repo_id=hf_model_id, revision=hf_model_revision, cache_dir=hf_cache_dir,
        allow_patterns=hf_model_files, local_files_only=True,
    ))
except LocalEntryNotFoundError as error:
    raise FileNotFoundError(
        f"Missing {hf_model_id} at {hf_model_revision} in {hf_cache_dir}. "
        "See README for separate resource setup; nothing was downloaded."
    ) from error

# A snapshot directory can exist after an incomplete copy; check every required file before loading.
hf_missing_files = [name for name in hf_model_files if not (hf_model_path / name).is_file()]
if hf_missing_files:
    raise FileNotFoundError(f"Incomplete snapshot {hf_model_path}: {hf_missing_files}. See README; no download was attempted.")

# Load matching local tokenizer and safetensors weights; no tweet data or labels enter this operation.
hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path, local_files_only=True, trust_remote_code=False)
hf_model = AutoModelForSequenceClassification.from_pretrained(
    hf_model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
)
sentiment_pipeline = transformers.pipeline(
    task="sentiment-analysis", model=hf_model, tokenizer=hf_tokenizer, device="cpu",
)
print("Model:", hf_model_id)
print("Revision:", hf_model_revision)
print("Model snapshot:", hf_model_path)
# The loaded model's configuration maps numerical output positions to sentiment names.
print("Sentiment labels:", sentiment_pipeline.model.config.id2label)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# The model ID, pinned revision, and project cache identify the local snapshot; missing or incomplete files stop loading without downloads.
# AutoClasses load its tokenizer and safetensors weights with remote code disabled; the CPU pipeline combines them and `id2label` exposes sentiment names.
# Loading external weights performs no tweet training, and those label numbers do not describe disaster classes.
#
# </details>

# %%
# Use one original sentence outside the dataset; the sentiment model supplies its own tokenization.
demo_text = "The film was delightful and beautifully acted."

# Calling the pipeline tokenizes the sentence, runs the fixed model, and processes its scores.
# The result is a list with one dictionary for this input: its top label and softmax score.
# Softmax turns two raw class scores into positive values summing to one; weights are not updated.
sentiment_result = sentiment_pipeline(demo_text)
print("Demonstration text:", demo_text)
print("Observed sentiment result:", sentiment_result)

# %% [markdown]
# <details><summary>💡 Explain this code</summary>
#
# Calling the pipeline tokenizes the illustrative sentence, runs fixed weights, and applies softmax to its two raw class scores.
# The one-item result contains the highest-scoring sentiment label and its score.
# One prediction tests the inference path; that score is neither population accuracy nor disaster F1.
#
# </details>

# %% [markdown]
# ### ✅ Evidence, Answer & Limits
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
# ### 🧠 What We Learned
# The sentence and output are **Data**; the checkpoint, task and label definitions
# provide **Information**. **Knowledge:** successful inference answers only that model's task.
#
# **Wisdom:** check label meaning before interpreting a large score as project success.
# Suitability for disaster classification is not established here; it needs task-aligned
# labels and independent, representative evaluation with reviewed annotations and event relationships.
#
# **⚖️ Ethical Takeaway:** Do not present sentiment confidence as evidence that a disaster occurred.

#
# ### 🤔 Question Yourself
# A sentiment model returns `POSITIVE` with a score near one. What does that tell you about whether a disaster occurred?
#
# <details><summary>💡 Show answer</summary>
#
# It reports the model's sentiment prediction for that input, not a disaster label or verified event.
# The score does not establish accuracy on disaster tweets; that would require task-aligned labels and representative evaluation.
#
# </details>

# <!--
# endregion
# region 5. Key Insights, Limitations & Recommendations
# -->

# %% [markdown]
# ## 5. Key Insights, Limitations & Recommendations
#
# **What is the Question ❓** What can we responsibly conclude and recommend from this project?
#
# **🎯 Goal & Approach — WHY:** Separate measured classification results from claims that require new evidence.
#
# **✅ Evidence, Answer & Limits:** The baseline reaches class-1 F1 0.7308 on the retained holdout,
# while missing 208 of 637 recorded positives. Training-only CV compares six configurations;
# it does not establish reliability on new events. The Transformer lab demonstrates sentiment inference only.
#
# **🧠 What We Learned:** Data quality and evaluation boundaries determine what a successful run can establish.
# Labels remain annotations, excluded conflicts change the population, and different wording can describe the same event.
# The training-text lab also shows that reducing word forms and preserving meaning are different goals; no lemmatized classifier was evaluated.
# Performance on independently reviewed tweets from separate events remains unestablished;
# the present experiment cannot supply that new evaluation evidence.
#
# **Recommendation:** Preserve this measured baseline and report its limitations alongside its scores.
# Before broader use, review annotations and event relationships, then evaluate on fresh representative examples.
# No operational benefit or emergency-response readiness has been demonstrated.
#
# **Ethical takeaway:** A model prediction should not be presented as verification that a disaster occurred.
#
# ### 🤔 Question Yourself
# What evidence would move this project from agreement with familiar labels towards a claim about new events?
#
# <details><summary>💡 Show answer</summary>
#
# Review annotation quality and event relationships, then evaluate a fixed workflow on fresh, representative examples from separate events.
# Define acceptable misses and false alarms for the intended use.
# The current results do not establish those conditions or operational benefit.
#
# </details>

# <!--
# endregion
# -->

# %% [markdown]
# <!-- project-wrap-up -->
# ## Wrap-up
#
# *Data → Information → Knowledge → Wisdom: from recorded tweets to a responsible next investigation.*
#
# ### Data — What did we use?
# We began with the supplied MasterSchool tweets and their recorded disaster labels,
# asking whether the text could help distinguish disaster-labelled examples from the rest.
# Repeated texts and conflicting labels showed why preparation had to come before modelling.
# The labels describe how the tweets were annotated; they do not verify that an event occurred.
#
# ### Information — What did we find?
# Once the data was prepared, TF-IDF and Logistic Regression gave us a baseline.
# Its misses and false alarms became the starting point for a separate Beyond the Baseline study.
# Frozen MiniLM representations, pretrained on external text, helped Logistic Regression
# find more disaster-labelled tweets, while also raising more false alarms on the same,
# already inspected holdout.
#
# ### Knowledge — What is the insight?
# That shifted our attention from the score to the meaning of the mistakes.
# Finding more labelled examples did not remove the trade-off; it changed the balance
# between missed examples and false alarms. Choosing between the approaches therefore
# needs a purpose and an understanding of what each error would cost.
#
# > **Insight**
# >
# > Fewer misses alone do not establish a better choice; the value depends on the cost of each error.
#
# ### Wisdom — What is the next question?
# Would that balance still hold on new events, and which errors would be acceptable for the intended use?
#
# That is where this educational, retrospective comparison leaves the story.
# The next investigation would define those limits, then evaluate the fixed approaches
# on independent, representative tweets from new events with reviewed labels.
# Whether the same pattern would hold beyond our examples remains an open question.
