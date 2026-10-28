# DA112_PROJECT_MAP.md

> **Project:** MasterSchool DA112 — NLP Classification (Disaster Tweets)
>
> **Version:** 1.3 — 2026-10-28
>
> **Status:** Source-grounded project plan. **This file does not authorize implementation.** Data approves each bounded work block and any required Git transport.
>
> **Progress:** Slice 0, Slice 1, Slice 2A, and Work block 1 are published. Preparation was implemented in `34c61b6` and educationally refactored in `f8e166c`. The approved duplicate policy and split remain fixed. Next: targeted teaching alignment before separately authorized modeling.
>
> **Repository:** https://github.com/zvuk1T/NLP_Classification_Project_Masterschool
>
> **Purpose:** A persistent source of truth for Data, Troi, and Spock: requirements, decisions, verified progress, and a coherent route to a portfolio-ready project.

---

## 1. Mission and non-negotiable direction

**Primary task:** Build an understandable, independently developed **binary classifier** that predicts whether a tweet describes a **real disaster** (`target = 1`) or **not a real disaster** (`target = 0`).

- **Current phase — Data's decision, 2026-10-27:** Data reports having already worked through this project at school and broadly understanding its methods. This reconstruction now emphasizes effective Data–Troi–Spock collaboration and a clear portfolio case study, followed by homepage integration. It is not a first-time replay of the entire course or evidence of a new school submission.
- **Understanding remains a quality requirement.** Briefly recap familiar material; explain unfamiliar concepts, consequential choices, and actual errors in more depth. Faster pacing does not mean skipping evidence or correctness checks.
- **WHY before HOW.** Keep the question, method, evidence, and limitations understandable without explaining every familiar operation again.
- **Minimum sound approach:** tweet-aware text preprocessing → TF-IDF → Logistic Regression → honest evaluation.
- **Data Literacy / DIKW:** Data → Information → Knowledge → Wisdom. Separate observed evidence from interpretation, and interpretation from recommendations.
- **Keep it school-level:** follow the guided Sprint 2 methodology, adapting what is genuinely needed for tweets. Avoid unnecessary production engineering, deep learning, elaborate architecture, and unrequested extensions.
- **No invented results or business impacts.** Never reuse example scores from lessons, previous portfolio claims, or another dataset as results of this project.
- **Homepage integration is a planned later objective**, including the appropriate project listing and featured/favorites placement. The actual site, affected files, and publication must be separately inspected and authorized.

**Approved implementation format:** As established by `AGENTS.md` and `HOW-WE-WORK-AND-LEARN.md`, the sole authoritative analysis is repository-root `disaster_tweets_classification.py`. Develop it in VS Code Python Interactive Window with `# %% [markdown]` explanations and `# %%` code cells. Preserve `Notebooks/NLP_Classification_Project_Masterschool.ipynb` unchanged as historical learner work. A derived Colab `.ipynb` export is a later, separately authorized deliverable, never a second independent implementation.

**Pacing:** One agreed work block → one complete implementation within scope → one joint review. A block may contain several analytical units and many cells; running cells in sequence does not require separate approval for each cell. Keep the established Main Question → Learning → Better Questions method, but do not turn routine checks into standalone tasks or reteach familiar material. Larger blocks do not excuse missing explanations inside code; apply the learning guide's inline walkthrough, ASCII-map, and optional ethical-takeaway standards.

## 2. Source authority, scope, and conflict resolution

Sources have different purposes. Do not flatten them into a single set of compulsory steps.

| Role | Source | What it governs |
|---|---|---|
| **A — School requirements** | User-provided MasterSchool **DA112 Sprint 3 — Self-paced** “Introduction & Key Takeaways” screenshots (2026-10-26) | The original assignment, required deliverables, and what was optional |
| **B — Guided methodology** | User-provided MasterSchool **DA112 Sprint 2 — Guided** lessons and `learning_materials/mentor_live_sessions/guided.ipynb` | Actual NLP teaching code and methodology; reference only, not reusable tweet-model results |
| **C — Mentor learning material** | `learning_materials/mentor_live_sessions/untitled41.py` | Mainly time-series teaching; use only relevant verified concepts, not as an NLP implementation |
| **D — Dataset and competition facts** | Kaggle official Disaster Tweets competition | Dataset semantics, competition data schema/rules, and F1 evaluation |
| **E — Actual project state** | Live inspection of this GitHub repository plus a separate local Git preflight | What files/code really exist; never infer local status from GitHub |
| **F — Optional post-core learning** | MasterSchool Hugging Face and GPT/Llama closing lessons; Hugging Face official documentation | Small practical learning experiment **after** the baseline, not a Sprint 3 requirement |
| **G — Teaching references** | Relevant official DataCamp lessons and available code, selected through `HOW-WE-WORK-AND-LEARN.md` | Accessible implementation patterns, analytical questions, Data Literacy, storytelling, and ethics; not additional compulsory courses |

**Conflict procedure:**
1. Platform/tool safety rules and Data's latest explicit authorization constrain all actions.
2. For “what must be submitted,” prefer verified **Sprint 3 instructions** over guided examples, README claims, or mentor notes.
3. For “how to implement safely,” use technically correct, leakage-resistant practice. Explain any necessary correction to a lesson example; never silently copy a problematic pattern.
4. For “what exists,” verify GitHub and the local workspace rather than trusting an old summary or this snapshot.
5. If evidence is missing, contradictory, license-sensitive, or involves a materially different methodological choice: **state the exact issue, present the smallest justified options, and STOP at that decision gate**. Do not invent a source or quietly choose an advanced workaround.

**Evidence limit:** The screenshots were supplied in the ChatGPT conversation on 2026-10-26, not checked into GitHub. The essential requirements are transcribed/summarized here so future sessions do not depend on access to that conversation. Do not claim the original images are locally present unless independently verified.

## 3. Verified curriculum map

### 3.1 Sprint 2 — GUIDED PROJECT (methodology reference)

**Dataset:** Sentence Polarity movie-review sentences — **10,662** sentences, evenly balanced:
- `rt-polarity.pos`: 5,331 positive sentences.
- `rt-polarity.neg`: 5,331 negative sentences.

Guided steps visible in the supplied lessons:

1. Introduce the binary sentiment-classification problem and data.
2. Read the two source files; construct text and class labels.
3. Preprocess text (lowercasing, handling punctuation/special characters, tokenization, stop words, optional lemmatization).
4. Shuffle and split labeled examples into train/test subsets (lesson illustrates 80/20).
5. Learn TF-IDF features on training text, train Logistic Regression, transform held-out text, and evaluate predictions.
6. Interpret precision, recall, F1, accuracy, and class-level results; confusion-matrix reasoning is part of the assignment guidance.
7. Learn K-fold cross-validation and compare fold-level F1 scores.
8. Learn hyperparameter search (the guided lesson explores Logistic Regression settings such as `C` and TF-IDF feature ranges).

**Scope rule:** Sprint 2 explains the method. Its labels, preprocessing choices, hyperparameters, sample output (including example F1 values), and model results **must not be transferred as established findings** to Disaster Tweets.

**Leakage correction:** A guided example cross-validates precomputed TF-IDF vectors. For our project, when CV is used, cross-validate a `Pipeline(TfidfVectorizer, LogisticRegression)` on raw training text so vocabulary/IDF are learned separately within each training fold.

### 3.2 Sprint 3 — SELF-PACED PROJECT (our principal assignment)

**Goal:** Predict whether each tweet is about a real disaster (`1`) or not (`0`), using labeled Disaster Tweets.

The supplied assignment's high-level steps are:

1. **Data loading:** Load `train.csv` with pandas and use the `text` and `target` columns.
2. **Text preprocessing:** Remove genuinely unnecessary characters; normalize as appropriate (lowercasing, stop-word handling, optional stemming or lemmatization). Do **not** assume movie-review cleaning suits tweets unchanged.
3. **Vectorization:** Bag of Words or TF-IDF. **Our approved design direction is TF-IDF.**
4. **Model selection/training:** Logistic Regression or Naive Bayes are suggested simple models. **Our approved design direction is Logistic Regression.** Random Forest/Neural Networks are only mentioned as possibilities, not mandates.
5. **Model evaluation:** Accuracy, precision, recall, F1, and a confusion matrix; evaluate predictions on **unseen labeled holdout data**.
6. **Hyperparameter tuning: OPTIONAL**, including trials with solver/`C` and cross-validation in the school guidance. Do not make this a prerequisite for the first valid model.

**Original school deliverables, explicitly shown:**
- A **Google Colab notebook** containing all code, clear comments/explanations, and the steps followed; the notebook link must be accessible to reviewers.
- A **recording of the live project presentation**, originally presented during the sprint's final session and submitted alongside the notebook. Presentation topics: problem/data, model architecture and training, challenges and decisions, and key learnings.

**Historical/reconstruction caveat:** These were the school's original submission requirements. This reconstruction must not claim the live session or recording exists. If a retrospective presentation is desired, Data must approve its format; label it honestly as retrospective.

**Not established as required by the supplied Sprint 3 screenshot:** a Kaggle leaderboard entry, a Kaggle competition submission file, a Transformer model, Hugging Face usage, a deployed app, or homepage integration.

### 3.3 Closing NLP/LLM lessons — separate learning context

The supplied closing lessons include:
- **Practical Session on Using Hugging Face Transformers for Multiple NLP Tasks:** Hugging Face `pipeline` demonstrations for sentiment analysis, summarization, question answering, and translation.
- **Overview of Popular LLMs (GPT, Llama) and Their Applications:** conceptual overview and examples, not a requirement to fine-tune or deploy an LLM.

**Decision:** A small, explainable Hugging Face inference exercise remains a desired optional follow-up after the baseline. It requires separate approval and is not a prerequisite for the portfolio case study or homepage integration.

## 4. Repository baseline — verified 2026-10-26, before this map was added

- Repository: `zvuk1T/NLP_Classification_Project_Masterschool`, public; default branch `main`.
- Inspected remote commit: `32fe0df4f28d8fe6e475bb0914bc699c57d489ad` (“Keep mentor lesson in Python format only”).
- Relevant tracked paths at that snapshot:
  - `README.md` — starter assignment description; loosely says “10,000 labeled tweets” (not the exact count of current labeled rows).
  - `DATA/train.csv` — **7,613** rows; only `text` and `target` columns.
  - `Notebooks/NLP_Classification_Project_Masterschool.ipynb` — **three code cells**: load CSV, `head()`, count `target` values. No fitted or evaluated NLP model. Its absolute `/Users/zeal.v/Desktop/...` data path is not portable.
  - `requirements.txt` — includes many currently unnecessary packages (e.g., TensorFlow, XGBoost, statsmodels); do not install them all just because they are listed.
  - `.gitignore` — existing narrow rules.
  - `learning_materials/mentor_live_sessions/untitled41.py` — mentor-related Python export; **already tracked on this public GitHub repository**.

**Historical state distinction:** At the 2026-10-26 snapshot above, only remote `main` had been inspected; local state and environment had not yet been verified. Preserve that snapshot as history, not a perpetual expected HEAD or a description of current local state.

**Local verification history, 2026-10-27:** The verified clone was `/Users/zeal.v/Desktop/vs_code/masterschool/NLP_Classification_Project_Masterschool`, on `main`, with origin `https://github.com/zvuk1T/NLP_Classification_Project_Masterschool.git`. An authorized fast-forward from `9d7bf5ed22005ad590cb852d987da5c2234350f5` reached `9fac37a305ff17e9e4049c6e29452ea55900d5b2`, changing only the two governance documents. Slice 2A then introduced local work, subsequently committed and published in `3d71400181eb86e865293104e4daa3a91889ace9`. At that inspection, the 12 school screenshots and two `.DS_Store` files were untracked; `guided.ipynb` was ignored and `untitled41.py` tracked. These are historical observations, not a current clean-working-tree claim.

**Later checkpoint, 2026-10-27:** The question-driven learning update was published in `247251e25a37b7acec0fb7fb3f69879d2d08afea`, changing only `AGENTS.md` and `HOW-WE-WORK-AND-LEARN.md`. Spock's subsequent read-only report placed local `main` at `3d71400…`, one commit behind that remote checkpoint. Recheck live local and remote state before the next authorized synchronization; this roadmap edit does not synchronize the Mac.

**Published implementation checkpoint:** Work block 1 was published in `34c61b6c51f946941b4d226df3cd06a27bc4a4e3`. Its educational refactor and the explicitly authorized school-reference notebook publication followed in `f8e166ca1e285773dd381f3853aef9cc094e99b3`. The reference notebook is now tracked, not merely local/ignored. Current local status still requires separate inspection; remote publication does not establish it.

### 4.1 Dataset audit facts (exact-text analysis of the tracked CSV)

Reproduced by the new Python analysis during Slice 2A on 2026-10-27, without altering the original table or CSV.

| Measurement | Verified value |
|---|---:|
| Total labeled rows | 7,613 |
| `target = 0` (non-disaster) | 4,342 |
| `target = 1` (disaster) | 3,271 |
| Missing `text` or `target` | 0 |
| Unique exact text strings | 7,503 |
| Additional rows from exact repeated texts | 110 |
| Distinct text strings appearing more than once | 69 |
| Original rows belonging to repeated-text groups | 179 |
| Distinct text strings carrying **both labels** | 18 |
| Original rows belonging to conflicting-label groups | 55 |

**Approved preparation policy:** Compare text after lowercasing, collapsing consecutive whitespace to one space, and trimming its ends. Preserve original text and source-row indices. Exclude every row in a normalized-text group containing both labels; keep the first source occurrence of each remaining consistent group. Apply the same prepared string as the later model input. Do not infer correct labels, overwrite the CSV, or change this policy without Data's decision.

**Reported execution and published analysis:** Normalization merges two pairs of distinct exact strings; 18 conflicting normalized groups contain 56 rows. Removing those rows and 74 redundant consistent occurrences retains 7,483 rows. Original exact-text audit counts above remain valid for their different comparison rule.

| Population | Rows | Label 0 | Label 1 |
|---|---:|---:|---:|
| Original | 7,613 | 4,342 | 3,271 |
| Prepared | 7,483 | 4,297 | 3,186 |
| Training | 5,986 | 3,437 | 2,549 |
| Holdout | 1,497 | 860 | 637 |

The approved split is stratified 80/20 with `random_state=42`. Spock reported zero original/prepared text overlap and exact preservation of the eight output objects through the refactor, including values, types, indices, membership, and order. Troi reviewed source and teaching, not a fresh runtime reproduction. Future authorized refactors must preserve that behavior and verify it proportionately.

**Interpretation limit:** Excluded groups have conflicting recorded labels. This does not establish that every excluded text is linguistically ambiguous, that all ambiguous texts were removed, or that retained labels are true. Exclusion changes the evaluated population; related events and near-duplicates may remain across partitions.

**Dataset provenance:** The public Kaggle competition provides more fields (`id`, `keyword`, `location`, `text`, and `target` for training); this repository contains only `text` and `target`. The exact transformation history is not documented. Use the supplied text-only CSV for the school-level baseline unless Data authorizes restoring other columns. Kaggle `test.csv` is unlabeled; never invent test labels or report a labeled holdout F1 for that file.

### 4.2 Mentor-code audit and publication boundary

`untitled41.py` was exported from Colab and mostly demonstrates **time-series forecasting** on `airline-passengers.csv` with XGBoost, `TimeSeriesSplit`, `GridSearchCV`, scikit-learn `Pipeline`/`ColumnTransformer`, and rolling features. It has incomplete illustrative fragments and undefined names; it is **not a runnable end-to-end NLP solution**.

Use selectively for understanding `Pipeline`, cross-validation, and tuning concepts. Do **not** reuse `XGBRegressor`, `TimeSeriesSplit`, time lags, R², or time-series preprocessing for this binary tweet-classification problem.

`learning_materials/mentor_live_sessions/guided.ipynb` was added in `f8e166c` under Data's explicit publication authorization, recorded in the supplied review handoff. It is an NLP school reference for sequential preparation, paired text/labels, splitting, and baseline methods, not a second Disaster Tweets implementation. Read relevant actual cells critically; do not execute, rewrite, or reuse saved scores as new project evidence.

**Publication boundary:** Both reference files are tracked in this public repository. Data's publication authorization is not independent proof of redistribution rights. Do not expand, republish, remove, or rewrite protected teaching material without the required scope and rights review. Local ignore rules do not hide public Git history; this documentation update changes neither reference file nor publication visibility.

## 5. Methodology contract for the first reliable baseline

These are **project design constraints**. Preparation exists; no Disaster Tweets model has been fitted at the published checkpoint.

### 5.1 Data and leakage

- Inputs are the approved prepared text; the original `text` and `target` remain intact. Preserve the tracked CSV.
- Keep the audit and approved policy in §4.1. Do not retune retention to maximize evaluation scores.
- Preserve the approved stratified **80/20**, `random_state=42` split, its source indices, membership, and order. Verify no text overlap under the agreed comparisons.
- Keep the holdout isolated from vocabulary learning, feature selection, hyperparameter decisions, and model fitting.
- Do not use the school's example `predict(X_train)` pattern to report generalization performance.

### 5.2 Text handling

- The approved minimal preparation is lowercasing and whitespace normalization, described in §4.1; do not silently add further cleaning.
- Tweets include hashtags, mentions, URLs, punctuation, informal spelling, and meaningful negation. **Do not blindly remove all hashtags, stop words, “not”, or disaster-specific tokens.**
- Stemming, lemmatization, complex regex pipelines, external linguistic datasets, and special tokenizers are **not baseline requirements**.
- Show a few representative before/after examples and any information that preprocessing may discard.
- Maintain exactly the same approved, deterministic text transformations for training and inference.

### 5.3 Features, model, and evaluation

- Use `TfidfVectorizer` + `LogisticRegression` as the first **explainable baseline**; favor scikit-learn `Pipeline` so fitting and prediction use consistent transformations.
- Start with simple defaults; increase `max_iter` only as needed to achieve convergence. Explain parameters actually changed.
- Fit TF-IDF and Logistic Regression on training examples **only**.
- Evaluate on untouched, labeled holdout examples. Primary metric: **binary F1 for `target=1`** (real disaster), matching Kaggle's positive-class framing.
- Report at least precision, recall, F1, accuracy, class support, and confusion matrix, with labels `[0, 1]` in an unambiguous order.
- Explain false positives vs. false negatives in **tweet-classification** terms. Modest class imbalance (**4,342 vs. 3,271** in the raw data) makes accuracy alone insufficient; use actual prepared/partition counts when interpreting evaluation.
- No made-up score thresholds, claims of real emergency-response impact, or assertions that an educational classifier is deployable.

### 5.4 Optional robustness improvements — only after a valid baseline

- If Data approves, use K-fold cross-validation **only on the training portion**; keep the holdout separate.
- For CV, pass the full `Pipeline(TfidfVectorizer, LogisticRegression)` so TF-IDF is fitted **inside each fold**. Do not CV previously fitted full-training TF-IDF features.
- If Data approves tuning, use a **small, explained** search (e.g., a few values of Logistic Regression `C` or TF-IDF `ngram_range`). Never run a broad grid merely because it appears in mentor material.
- Select/tune using training-fold results; use the holdout for a final unbiased assessment, not repeated model selection.
- Alternative models (Naive Bayes, Random Forest, neural networks) are **not required** and need separate approval.

## 6. Delivery plan — coherent work blocks, one at a time

**Rule:** Data approves a complete, bounded block, not every routine operation inside it. Spock implements and verifies that block, then Data and Troi review its decisions, evidence, and explanations. Stop inside a block only for a material unresolved decision, scope change, missing essential evidence, or correctness/safety problem; continue unaffected approved work where practical. Completing a block does not authorize the next or Git transport.

**Continuity:** The completed slices below remain historical evidence, not a checklist to restart. The four work blocks group the school's analytical stages without changing the safeguards in §5 or the learning method. Block 1 is published; the current teaching alignment is a bounded follow-up, not a new analytical phase.

### Completed foundation — Slice 0: Local and remote repository preflight

1. Locate the correct local clone, without assuming an old Mac path.
2. Verify local repo identity, `origin`, current branch, `git status`, local HEAD and remote `origin/main`; check ahead/behind/divergence.
3. Compare local/remote files relevant to this map; identify any untracked, private, or mentor files.
4. Confirm Python/Jupyter environment only by inspection; **do not install** dependencies.
5. Report exact local path, branch, SHA/status, blockers, and a safe synchronization recommendation.

**Completed:** Read-only preflight distinguished remote and local facts. Any subsequent synchronization was separately authorized; a fresh chat does not require repeating the whole investigation.

### Completed foundation — Slice 1: School requirements and dataset quality

1. Read this map, the official Sprint 3 requirement summary, the notebook, CSV, and mentor reference.
2. Inventory the historical notebook's starter cells and nonportable paths as source evidence, not an implementation target.
3. Recheck data shape, labels, blanks, exact duplicates, label conflicts, and text examples.
4. Explain how tweet text differs from movie-review sentences.
5. Propose **one** minimal duplicate/holdout policy, with alternatives only if material.
6. Summarize exactly which files would change in the next slice.

**Historical outcome:** Source and dataset review completed; at that checkpoint the duplicate/holdout policy was proposed but not yet approved. The later approval and implementation are recorded in §4.1 and Block 1; do not reopen them from this history.

### Completed foundation — Slice 2A: First working Python analysis (2026-10-27)

**Question:** What do the dataset structure, target distribution, and duplicate tweets tell us before modeling?

1. Prepare an ignored project-local `.venv` with a suitable installed Python, `pandas==2.2.3`, `ipykernel`, and their dependencies. Preserve dependency declarations.
2. Create root `disaster_tweets_classification.py` with question-led explanations and executable percent cells.
3. Load the supplied CSV portably; inspect columns, shape, missing/blank values, label validity, class counts, exact-text duplicates, and conflicting labels.
4. Distinguish observed counts from interpretations, uncertainty, and the next proposed decision.
5. Execute code, compare with Slice 1 findings, and verify cells in a fresh Interactive Window kernel where available. Report any unverified GUI behavior separately.

**Completed and published:** Commit `3d71400181eb86e865293104e4daa3a91889ace9` contains only the new `.py`, this map, and the `.venv/` addition to `.gitignore`. Original data and historical materials were unchanged. At this checkpoint no deduplication, preprocessing, split, vectorization, or model was implemented.

**Historical verified outcome:** Six code cells and eight Markdown cells were present. Python **3.11.11**, pandas **2.2.3**, NumPy **1.26.4**, and ipykernel **7.4.0** ran in the ignored project `.venv`. Only pandas and ipykernel were requested directly; existing `requirements.txt` pins constrained dependencies without installing the full file. Dependency consistency passed. This is not a fresh environment inspection.

- **Technical PASS reported for Slice 2A:** Script execution from two working directories and all six cells in a fresh VS Code project kernel succeeded. The DataFrame equaled a fresh CSV read. Counts matched Slice 1: 7,613 rows, classes 4,342/3,271, 7,503 exact texts, 69 repeated groups/179 rows/110 additional occurrences, 92 additional identical `(text, target)` rows, and 18 conflicting groups/55 rows. Missing, blank, and invalid-label counts were zero.
- **Teaching history:** The initial review accepted the question-led structure, but Data later identified insufficient code-cell explanation and excessive verification in the extended analysis. That acceptance did not settle the educational quality; the later refactor addressed those gaps.
- **Data's Learning PASS: UNASSESSED.** Colab export and execution have not been attempted.

### Work block 1 — Prepare the data for modeling (**PUBLISHED; targeted teaching alignment next**)

**Main Question:** How can we prepare these tweets for a fair, reproducible model evaluation?

The approved preparation and split were published in `34c61b6c51f946941b4d226df3cd06a27bc4a4e3`; §4.1 records their policy and results. No vectorizer or model was fitted.

**Educational refactor:** `f8e166ca1e285773dd381f3853aef9cc094e99b3` improved the entire assistant-authored analysis: meaningful question-led units, code-cell explanations, visible normalization steps, readable stage counts, and evidence-linked learning. Assertions, hashing, repeated splitting, and obsolete PASS messages were removed from the permanent teaching flow; essential input safeguards and analytical checks remain. The historical notebook and raw CSV were not rewritten.

**Evidence status:** Spock reported seven executable cells running in a fresh VS Code project kernel, script execution from two directories, and exact equality of the eight output objects against the preceding implementation. Those are reported runtime results, not independently rerun by Troi. Troi's source/teaching review was **PASS WITH MINOR ADJUSTMENTS**; Data's independent Learning PASS remains **UNASSESSED**.

**Next bounded follow-up:** Apply the updated guide to the existing `.py`, preserving the successful refactor rather than starting over. Convert relevant mental models to ASCII, add one contextual ethical takeaway where warranted, retain first-use inline explanations, and correct the claim that all ambiguous examples are absent. The supported wording is: “Examples with conflicting recorded labels are excluded; other ambiguous or incorrectly labeled texts may remain.” Preserve the approved preparation, all eight output interfaces, and exact split membership/order.

**Acceptance:** A readable, source-faithful analysis with these targeted teaching adjustments and proportionate verification; no new method, model, policy, or test framework. The map records the next task but does not authorize its execution or transport.

### Work block 2 — Baseline model and evaluation (**not started; requires separate approval**)

**Main Question:** How well does the simple text classifier perform on held-out tweets, and where does it fail?

- Implement TF-IDF + Logistic Regression together, fit on training data only, predict on the holdout, and report F1, precision, recall, accuracy, class support, and a confusion matrix.
- Inspect a few false positives and false negatives, explain limitations, and connect the verified results to the question. Do not use this holdout error review to tune the model against the same test set.
- Verify reproducibility with the actual environment and inputs. Keep familiar mechanics concise; explain new concepts and material decisions where used.

**Acceptance:** One executed, explainable baseline with genuine evaluation results and bounded interpretation. No automatic tuning, alternative models, or next-block implementation.

### Work block 3 — Portfolio case study (**requires approval**)

**Main Question:** How can a reader understand the problem, decisions, evidence, and limitations without reading our conversations?

- Refine the analytical narrative, README, and a small set of useful visuals in the approved file scope. Present the work honestly as a MasterSchool educational reconstruction.
- Explain what was classified, why the approach was chosen, what the model achieved on the defined holdout, where it failed, and what follows from the evidence.
- Preserve the authoritative `.py` and historical notebook. A derived Colab export or retrospective recording remains separately authorized; neither is an automatic prerequisite for this portfolio adaptation.

**Acceptance:** A clear, evidence-backed case study and reproducibility instructions are ready for website presentation. No invented business impact, historical submission, recording, or learning-assessment claims.

### Work block 4 — Homepage integration (**planned; requires specific authorization**)

**Main Question:** How should this project appear coherently within the existing portfolio website?

- Inspect the actual website repository and existing layout before choosing files or implementation details; do not assume the website lives in this NLP repository.
- Within approved scope, add the case study to the appropriate project section and the featured/favorites area, matching the existing design and navigation.
- Verify content, links, relevant builds, and page behavior. Publishing or deployment requires explicit authorization; a local preview is not a published page.

**Acceptance:** The agreed project placements and links are verified. Report separately whether the website changes are prepared, committed, or actually published.

### Optional follow-ups — CV, tuning, and Hugging Face

These are **not prerequisites for blocks 3 or 4**. After the baseline, Data decides whether their learning or analytical value justifies the extra work.

- **CV/tuning:** If approved, use training-only full-pipeline cross-validation and a small justified parameter search, preserving §5.4. Never optimize against the final holdout.
- **Hugging Face:** If approved, inspect a real model card and run one small inference example in separately approved learning cells. Explain task alignment, limitations, and why sentiment is not disaster detection. No fine-tuning, deployment, chatbot, RAG, GPU provisioning, or cloud expansion.
- Keep optional evidence separate from the baseline. Do not turn an exploratory demonstration into a claimed comparable benchmark or a mandatory new project.

## 7. Data Literacy, DIKW, and storytelling contract

All narrative claims must follow this reasoning, without compulsory headings for every operation:

1. **Data:** What was observed? (rows, class balance, example text, duplicate counts, raw predictions)
2. **Information:** What did our transformations/metrics quantify? (F1, precision/recall, confusion matrix, error counts)
3. **Knowledge:** What interpretation is supported, under which assumptions and limits? Do not treat a possible cause of label disagreement as established.
4. **Wisdom:** What justified next step follows, given evidence and consequences? It may be further investigation or a narrower claim, not an invented operational impact.

**Narrative question:** “Can a simple, interpretable text model distinguish disaster-related tweets from non-disaster usage — and where does it fail?”

Do not convert correlation, a held-out metric, or a small notebook demonstration into operational, safety, or business-impact claims. Follow the learning guide for ASCII mental maps and optional contextual ethical takeaways; those aids do not introduce new analytical work.

## 8. Spock operating protocol and authorization

**Plan ownership — Data's explicit reassignment, 2026-10-27:** Data and Troi maintain this canonical project roadmap. `AGENTS.md` now states ownership of all three canonical documents explicitly. Spock supplies implementation evidence and proposed corrections; authoring changes requires Data's specific reassignment.

Before an approved work block:

1. Read the current contract and learning guide under `AGENTS.md`'s reading requirements, plus the relevant current map. Identify the approved work; do not restart completed reviews solely because a new chat begins.
2. Check current repository/files before describing implementation state; do not treat historical checkpoints as live. Synchronize only when explicitly authorized.
3. State **WHY**, then the bounded **HOW**, expected files, checks, and a clear end-of-block review point.
4. If required source material or a material choice is missing, name the gap rather than fabricate a lesson, score, file, or requirement.

During an approved block:

- Work only in the correct existing repository. **Never create a substitute repo** or repurpose another MasterSchool project.
- Respect all applicable `AGENTS.md` and higher-priority instructions; this map does not replace them.
- Preserve user's existing work. No force push, `reset --hard`, destructive clean-up, automatic stashing, or silent conflict resolution.
- Do not edit outside the approved file/path scope. Avoid installing or importing unnecessary advanced libraries.
- Never silently rewrite the mentor material or copy its code into the authoritative Python analysis.
- Complete routine implementation and verification within scope without repeated permission requests. Escalate genuine decision gates, not every cell or familiar operation.
- Return one concise block report: changed files, actual checks/results, key decisions, limitations, and the next proposed question. Distinguish Technical PASS, Teaching PASS, and Learning PASS; leave learning **UNASSESSED** without independent evidence, rather than imposing an exam after every block.
- **Commit/push only if the current task explicitly authorizes that transport.** Do not assume approval to edit means approval to publish.
- **STOP at the agreed block boundary**; do not finish the remaining project in one run or automatically begin the next block.

**Approval overrides:** Data may explicitly change priorities or choose a different sound approach. Record the change and its reason in the work report or a separately approved document revision. Do not quietly reinterpret a new instruction as permission to modify the whole project.

## 9. Decision register (do not silently resolve)

| Decision | Current state | Required handling |
|---|---|---|
| Current focus and pacing | **Agreed:** familiar school reconstruction, sound understanding, collaboration, then portfolio; four coherent work blocks | Larger blocks do not remove explanations of new objects or transformations |
| Canonical maintenance | **Data + Troi:** contract, guide, and project map | Spock reports evidence and proposes corrections; no silent canonical edits |
| Publication and synchronization | Preparation published in `34c61b6`; refactor and school reference in `f8e166c` | Historical remote checkpoints, not a current local-state claim; inspect before authorized synchronization |
| Primary implementation format | **Decided:** root `disaster_tweets_classification.py`, VS Code percent cells | Preserve the historical notebook; derived Colab export needs separate authorization |
| Repeated/conflicting labels | **Approved and implemented:** lowercase/whitespace comparison, exclude conflicting groups, keep first consistent occurrence | Preserve §4.1; do not reopen policy during educational edits |
| Split and seed | **Approved and implemented:** stratified 80/20, `random_state=42`; 5,986 training / 1,497 holdout | Preserve exact membership, order, labels, and source indices |
| Educational follow-up | **Next:** targeted alignment of existing `.py` with the updated guide | ASCII maps, relevant ethical takeaway, precise conflict wording, preserved inline reasoning; no model work |
| Colab workflow | **Deferred; unresolved** | Separately authorize export, execution, and sharing |
| CV/tuning | **Optional, not approved** | Separate post-baseline decision; not required before portfolio preparation |
| Teaching-material publication | `guided.ipynb` publication explicitly authorized; independent rights verification not established here | Both reference files remain unchanged; no broader reuse or publication permission |
| Retrospective presentation/recording | **Not created** | Do not infer a recording from prior coursework |
| Hugging Face mini-lab | **Desired optional follow-up, not implemented** | Separate post-baseline approval |
| Website/homepage | **Planned:** project section plus featured/favorites placement | Inspect the actual site and approve changes and publication in block 4 |

## 10. Project completion criteria

### Core analytical completion

The preparation checkmarks record the published implementation and Spock's reported runtime verification, not a fresh execution by Troi.

- [x] Repo identity and historical local path verified; recheck live Git state before work.
- [x] Source data audited; duplicate/label conflicts handled under the approved §4.1 policy.
- [x] Reproducible train/holdout separation reported and documented.
- [x] Minimal tweet-aware preprocessing implemented with original text preserved.
- [ ] Targeted educational follow-up completed and reviewed under the current guide.
- [ ] TF-IDF and Logistic Regression fitted without holdout or CV-fold leakage.
- [ ] Real, reproducible holdout F1/precision/recall/accuracy/confusion matrix.
- [ ] Model errors, limitations, and DIKW-driven conclusions documented after evaluation.

### Portfolio communication and integration (separate from model correctness)

- [ ] Full authoritative `.py` analysis readable end to end, with reproducible execution instructions.
- [ ] Case study, README, and selected visuals communicate the question, decisions, evidence, errors, and limits.
- [ ] Approved website project and featured/favorites placements verified; publication status reported truthfully.
- [x] Original school deliverable expectations identified without claiming unperformed historical submission.
- [ ] Any separately approved Colab export or presentation/recording truthfully labeled and verified; neither is implied by case-study completion.
- [ ] Optional extensions marked optional, with separate evidence and approval; they do not block a sound baseline.

**Not required for core analytical completion:** Kaggle leaderboard rank, deep learning, Transformer fine-tuning, model deployment, or guaranteed real-world impact. Homepage integration is a separate planned portfolio block, not evidence of model quality.

## 11. Source register

**Primary supplied evidence (screen captures shared 2026-10-26; not currently tracked in the repo):**
- Sprint 3 “Introduction & Key Takeaways” and deliverables: `SCR-20261026-snhr.png` (also a second capture `SCR-20261026-skzi.png`).
- Sprint 2 overview and guided lessons: `SCR-20261026-slux.png`, `SCR-20261026-slyo.png`, `SCR-20261026-smbw.png`, `SCR-20261026-smgd.png`, `SCR-20261026-smix.png`, `SCR-20261026-smpm.png`, `SCR-20261026-sndx.png`.
- Closing Hugging Face and GPT/Llama lessons: `SCR-20261026-sobc.png`, `SCR-20261026-snrc.png`.

**Published school reference:** `learning_materials/mentor_live_sessions/guided.ipynb` at `f8e166ca1e285773dd381f3853aef9cc094e99b3`. Data's publication approval and Spock's execution/equivalence report were supplied in the independent-review handoff; source review does not independently reproduce that runtime evidence.

**Online references:**
- MasterSchool Sprint 3 course context (access may require login):
  https://app.masterschool.com/campus/unit/c7f2a58e-2596-42df-b3af-0c21fddb9e3a.899442d4-6ee7-486a-b844-d6edd15e8cf8/sprint/c7f2a58e-2596-42df-b3af-0c21fddb9e3a.899442d4-6ee7-486a-b844-d6edd15e8cf8.3
- Kaggle Disaster Tweets (data, evaluation, competition rules):
  https://www.kaggle.com/competitions/nlp-getting-started
- Kaggle dataset schema and license notice:
  https://www.kaggle.com/competitions/nlp-getting-started/data
- scikit-learn — common pitfalls / data leakage:
  https://scikit-learn.org/stable/common_pitfalls.html
- Hugging Face Transformers pipelines:
  https://huggingface.co/docs/transformers/main_classes/pipelines
- Verified project files:
  https://github.com/zvuk1T/NLP_Classification_Project_Masterschool
- Relevant official DataCamp teaching and conceptual references:
  See `HOW-WE-WORK-AND-LEARN.md`, sections 4 and 6.

---

## 12. Current execution instruction

**Work block 1 and its educational refactor are published. The next proposed task is targeted teaching alignment of the existing Python analysis, not new data preparation or Block 2.**

Continue from `f8e166ca1e285773dd381f3853aef9cc094e99b3` and any later verified, authorized work; preserve its improvements. Apply the current guide's code-cell explanation, ASCII-map, optional ethical-takeaway, and precise-claim standards within an explicit task. The eight stable interfaces are `tweets`, `model_tweets`, `train_rows`, `holdout_rows`, `X_train`, `y_train`, `X_holdout`, and `y_holdout`; preserve their values, labels, indices, membership, and order.

This canonical-document update does not itself edit or execute Python, synchronize the Mac, authorize model training, or publish implementation changes. The earlier model-building prompt was superseded by the educational work; Block 2 needs a new explicit task after this review. Local synchronization and later Git transport require their own stated scope.

**Spock: read the current canonical documents, complete only the explicitly approved bounded task, report evidence and limits, and STOP.**
