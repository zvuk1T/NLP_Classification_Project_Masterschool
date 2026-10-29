# DA112_PROJECT_MAP.md

> **Project:** MasterSchool DA112 — NLP Classification (Disaster Tweets)
>
> **Version:** 1.9 — 2026-10-29
>
> **Status:** Source-grounded project plan. **This file does not authorize implementation.** Data approves each bounded work block and any required Git transport.
>
> **Progress:** The core MasterSchool Sprint 3 analysis and **TF-IDF + Logistic Regression baseline** are complete (original holdout F1 **0.7308**, 208/637 recorded positives missed); the baseline-focused README was published in `7a4007b`. Guided classical CV (`53d473e`) and Naive Bayes (`ece01b8`) are later educational additions. **Checkpoint `8524cb7` adds training-only EDA/NLTK/Plotly teaching, pretrained DistilBERT sentiment inference, and a separate frozen-MiniLM feature experiment.** These files report local execution; this documentation update does not independently rerun them. **Primary story: Sprint 3 baseline; Beyond the baseline: optional learning.** The entire Homepage implementation belongs to the separate Homepage project; LSTM is excluded.

> **Repository:** https://github.com/zvuk1T/NLP_Classification_Project_Masterschool
>
> **Purpose:** A persistent source of truth for Data, Troi, and Spock: requirements, decisions, verified progress, and a coherent route to a portfolio-ready project.

---

## 1. Mission and non-negotiable direction

**Primary task:** Build an understandable, independently developed **binary classifier** that predicts whether a tweet describes a **real disaster** (`target = 1`) or **not a real disaster** (`target = 0`).

- **Current phase — 2026-10-29:** The original school Sprint 3 and six training-only classical CV configurations are documented. The main Python file also contains EDA/lemmatization learning and a reported executed pretrained DistilBERT sentiment example; the second Python file reports frozen-MiniLM embeddings, new classifier CV and retrospective holdout scores. These extensions are **Beyond the baseline**, not required school steps or a new unbiased test. A documentation review does not verify their clean-environment reproduction, actual graphical rendering or a new school submission.
- **Understanding remains a quality requirement.** Briefly recap familiar material; explain unfamiliar concepts, consequential choices, and actual errors in more depth. Faster pacing does not mean skipping evidence or correctness checks.
- **WHY before HOW.** Keep the question, method, evidence, and limitations understandable without explaining every familiar operation again.
- **Minimum sound approach:** tweet-aware text preprocessing → TF-IDF → Logistic Regression → honest evaluation.
- **Data Literacy / DIKW:** Data → Information → Knowledge → Wisdom. Separate observed evidence from interpretation, and interpretation from recommendations.
- **Keep it school-level:** follow the guided Sprint 2 methodology, adapting what is genuinely needed for tweets. Avoid unnecessary production engineering, deep learning, elaborate architecture, and unrequested extensions.
- **No invented results or business impacts.** Never reuse example scores from lessons, previous portfolio claims, or another dataset as results of this project.
- **Cross-project boundary:** `zvuk1T/zarko-vukovic-homepage` exclusively owns its public project catalog, Featured placement, Overview/page architecture, visual export/integration, routing, builds, Git transport and deployment under its own canon. DA112 provides evidence only; it does not execute website tasks.

**Python roles:** `disaster_tweets_classification.py` is authoritative for the original school Sprint 3, TF-IDF baseline and fixed holdout. `disaster_tweets_huggingface.py` is a separate **frozen-MiniLM study**, not another implementation of that baseline. Both use VS Code percent cells. Preserve the historic learner notebook unchanged; separately approved Colab export cannot become a parallel implementation.

**Pacing:** One agreed work block → one complete implementation within scope → one joint review. A block may contain several analytical units and many cells; running cells in sequence does not require separate approval for each cell. Keep **Main Question → evidence → explicit answer → Learning → Better Questions** at meaningful-unit level. Preserve first-use explanations, **Mermaid mental maps in separate Markdown cells**, consequential **⚠️ Watch Out** warnings, and evidence-grounded **⚖️ Ethical Takeaways** as defined by the current learning guide.

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
8. Compare `LogisticRegression` and `LinearSVC` across TF-IDF `ngram_range=(1, n)` for `n=1..4`, using 10-fold CV in the actual `guided.ipynb`. This guided loop does **not** vary Logistic Regression `C`; `C`/solver trials are separate optional Sprint 3 ideas.

**Scope rule:** Sprint 2 explains the method. Its labels, preprocessing choices, hyperparameters, sample output (including example F1 values), and model results **must not be transferred as established findings** to Disaster Tweets.

**Leakage correction:** A guided example cross-validates precomputed TF-IDF vectors. For our project, when CV is used, cross-validate a `Pipeline(TfidfVectorizer, LogisticRegression)` on raw training text so vocabulary/IDF are learned separately within each training fold.

### 3.2 Sprint 3 — SELF-PACED PROJECT (our principal assignment)

**Goal:** Predict whether each tweet is about a real disaster (`1`) or not (`0`), using labeled Disaster Tweets.

The supplied assignment's high-level steps are:

1. **Data loading:** Load `train.csv` with pandas and use the `text` and `target` columns.
2. **Text preprocessing:** Remove genuinely unnecessary characters; normalize as appropriate (lowercasing, stop-word handling, optional stemming or lemmatization). Do **not** assume movie-review cleaning suits tweets unchanged.
3. **Vectorization:** Bag of Words or TF-IDF. **Our approved design direction is TF-IDF.**
4. **Model selection/training:** Logistic Regression or Naive Bayes are suggested simple models. **Our published baseline is Logistic Regression.** Naive Bayes was added only as a post-baseline learning comparison in `ece01b8`. Random Forest/Neural Networks are possibilities, not required methods; LSTM is expressly outside this reconstruction's scope.
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

**Current evidence — 2026-10-29:** The primary Python file reports one locally executed **DistilBERT SST-2 sentiment** inference (`POSITIVE`, softmax about **0.999887**) on an original movie-review sentence; this is not disaster classification. The separate MiniLM Python file reports locally executed **frozen 384-feature embeddings** plus Logistic Regression/LinearSVC CV and a retrospective holdout study. Neither is a Sprint 3 school requirement, LSTM training or encoder fine-tuning. This map's revision did not reproduce those runs.

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

**Published implementation checkpoints:** Work block 1 was published in `34c61b6c51f946941b4d226df3cd06a27bc4a4e3`; its educational refactor and authorized school-reference publication followed in `f8e166ca1e285773dd381f3853aef9cc094e99b3`. The canonical contract was streamlined in `5c859f13e07208ceee50d79cb6023b2280f3ed73`. Work block 2 — the explained TF-IDF + Logistic Regression baseline and holdout evaluation — was published in `b106c9833d76942b4fc02ee0109c35c0d4af2edb`. The concise-explanation and standalone-ASCII-map rules were clarified in `6f427574e4cc25f19278320c57d215cb354d0a48`. Work block 3 — the portfolio-facing README and final case-study presentation — was published in `7a4007b4137e4a845769bd24e96e00be89a26157`. Current local status still requires separate inspection; remote publication does not establish it.

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

These are **project design constraints** that produced the published baseline. Preparation and the first Disaster Tweets model now exist; preserve these constraints when interpreting the result or considering optional post-baseline work.

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

### 5.4 Post-baseline learning comparisons — not mandatory Sprint 3 requirements

- **Completed in `53d473e`:** ten-fold stratified cross-validation on the original 5,986 training rows, with full TF-IDF + classifier Pipelines independently fitted within each fold. Logistic Regression and LinearSVC were compared at `ngram_range=(1, 1)` and `(1, 2)`. The published baseline and original holdout metrics were preserved.
- **Completed in `ece01b8`:** `MultinomialNB` added at the same two n-gram settings and the same ten `StratifiedKFold` partitions. The code now compares six TF-IDF + classifier Pipelines (60 fold fits). Spock reported successful execution without warnings; Troi reviewed the source and checked the recorded Naive Bayes fold summaries. The original baseline and holdout stay unchanged.
- Each Pipeline refits TF-IDF inside **each training fold**, not before CV. In the published results, `MultinomialNB` has mean class-1 F1 **0.7219** for `(1, 1)` (std **0.0147**) and **0.7113** for `(1, 2)` (std **0.0156**), below the two existing classifiers' corresponding means. Nonnegative TF-IDF weights are usable but are not literal token counts; neither small spread nor class scores validate independence assumptions or generalization.
- The holdout has already been inspected. Neither ranking candidates nor adding Naive Bayes turns that holdout into a fresh independent final test. No new deployment or real-world reliability claim follows.
- **LSTM deliberately excluded:** its sequence/embedding/deep-learning workflow would broaden this school-level reconstruction. No neural-network model training is planned in DA112. Other model families require a new Data decision.

## 6. Delivery plan — coherent work blocks, one at a time

**Rule:** Data approves a complete, bounded block, not every routine operation inside it. Spock implements and verifies that block, then Data and Troi review its decisions, evidence, and explanations. Stop inside a block only for a material unresolved decision, scope change, missing essential evidence, or correctness/safety problem; continue unaffected approved work where practical. Completing a block does not authorize the next or Git transport.

**Continuity:** Completed NLP work blocks 1–3 remain historical evidence, not tasks to restart. **The former NLP Work block 4 (Homepage integration) was superseded by Data's decision to do all website work exclusively in the separate Homepage project.** DA112 supplies source-backed facts and provenance, not a website implementation phase.

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

### Work block 1 — Prepare the data for modeling (**COMPLETE AND PUBLISHED**)

**Main Question:** How can we prepare these tweets for a fair, reproducible model evaluation?

The approved preparation and split were published in `34c61b6c51f946941b4d226df3cd06a27bc4a4e3`; §4.1 records their policy and results. The educational refactor in `f8e166ca1e285773dd381f3853aef9cc094e99b3` established the question-led structure, first-use code explanations, visible transformations, readable checks, and evidence-linked learning.

The current `.py` also reflects the later teaching standard: standalone ASCII mental maps, a contextual ethical takeaway, and the precise claim that conflicting recorded labels are excluded while other ambiguous or incorrectly labeled texts may remain. The approved eight preparation interfaces and exact split remain the basis for the published baseline.

**Acceptance status:** Complete. Technical evidence for preparation and preservation was reported during implementation; Troi reviewed the source/teaching design. Data's independent Learning PASS remains **UNASSESSED**.

### Work block 2 — Baseline model and evaluation (**COMPLETE AND PUBLISHED**)

**Main Question:** How well does the TF-IDF + Logistic Regression baseline classify the held-out tweets, and where does it fail?

Published in `b106c9833d76942b4fc02ee0109c35c0d4af2edb`. The single scikit-learn Pipeline learns TF-IDF and Logistic Regression from the 5,986 training observations only, then predicts the unchanged 1,497-row holdout.

**Published holdout evidence:**
- Positive-class F1: **0.7308**
- Precision, class 1: **0.7989**
- Recall, class 1: **0.6735**
- Accuracy: **0.7889**
- Confusion matrix: **752 TN, 108 FP, 208 FN, 429 TP**
- The model misses **208 of 637 recorded positives (32.65%)**.

The analysis inspects a small reproducible set of false positives and false negatives without using them to tune the baseline. Results describe the retained fixed holdout; they do not establish annotation truth, event independence, deployment readiness, or real-world emergency benefit.

**Acceptance status:** Complete. The original baseline and bounded error analysis were published, followed by the training-only CV comparison in `53d473e` and the Naive Bayes extension in `ece01b8`. All six candidates remain retrospective educational comparisons, not replacements for the fixed holdout baseline.

### Work block 3 — Portfolio case study (**COMPLETE AND PUBLISHED**)

**Main Question:** How can a reader understand the problem, decisions, evidence, and limitations without reading our conversations?

Published in `7a4007b4137e4a845769bd24e96e00be89a26157`. The README now presents the project as an explainable MasterSchool educational case study: problem, preparation decisions, baseline mechanism, holdout evidence, error patterns, limitations, reproducibility, and source provenance.

The authoritative `.py` remains the executable analysis and learning resource; the historical learner notebook remains separate. No new school submission, presentation recording, independent Learning PASS, business impact, deployment readiness, or emergency-response benefit is claimed.

**Acceptance status:** The README published in `7a4007b` correctly presents the **original** baseline but predates classical CV and both Transformer learning labs. It is not currently a complete overview of both Python files. Any NLP README revision requires separate authorization; the Homepage project owns webpage design, graphics and integration.

### Homepage handoff — owned solely by the Homepage project (former NLP Work block 4 superseded)

**Decision — 2026-10-29:** The entire webpage implementation, including cards, Featured and collection choices, links, project page/Overview, Plotly/Mermaid visuals, accessibility, build and deployment, belongs **only to `zvuk1T/zarko-vukovic-homepage`** and its canonical approvals. No such actions are authorized in DA112.

**Evidence the NLP repository makes available:** Original MasterSchool Sprint 3 TF-IDF + Logistic Regression evaluated **1,497** held-out records: class-1 F1 **0.7308**, **208 of 637** recorded positives missed. Training-only classical CV is separate. The pinned, frozen MiniLM experiment reports mean CV F1 **0.773690** (Logistic Regression) and **0.768419** (LinearSVC), plus **retrospective** known-holdout F1 **0.7650 / 0.7718**. Its matched Logistic Regression holdout shows 43 fewer false negatives and 17 more false positives. These are recorded experiment findings, not a new independent final test, verified causal benefit or reliable emergency response.

**Approved public-story hierarchy:** **Sprint 3 → TF-IDF baseline → observed errors and limits** comes first; classical-model CV, pretrained sentiment and frozen-MiniLM work are labeled **Beyond the baseline**. The NLP source does not determine the Homepage's visual design or link destination. Original code and author-created charts may inform site assets, but existing Plotly/Mermaid cells are not by themselves verified website deliverables. Never copy private records, mentor-owned teaching files, or unapproved course screenshots as website assets.

### Beyond the baseline — completed educational extensions

**Classical CV:** The guided question-answer, manual Pipeline and Logistic Regression/LinearSVC CV learning was published in [`53d473e`](https://github.com/zvuk1T/NLP_Classification_Project_Masterschool/commit/53d473e22ff48ad5f32e43da514e24cd5015217e); [`ece01b8`](https://github.com/zvuk1T/NLP_Classification_Project_Masterschool/commit/ece01b8cb4184b6b5ed6bd0d7a75136dbd71a22a) added `MultinomialNB`, producing six configurations on the same ten training-only folds. The original holdout baseline was not replaced.

**Expanded school teaching and sentiment:** In [`8524cb7`](https://github.com/zvuk1T/NLP_Classification_Project_Masterschool/commit/8524cb7cd28816940be19465780a576f09862616), the main `.py` added training-only text-length EDA, tokenization/lemmatization examples, Plotly views, Mermaid maps and a reported executed DistilBERT sentiment example. Sampled linguistic transformations are **not new baseline inputs**. Its `POSITIVE` softmax score is not evidence of disaster detection.

**Frozen MiniLM representation:** The same commit added `disaster_tweets_huggingface.py`. It reports a pinned pretrained encoder producing **384-number vectors**, two fixed classifiers evaluated with ten training folds, and retrospective evaluation on an already examined 1,497-row holdout. Logistic Regression and LinearSVC mean CV F1 were reported as **0.773690** and **0.768419**; retrospective holdout F1 **0.7650** and **0.7718**. The encoder is frozen, not fine-tuned; external pretraining is extra information, and no independent event-level evaluation is supplied. These are source-file runtime reports, not fresh execution by Troi.

**LSTM:** Deliberately excluded. Data's Learning PASS remains **UNASSESSED**. None of these extensions is a missing required Sprint 3 step or authorization for additional models or website work.

## 7. Data Literacy, DIKW, and storytelling contract

Our narrative uses DIKW to develop evidence into understanding and justified judgment, without compulsory headings for each operation:

1. **Data — What was recorded?** Tweet text, recorded labels, predictions, source indices, and provenance.
2. **Information — Who, what, where, when, and under which definitions?** Organize counts and metrics within the known population and denominator. Do not infer absent event times or locations from this text-only dataset.
3. **Knowledge — How do patterns relate, and why does it matter?** Explain supported relationships and mechanisms, including assumptions and limits. Accuracy mixes classes and can hide missed positives; this does not reveal why an individual tweet was misclassified.
4. **Wisdom — What judgment is justified?** Use knowledge, limitations, and consequences to choose a better evidence-answerable question, justified decision or action, narrower claim, or deliberate pause.

**Insight connects Knowledge to Wisdom; it is not a fifth DIKW level.** Better Questions is our inquiry method, not an official DataCamp definition of Wisdom. A better question should follow evidence that changed our understanding, not merely announce the next Python operation; it never authorizes a new block by itself.

The published analysis already follows this cycle in its learning reflections. These canonical clarifications do not change the baseline, its fixed holdout, or Data's **UNASSESSED** independent Learning PASS.

**Narrative question:** “Can a simple, interpretable text model distinguish disaster-related tweets from non-disaster usage — and where does it fail?”

Do not convert correlation, a held-out metric, or a small notebook demonstration into operational, safety, or business-impact claims. Preserve meaningful **🗺️ Mental Model** diagrams in separate `# %% [markdown]` cells, contextual **⚠️ Watch Out** warnings, and evidence-grounded **⚖️ Ethical Takeaway** reflections under the learning guide. They are reasoning aids, not disposable decoration or new analytical work.

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
| Current focus and pacing | **Agreed:** MasterSchool Sprint 3 baseline first, optional learning separately labeled | No additional model is required merely to prepare public presentation |
| Canonical maintenance | **Data + Troi:** contract, guide, and project map | Spock reports evidence and proposes corrections; no silent canonical edits |
| Publication and synchronization | Both Python files exist at NLP checkpoint `8524cb7` | Remote GitHub does not establish Mac or deployed Homepage state |
| Python file roles | **Decided:** primary Sprint 3/baseline `disaster_tweets_classification.py`; supplementary MiniLM `disaster_tweets_huggingface.py` | No second school baseline or parallel notebook |
| Repeated/conflicting labels | **Approved and implemented:** lowercase/whitespace comparison, exclude conflicting groups, keep first consistent occurrence | Preserve §4.1; do not reopen policy during educational edits |
| Split and seed | **Approved and implemented:** stratified 80/20, `random_state=42`; 5,986 training / 1,497 holdout | Preserve exact membership, order, labels, and source indices |
| Baseline and portfolio | **Complete:** explained baseline in `b106c98`; portfolio case study in `7a4007b` | Preserve reported holdout evidence and limits; do not reopen completed work merely because a new chat starts |
| DIKW → Better Questions | **Agreed:** Information contextualizes Data; Knowledge explains supported patterns; insight bridges understanding and judgment; Wisdom may justify a better question, action, decision, or pause | Questions must arise from changed understanding, not the next code operation; no automatic next block |
| Colab workflow | **Deferred; unresolved** | Separately authorize export, execution, and sharing |
| CV/model comparison | **Completed in `53d473e` and `ece01b8`** | Six configurations, same ten training-only folds; original holdout remains unchanged |
| Teaching-material publication | `guided.ipynb` publication explicitly authorized; independent rights verification not established here | Both reference files remain unchanged; no broader reuse or publication permission |
| Retrospective presentation/recording | **Not created** | Do not infer a recording from prior coursework |
| Naive Bayes | **Completed and published in `ece01b8`** | Same folds and n-gram settings; F1 0.7219 / 0.7113; no holdout reuse |
| LSTM | **Deliberately excluded by Data on 2026-10-28** | Not required by Sprint 3; no LSTM or other neural-network training in DA112 |
| Hugging Face / MiniLM | **Published in `8524cb7`; both scripts report local execution** | DistilBERT sentiment and separate frozen-MiniLM CV/retrospective holdout; not independently rerun by this docs edit |
| Homepage handoff | **Entirely owned by `zvuk1T/zarko-vukovic-homepage` by Data's decision** | NLP supplies sources only; Homepage decides pages, cards, graphics, links, build and publication |

## 10. Project completion criteria

### Core analytical completion

The preparation checkmarks record the published implementation and Spock's reported runtime verification, not a fresh execution by Troi.

- [x] Repo identity and historical local path verified; recheck live Git state before work.
- [x] Source data audited; duplicate/label conflicts handled under the approved §4.1 policy.
- [x] Reproducible train/holdout separation reported and documented.
- [x] Minimal tweet-aware preprocessing implemented with original text preserved.
- [x] Current teaching standards reflected in the published analysis, including first-use explanations, standalone ASCII maps, precise claim boundaries, and contextual ethics.
- [x] TF-IDF and Logistic Regression fitted without holdout or CV-fold leakage in the published baseline.
- [x] Holdout F1/precision/recall/accuracy/support/confusion matrix published from the fixed 1,497-row evaluation set.
- [x] Model errors, limitations, and DIKW-driven conclusions documented after evaluation.

### Portfolio communication and integration (separate from model correctness)

- [x] Full authoritative `.py` analysis readable end to end, with reproducible execution instructions.
- [x] Baseline README covers its original question, data, evidence, errors and limits, but does not yet list every later lab.
- [x] Both Python sources identify evidence for a Homepage handoff, with the school baseline and extensions distinguishable.
- [ ] Homepage project separately approves and verifies presentation, visual assets, destinations and publication. **Not an NLP completion criterion.**
- [x] Original school deliverable expectations identified without claiming unperformed historical submission.
- [ ] Any separately approved Colab export or presentation/recording truthfully labeled and verified; neither is implied by case-study completion.
- [x] Classical CV/Naive Bayes, pretrained DistilBERT sentiment and frozen MiniLM are documented as **Beyond the baseline**; they do not change original school completion criteria.

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

**NLP source checkpoint:** `8524cb7cd28816940be19465780a576f09862616` contains the two Python files; original baseline `b106c98`, guided CV `53d473e` and Naive Bayes `ece01b8` remain separate historical evidence. Do not repeat already completed modeling simply for website preparation.

**Public presentation hierarchy:** **MasterSchool Sprint 3 → data preparation → TF-IDF + Logistic Regression baseline (fixed holdout F1 0.7308) → false positives, missed labels and limits** is the primary story. Classical CV, sentiment inference, and frozen MiniLM are **Beyond the baseline**, with their own denominators, task labels, pretrained information and retrospective evidence limits. README remains baseline-oriented; any NLP documentation revision needs distinct approval.

**Next location:** All decisions and work for the Homepage's cards, featured/project placements, Overview, assets, links, rendering, local builds, Git, deployment and publication occur **only in `zvuk1T/zarko-vukovic-homepage`**, governed by its own canon and Data's authorization. No website changes are authorized in this NLP repo. This documentation reconciliation is not a fresh Python run, clean-environment reproducibility check, local Git synchronization, or publication approval.

**Learning PASS: UNASSESSED.** STOP at the authorized repository and task boundary.
