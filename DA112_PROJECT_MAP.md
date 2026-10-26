# DA112_PROJECT_MAP.md

> **Project:** MasterSchool DA112 — NLP Classification (Disaster Tweets)
>
> **Version:** 1.0 — 2026-10-26
>
> **Status:** Source-grounded project plan. **This file does not authorize implementation.** Data approves each bounded work slice.
>
> **Repository:** https://github.com/zvuk1T/NLP_Classification_Project_Masterschool
>
> **Purpose:** A persistent, self-contained source of truth for Spock and Data, with requirements, methodological boundaries, verification gates, and a small-step execution sequence.

---

## 1. Mission and non-negotiable direction

**Primary task:** Build an understandable, independently developed **binary classifier** that predicts whether a tweet describes a **real disaster** (`target = 1`) or **not a real disaster** (`target = 0`).

- **Learning first; portfolio second.** This is a retrospective reconstruction of an unfinished MasterSchool exercise, not a claim that a new school submission or live presentation has already occurred.
- **WHY before HOW.** The learner must be able to explain the problem, dataset, split, text processing, TF-IDF, model, metrics, decisions, and limitations.
- **Minimum sound approach:** tweet-aware text preprocessing → TF-IDF → Logistic Regression → honest evaluation.
- **Data Literacy / DIKW:** Data → Information → Knowledge → Wisdom. Separate observed evidence from interpretation, and interpretation from recommendations.
- **Keep it school-level:** follow the guided Sprint 2 methodology, adapting what is genuinely needed for tweets. Avoid unnecessary production engineering, deep learning, elaborate architecture, and unrequested extensions.
- **No invented results or business impacts.** Never reuse example scores from lessons, previous portfolio claims, or another dataset as results of this project.
- **Homepage integration is out of scope** until Data separately authorizes it.

## 2. Source authority, scope, and conflict resolution

Sources have different purposes. Do not flatten them into a single set of compulsory steps.

| Role | Source | What it governs |
|---|---|---|
| **A — School requirements** | User-provided MasterSchool **DA112 Sprint 3 — Self-paced** “Introduction & Key Takeaways” screenshots (2026-10-26) | The original assignment, required deliverables, and what was optional |
| **B — Guided methodology** | User-provided MasterSchool **DA112 Sprint 2 — Guided** lesson screenshots | Beginner/intermediate workflow and pedagogical reference; not reusable tweet-model results |
| **C — Mentor learning material** | `learning_materials/mentor_live_sessions/untitled41.py` | Concepts taught in live lessons, where relevant and technically appropriate; not a school rubric |
| **D — Dataset and competition facts** | Kaggle official Disaster Tweets competition | Dataset semantics, competition data schema/rules, and F1 evaluation |
| **E — Actual project state** | Live inspection of this GitHub repository plus a separate local Git preflight | What files/code really exist; never infer local status from GitHub |
| **F — Optional post-core learning** | MasterSchool Hugging Face and GPT/Llama closing lessons; Hugging Face official documentation | Small practical learning experiment **after** the baseline, not a Sprint 3 requirement |

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

**Decision:** After the baseline project is sound, do **one small, explainable Hugging Face inference exercise**. It is educational enrichment, not an alternative to completing the core school project.

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

**Important state distinction:** Remote `main` was inspected. The local clone location, current local branch, clean/dirty status, ahead/behind state, Python environment, and local files **have not been independently verified here**. Treat them as unknown, not clean or synchronized by assumption. This commit hash is a historical audit snapshot, **not** a perpetual expected HEAD after future commits.

### 4.1 Dataset audit facts (exact-text analysis of the tracked CSV)

| Measurement | Verified value |
|---|---:|
| Total labeled rows | 7,613 |
| `target = 0` (non-disaster) | 4,342 |
| `target = 1` (disaster) | 3,271 |
| Missing `text` or `target` | 0 |
| Unique exact text strings | 7,503 |
| Additional rows from exact repeated texts | 110 |
| Distinct text strings appearing more than once | 69 |
| Distinct text strings carrying **both labels** | 18 |

**Data-quality decision gate:** Duplicate handling is **not decided**. Never automatically delete or relabel these observations. Before splitting, propose and justify a transparent policy for identical texts (including contradictory labels), report how many observations it affects, and obtain Data's approval. Ensure identical text does not silently appear in both train and test, artificially improving results. Check collisions introduced by any normalization as well.

**Dataset provenance:** The public Kaggle competition provides more fields (`id`, `keyword`, `location`, `text`, and `target` for training); this repository contains only `text` and `target`. The exact transformation history is not documented. Use the supplied text-only CSV for the school-level baseline unless Data authorizes restoring other columns. Kaggle `test.csv` is unlabeled; never invent test labels or report a labeled holdout F1 for that file.

### 4.2 Mentor-code audit and publication boundary

`untitled41.py` was exported from Colab and mostly demonstrates **time-series forecasting** on `airline-passengers.csv` with XGBoost, `TimeSeriesSplit`, `GridSearchCV`, transformers/pipelines, and rolling features. It has incomplete illustrative fragments and undefined names; it is **not a runnable end-to-end NLP solution**.

Use selectively for understanding `Pipeline`, cross-validation, and tuning concepts. Do **not** reuse `XGBRegressor`, `TimeSeriesSplit`, time lags, R², or time-series preprocessing for this binary tweet-classification problem.

**Publication issue:** Earlier setup instructions intended mentor materials to remain local-only, but the currently inspected `untitled41.py` is committed in a **public** repository. Do not claim it is private. Do not republish, expand, remove, or rewrite mentor-owned material without checking redistribution permission and obtaining Data's approval. Local Git ignore settings do not hide already committed public history.

## 5. Methodology contract for the first reliable baseline

These are **project design constraints**, not claims that any model has been implemented.

### 5.1 Data and leakage

- `X = text`; `y = target`. Preserve the original tracked CSV untouched.
- Validate labels, missing/empty text, class distribution, exact duplicates, conflicting labels, and potential tweet-specific artifacts.
- Decide the duplicate policy **before** train/test construction. Do not tune that policy to maximize test score.
- After approval, use a documented, reproducible holdout design (the guided course uses **80/20**; stratification and a fixed `random_state` such as `42` are reasonable defaults). Confirm no overlapping examples remain across partitions under the approved duplicate policy.
- Keep the holdout isolated from vocabulary learning, feature selection, hyperparameter decisions, and model fitting.
- Do not use the school's example `predict(X_train)` pattern to report generalization performance.

### 5.2 Text handling

- Start with a **minimal, understandable** preprocessing policy: handle missing/blank input, normalize selected noise, and explain the choice.
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
- Explain false positives vs. false negatives in **tweet-classification** terms. Modest class imbalance (**4,342 vs. 3,271**) makes accuracy alone insufficient.
- No made-up score thresholds, claims of real emergency-response impact, or assertions that an educational classifier is deployable.

### 5.4 Optional robustness improvements — only after a valid baseline

- If Data approves, use K-fold cross-validation **only on the training portion**; keep the holdout separate.
- For CV, pass the full `Pipeline(TfidfVectorizer, LogisticRegression)` so TF-IDF is fitted **inside each fold**. Do not CV previously fitted full-training TF-IDF features.
- If Data approves tuning, use a **small, explained** search (e.g., a few values of Logistic Regression `C` or TF-IDF `ngram_range`). Never run a broad grid merely because it appears in mentor material.
- Select/tune using training-fold results; use the holdout for a final unbiased assessment, not repeated model selection.
- Alternative models (Naive Bayes, Random Forest, neural networks) are **not required** and need separate approval.

## 6. Delivery plan — one bounded work slice at a time

**Rule:** Completing one slice does not authorize the next. Each slice ends with evidence, changed-file inventory, a plain-language explanation, and **STOP for Data's approval**. When Data approves a slice, finish its permitted work without unnecessary intermediate permission requests.

### Slice 0 — Local and remote repository preflight (**NEXT**; READ-ONLY)

1. Locate the correct local clone, without assuming an old Mac path.
2. Verify local repo identity, `origin`, current branch, `git status`, local HEAD and remote `origin/main`; check ahead/behind/divergence.
3. Compare local/remote files relevant to this map; identify any untracked, private, or mentor files.
4. Confirm Python/Jupyter environment only by inspection; **do not install** dependencies.
5. Report exact local path, branch, SHA/status, blockers, and a safe synchronization recommendation.

**Acceptance:** Remote/local facts are distinguished; no changes made; ambiguities explicitly reported. **STOP.**

### Slice 1 — School requirements and dataset quality (first content work)

1. Read this map, the official Sprint 3 requirement summary, the notebook, CSV, and mentor reference.
2. Inventory the notebook's reusable starter cells and nonportable paths.
3. Recheck data shape, labels, blanks, exact duplicates, label conflicts, and text examples.
4. Explain how tweet text differs from movie-review sentences.
5. Propose **one** minimal duplicate/holdout policy, with alternatives only if material.
6. Summarize exactly which files would change in the next slice.

**Acceptance:** The dataset issues and the proposed decision are understandable; **no split/model training yet**. **STOP for Data's data-policy decision.**

### Slice 2 — Portable notebook, approved preprocessing and holdout

1. Fix data loading portably for the **approved** local and/or Colab workflow; do not hardcode another person's absolute path.
2. Preserve original data, implement the agreed duplicate policy transparently, and document before/after counts.
3. Create the approved reproducible train/holdout split.
4. Add only the minimal tweet-aware preprocessing that can be explained.
5. Verify the split's class proportions and the agreed no-overlap invariant.

**Acceptance:** Notebook runs through split/preprocessing without leakage or unjustified deletion, but **has no trained model yet**. **STOP.**

### Slice 3 — TF-IDF + Logistic Regression baseline and honest evaluation

1. Add the simple scikit-learn text-classification pipeline; explain TF-IDF and model logic.
2. Fit on training examples only.
3. Predict on the untouched holdout and compute binary F1, precision, recall, accuracy, and confusion matrix.
4. Interpret error types and at least a few illustrative mistakes without exposing labels to training logic.
5. State limitations, reproducibility details, and what has **not** been established.

**Acceptance:** A repeatable, leakage-resistant, educational baseline and evidence-backed metrics exist. **STOP.**

### Slice 4 — Optional CV and modest tuning (**requires separate approval**)

- First explain whether CV/tuning will materially improve learning.
- If approved, perform a minimal training-only, full-pipeline CV comparison.
- Tune only a small, justified parameter set if still useful; record results separately from final holdout.
- Reassess the holdout only as the final evaluation for the chosen approach, not an optimization loop.

**Acceptance:** Gains/trade-offs are evidenced, not assumed. If omitted, the baseline remains a valid core project. **STOP.**

### Slice 5 — School-style communication and deliverables (**requires approval**)

- Make the notebook readable end to end: question → data → decisions → method → evidence → limitations.
- Verify the notebook's Colab accessibility, executable cells, understandable commentary, and actual outputs.
- Draft a concise retrospective presentation outline covering the topics required by the original school assignment.
- Treat a recording as **not yet produced** until a real, verified recording exists. Do not claim submission.
- README clean-up and any portfolio adaptation are separate authorized edits, not automatic side effects.

**Acceptance:** A reviewer can reproduce the model and follow the DIKW reasoning. Label historical and newly created deliverables truthfully. **STOP.**

### Slice 6 — Optional Hugging Face mini-lab (**after core completion, separate approval**)

A deliberately small introduction, not a Transformer rebuild:

1. Read one real Hugging Face model card: model task, dataset, limitations, license, and input/output expectations.
2. Use `transformers.pipeline` in a **separate learning cell/notebook** with a small text-classification inference example; no fine-tuning.
3. Explain that **sentiment ≠ disaster detection**. A negative sentiment label is not evidence of a real disaster.
4. If approved, try a small zero-shot disaster/non-disaster demonstration. Present it as an exploratory illustration, not automatically a comparable benchmark.
5. Compare conceptually: bag-of-words/TF-IDF + trained linear classifier vs. tokenizer + pretrained Transformer inference.
6. No LLM deployment, chatbot, RAG integration, GPU provisioning, or cloud services in this mini-lab.

**Acceptance:** Data can explain what the model does, where it came from, and why task alignment matters. **STOP.**

## 7. Data Literacy, DIKW, and storytelling contract

All narrative claims must follow this ladder:

1. **Data:** What was observed? (rows, class balance, example text, duplicate counts, raw predictions)
2. **Information:** What did our transformations/metrics quantify? (F1, precision/recall, confusion matrix, error counts)
3. **Knowledge:** What cautiously explains these results? (language ambiguity, tokenization, class mix, mislabeled duplicates, model assumptions)
4. **Wisdom:** What justified next step follows, given the evidence and limitations? (e.g., inspect false negatives, try a justified CV experiment — **not** “improves emergency response by X%”)

**Narrative question:** “Can a simple, interpretable text model distinguish disaster-related tweets from non-disaster usage — and where does it fail?”

Do not convert correlation, a held-out metric, or a small notebook demonstration into operational, safety, or business-impact claims.

## 8. Spock operating protocol and authorization

At the start of each new session:

1. Read this `DA112_PROJECT_MAP.md` **in full** and identify the current explicitly approved slice.
2. Check current repository/files before describing implementation state; do not treat the snapshot above as live.
3. State **WHY**, then the bounded **HOW**, expected files, checks, and a clear STOP condition.
4. If required source material or a material choice is missing, say what is missing rather than fabricate a lesson, score, file, or requirement.

During an approved slice:

- Work only in the correct existing repository. **Never create a substitute repo** or repurpose another MasterSchool project.
- Respect all applicable `AGENTS.md` and higher-priority instructions; this map does not replace them.
- Preserve user's existing work. No force push, `reset --hard`, destructive clean-up, automatic stashing, or silent conflict resolution.
- Do not edit outside the approved file/path scope. Avoid installing or importing unnecessary advanced libraries.
- Never silently rewrite the mentor material or copy its code into the notebook.
- Show meaningful tests/results, file changes, unresolved limitations, and local/remote Git state.
- **Commit/push only if the current task explicitly authorizes that transport.** Do not assume approval to edit means approval to publish.
- End every slice with **STOP**; await authorization for the next.

**Approval overrides:** Data may explicitly change priorities or choose a different sound approach. Record the change and its reason in the work report or a separately approved document revision. Do not quietly reinterpret a new instruction as permission to modify the whole project.

## 9. Decision register (do not silently resolve)

| Decision | Current state | Required handling |
|---|---|---|
| Local working-tree status and synchronized branch | **Unknown** | Verify in read-only Slice 0 |
| Exact handling of repeated/conflicting tweet labels | **Unresolved** | Explain options and get Data's decision before train/test |
| Colab data-loading and sharing workflow | **Unresolved** | Choose a portable path in authorized Slice 2 |
| Final split counts and random seed | **Proposed:** 80/20, stratified, fixed `42` | Confirm after duplicate policy; report actual counts |
| CV/tuning | **Optional, not approved** | Separate learning decision after baseline |
| Mentor file redistribution permissions | **Not verified; file currently public** | Do not assume permission; Data decides corrective action |
| Retrospective presentation/recording | **Not created** | Discuss only in authorized communication slice |
| Hugging Face mini-lab | **Desired, not yet implemented** | Separate post-core optional slice |
| Website/homepage | **Out of scope** | New explicit authorization only |

## 10. Project completion criteria

### Core analytical completion
- [ ] Repo path, current Git state, and provenance verified.
- [ ] Source data audited; duplicate and label conflicts handled under an approved, documented policy.
- [ ] Reproducible and defensible train/holdout separation.
- [ ] Preprocessing is tweet-aware and explainable.
- [ ] TF-IDF and Logistic Regression fit without holdout or CV-fold leakage.
- [ ] Real, reproducible holdout F1/precision/recall/accuracy/confusion matrix.
- [ ] Explicit limitations, model errors, and DIKW-driven conclusions.

### Communication completion (separate from model correctness)
- [ ] Readable notebook, portable or documented Colab execution path.
- [ ] Original school deliverable expectations identified without claiming unperformed historical submission.
- [ ] Any presentation/recording truthfully labeled and actually accessible, if produced.
- [ ] Optional extensions marked as optional, with their own evidence and approval.

**Not part of core completion:** Kaggle leaderboard rank, deep learning, Transformer fine-tuning, deployment, homepage integration, or guaranteed real-world impact.

## 11. Source register

**Primary supplied evidence (screen captures shared 2026-10-26; not currently tracked in the repo):**
- Sprint 3 “Introduction & Key Takeaways” and deliverables: `SCR-20261026-snhr.png` (also a second capture `SCR-20261026-skzi.png`).
- Sprint 2 overview and guided lessons: `SCR-20261026-slux.png`, `SCR-20261026-slyo.png`, `SCR-20261026-smbw.png`, `SCR-20261026-smgd.png`, `SCR-20261026-smix.png`, `SCR-20261026-smmj.png`, `SCR-20261026-smpm.png`, `SCR-20261026-sndx.png`.
- Closing Hugging Face and GPT/Llama lessons: `SCR-20261026-sobc.png`, `SCR-20261026-snrc.png`.

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

---

## 12. Current execution instruction

**The next authorized work, when Data requests it, is Slice 0 (read-only local/remote preflight).**

Do **not** start a new notebook, train a model, reorganize the repository, modify the mentor file, install packages, write a presentation, or perform Hugging Face experiments merely because this map describes future slices.

**Spock: read the map, verify the facts relevant to the requested slice, do only that slice, report, and STOP.**
