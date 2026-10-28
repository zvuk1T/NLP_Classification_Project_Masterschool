# DA112 — Working and Learning Contract

Repository: `zvuk1T/NLP_Classification_Project_Masterschool` (public).
**Understanding before automation. WHY before HOW. Question before method. Evidence before conclusion.**

## Read first; ownership

This is the sole binding repository-level operational contract, subject to higher-priority instructions and Data's latest explicit task. Read it and [HOW-WE-WORK-AND-LEARN.md](HOW-WE-WORK-AND-LEARN.md) in full when starting/resuming repository work and after synchronization. Read the current decisions, status, and relevant requirements in [DA112_PROJECT_MAP.md](DA112_PROJECT_MAP.md). Every Spock handoff must name these documents; thereafter revisit relevant guidance, not unrelated audits.

Data owns decisions and acceptance; Troi leads learning design/review; Spock leads technical implementation/verification. All may challenge assumptions with evidence; roles grant no permission. **Data and Troi maintain all three canonical documents**, normally through approved remote edits. Spock proposes corrections but edits them only under explicit reassignment. The guide owns teaching detail; the map owns requirements, decisions, history, and roadmap. Other repositories are references, not contracts.

## Authorization and Git

Review is read-only. A clear edit request authorizes named content changes, not unrelated work. Do approved work without repeated confirmation; raise consequential uncertainty before the dependent action and continue unaffected work when safe. STOP on Data's instruction.

Staging, commits, pushes, PRs, synchronization, publication, and visibility changes require scope-specific authorization. An explicit request to edit named files directly on GitHub includes its necessary commit, not a local pull or PR. Before transport, verify repository, branch, current revision, exact files, diff, and sensitive-material exposure. A plain affirmative to an exact proposal is enough.

Preserve local work and untracked/ignored materials. No force push, destructive reset/cleanup, automatic stash, or silent conflict resolution. Fetch/pull/checkout change local state. Report local and remote state separately; GitHub cannot verify Data's computer.

## Scope, format, and execution

Reconstruct MasterSchool DA112 in coherent approved blocks, not per-cell approval loops or the entire project at once. Faster pacing reduces administration, not explanations. Portfolio presentation follows sound evidence.

Root `disaster_tweets_classification.py` is the sole implementation: VS Code `# %% [markdown]` and `# %%` cells. Preserve `Notebooks/NLP_Classification_Project_Masterschool.ipynb` as historical learner work. School/mentor notebooks remain references; a derived Colab export needs separate approval.

Inspect affected files and inputs directly. Verify interpreter, working directory, dependencies, and data before execution. Use the existing project `.venv`; installations, environment changes, downloads, reorganization, or infrastructure require authorization. From the verified Mac repository root, run `.venv/bin/python disaster_tweets_classification.py`. For Interactive Window, choose that environment, restart the kernel, and run cells in order. Report unavailable GUI checks; do not infer execution from saved outputs.

## Source-grounded learning

MasterSchool defines the assignment and supplies the primary coding reference; DataCamp supplies verified teaching examples. Before analytical handoffs/coding, inspect relevant actual school/mentor cells and DataCamp instructions/available code using the [source guidance](HOW-WE-WORK-AND-LEARN.md#4-complexity-and-the-datacamp-bridge). Titles, inventories, or old chats are not code reviews. Distinguish starter code, learner work, official solutions, and assistant adaptations. Disclose missing evidence; reuse applicable verified sources instead of auditing whole tracks.

Follow the guide's **Main Question → Learning → Better Questions** cycle at meaningful-unit level. Connect purpose, relevant questions, method, verified evidence, interpretation, transferable insight, and a justified next question, decision, or pause. Do not manufacture reasoning history, causality, business impact, or learner understanding. Answer direct questions directly; no compulsory quizzes or SMART checklists.

**Explain each important new table, column, feature, mask, grouped result, and intermediate object at first use inside its executable cell.** Nearby comments explain inputs, transformation, outputs, meaning, and purpose, including row/index meaning where relevant. Markdown cannot replace this walkthrough. Explain mechanisms, not just method names; avoid narrating trivial lines.

Prefer direct top-to-bottom code and visible intermediate steps. Simplify before commenting; explain unfamiliar concepts with verified relevant DataCamp references. Levels guide teaching, not blanket technique bans. No unnecessary helpers, classes, dependencies, or defensive scaffolding. Preserve authentic learner/mentor code; refactor assistant-authored analysis only within explicit scope.

Keep essential analytical checks readable. Do not add assertions, hashing, repeated-split checks, or test infrastructure to the teaching file by default. Preserve essential input/path safeguards; perform additional authorized verification temporarily and report it. Do not hide failures or replace many assertions with many branches.

Apply the guide's [writing standard](HOW-WE-WORK-AND-LEARN.md#writing-and-student-facing-voice): 1–3 sentences per meaningful explanation, a fourth only when necessary; target 15–20 words, normally at most 25 per sentence. Preserve essential reasoning and first-use inline explanations. Keep useful emoji headings. Each compact ASCII mental map belongs in its own `# %% [markdown]` cell, with a fenced `text` diagram; no Mermaid or Unicode diagram characters. Add one evidence-grounded **⚖️ Ethical Takeaway** only where relevant, without invented harms; follow the [visual-aid guidance](HOW-WE-WORK-AND-LEARN.md#optional-aids-ascii-maps-and-ethics).

## Analytical and public-source safeguards

Preserve the raw CSV and the map's approved preparation, labels, retained rows, and split. No silent policy changes or guessed correct labels. Fit learned transformations on training data only; never tune against the holdout. Baseline: TF-IDF + Logistic Regression; primary metric: positive-class (`target=1`) F1, with supporting metrics. Kaggle's unlabeled test data is not labeled evaluation. CV, tuning, Hugging Face, export, and website work need separate approval.

Separate observation, interpretation, hypothesis, and recommendation. Label disagreement does not establish ambiguity or truth; text separation does not prove event independence. Never reuse school scores as project results.

Keep credentials, private notes, session links, and unapproved protected course material out of this public repository. Existing publication does not establish redistribution rights; ignore rules do not erase history. Changing or republishing learner/mentor materials requires explicit approval and appropriate rights review.

## Verify and finish

Review the actual diff, explanations, and outputs. Verify changes proportionately: documentation checks are not model execution; Git transport is not analytical or educational acceptance.

- **Technical PASS:** relevant execution and analytical checks succeeded; separate new, reused, and unverified evidence.
- **Teaching PASS:** purpose before code, mechanisms inside code, and supported insight afterwards. Headings, emojis, and execution alone are insufficient.
- **Learning PASS:** Data independently explains and transfers the reasoning; otherwise **UNASSESSED**.

Report changed paths, evidence, limitations, relevant Git state, and the next proposed question or decision; STOP at the approved boundary. No automatic next block or Git transport. Keep knowledge in these existing documents and beside the analysis, not new logs, templates, or parallel instruction systems.
