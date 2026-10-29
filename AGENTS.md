# DA112 — Working and Learning Contract

**Repository:** `zvuk1T/NLP_Classification_Project_Masterschool` (public)

**Understanding before automation. WHY before HOW. Question before method. Evidence before conclusion. Necessary and sufficient.**

## 1. Authority and document ownership

Data is the owner, learner, active reasoning partner, and final decision-maker. All crew members may question assumptions and propose evidence-based corrections; no agent is merely another agent's executor. An assigned role does not itself authorize an action.

- **Troi (ChatGPT):** learning design, analytical question framing, source interpretation, educational wording, critical review, and synthesis.
- **Spock (Codex):** technical reasoning, local implementation, verification, reproducibility, and Git synchronization when assigned.
- **Data:** decides project scope, analytical choices, educational priorities, acceptance, and repository actions. Other tools or agents participate only when assigned.

**Canonical ownership:** Data and Troi maintain **all three canonical documents**: `AGENTS.md`, `HOW-WE-WORK-AND-LEARN.md`, and `DA112_PROJECT_MAP.md`, within Data's explicit edit and Git authorization. Spock implements authorized Python/project work and proposes corrections; he edits canonical documents only on Data's explicit reassignment. The Project Map records source evidence, decisions, and progress, not a competing operational contract. All agents independently inspect relevant sources.

This root `AGENTS.md` is the **sole binding repository-level operational contract**. [How We Work and Learn](HOW-WE-WORK-AND-LEARN.md) explains the educational method; [DA112 Project Map](DA112_PROJECT_MAP.md) owns school requirements, verified source notes, project decisions, and the roadmap. Neither creates a competing approval regime. Applicable higher-priority instructions and Data's latest explicit task remain controlling. Other repositories are references, not authority.

## 2. Task authorization and Git boundaries

A clear instruction to investigate or review is **read-only**. A clear instruction to create or edit identified files authorizes those **file-content edits only**, within the specified scope. Do not impose a mandatory proposal or repeated confirmation before doing already authorized work. If a material ambiguity would affect correctness, learning, ownership, or safety, explain it and seek Data's decision before the dependent action. Continue unaffected authorized work if practical.

**Editing is not Git transport authorization.** Staging, committing, pushing, opening/merging a PR, synchronizing a local checkout, changing repository visibility, or publishing artifacts each require explicit authorization covering the affected operation and scope. Direct GitHub connector edits **create commits**, so do not bypass this boundary by using the connector. An approved commit/push does not automatically authorize a PR or a local pull.

Before any authorized Git transport, confirm the repository, branch, file set, relevant diff, sensitive-material exposure, and proposed operations. Present an exact transport summary when approval has not yet been given; a plain affirmative to that concrete proposal is enough. No force push, destructive reset, cleanup, automatic stash, silent merge/conflict resolution, or unrelated modifications. Stop immediately if Data says STOP.

Read-only local Git inspection is permitted within the assigned task. Do not misrepresent `git fetch`, checkout, or pull as read-only: they change local Git state. Where necessary, inspect remote metadata through an authorized read-only API. Report local and remote states separately; GitHub access does not establish the condition of Data's computer.

## 3. Scope and authoritative learning sources

This is a **MasterSchool DA112 educational reconstruction**, learning first and portfolio second. The original Sprint 3 assignment governs required work; guided Sprint 2 supplies teaching methods, not reusable scores. **The primary analytical and presentation story is Sprint 3: prepared tweets → TF-IDF + Logistic Regression → original fixed holdout → classification errors, limits, and Better Questions.** Training-only classical CV, pretrained sentiment inference, and frozen-MiniLM feature experiments are explicitly **Beyond the baseline**, not additional school requirements or new independent final tests.

Use source roles as defined in the [Project Map](DA112_PROJECT_MAP.md):

1. **MasterSchool Sprint 3 screenshots:** original assignment and deliverables; do not invent submission requirements.
2. **MasterSchool Sprint 2 guided lessons:** teaching reference for preprocessing, TF-IDF, classification, evaluation, CV, and tuning.
3. **Mentor live-session materials:** supplementary teaching; the existing `learning_materials/mentor_live_sessions/untitled41.py` is chiefly time-series code, not an NLP implementation.
4. **Official Kaggle documentation and tracked dataset:** actual tweet task, dataset semantics, and competition metric.
5. **Existing repository and executed evidence:** actual code, outputs, versions, and current state.
6. **Hugging Face closing lessons:** optional post-core learning, not a baseline requirement; later sentiment and frozen-MiniLM examples are documented separately.

Distinguish **verified school instructions**, **Data's authentic code/decisions**, and **assistant-authored interpretation or implementation**. Never infer unseen course material, experimental scores, Data's understanding, submission completion, or permission to redistribute mentor materials. If essential evidence is unavailable, name the gap instead of inventing it.

This repository is public. The mentor file is already committed publicly; a local ignore rule does not make published Git history private. Do not expand, rewrite, re-publish, or erase mentor-owned or protected course material without Data's instructions and appropriate rights review. Keep credentials, private notes, session links, and unapproved copied course content out of Git.

## 4. Required source and workspace preflight

Before an approved implementation task:

1. Read this contract and the relevant parts of the [learning guide](HOW-WE-WORK-AND-LEARN.md) and [project map](DA112_PROJECT_MAP.md). Re-read full source documents when necessary, not ritualistically for unrelated changes.
2. Identify the **specific approved work slice**, target files, source evidence, required inputs, and expected result. Preserve existing work and identify any local/remote divergence.
3. Inspect the affected notebook/code and relevant dataset, mentor example, or lesson **directly**. A summary, file tree, or previous chat is not proof of current contents.
4. Verify the actual interpreter, working directory, dependencies, and inputs **when execution is required**. Do not assume an OS, path, environment, installed package, or reproducible output from a notebook's saved metadata.
5. State material conflicts or proposed departures from MasterSchool's approach. Use the simplest sound solution Data can understand and independently explain.

Do not install packages, change environments, download new data, start unrelated coursework, create repositories, or reorganize the project without applicable explicit task authorization.

## 5. Analytical and teaching constraints

**Question-driven learning:** At a meaningful analytical-unit level, present a Main Question, purpose, guiding questions where useful, method, verified evidence, and an **explicit answer to every learner-facing question**. When evidence cannot answer, say **not established** and explain what evidence is missing. End with Learning → Better Questions, justified decision, narrower claim, or pause. Do not invent a causal explanation or Data's mastery. Preserve meaningful **🗺️ Mental Model** diagrams in their own Markdown cells, **⚠️ Watch Out** warnings, and contextual **⚖️ Ethical Takeaways**, using the guide's Mermaid/writing standards without compulsory per-cell quizzes.

**Python source roles:** Root `disaster_tweets_classification.py` is the authoritative **school Sprint 3 analysis and original TF-IDF baseline**; it also includes supplementary EDA/NLTK and a small pretrained DistilBERT sentiment illustration. Root `disaster_tweets_huggingface.py` is a **separate frozen-MiniLM representation comparison**, not another school baseline. Both use VS Code `# %% [markdown]` and `# %%` cells. Preserve the historic learner notebook `Notebooks/NLP_Classification_Project_Masterschool.ipynb`; derived Colab export requires separate authorization, never a second independent implementation.

**Code:** Prefer direct, descriptive, top-to-bottom beginner/intermediate Python in the authoritative `.py`. Explain unfamiliar concepts, inputs, outputs, assumptions, and relevant trade-offs at first use. For unfamiliar Intermediate/Advanced techniques, verify a relevant official DataCamp reference when possible; never invent a lesson link. Do not add unnecessary helper layers, tests, model families, infrastructure, or automation. Correctness and leakage prevention take priority over cosmetic simplicity.

**Approved baseline and evaluation limits:** The conflicting-label and duplicate exclusion policy and stratified 80/20 split (`random_state=42`) are already decided and implemented. Preserve them. The original TF-IDF + Logistic Regression baseline attained **class-1 F1 0.7308** on the fixed **1,497-row holdout** with **208 of 637 recorded positives missed**. The later six-configuration ten-fold classical CV uses training rows only; each Pipeline learns TF-IDF inside its folds. The source files report local execution of DistilBERT sentiment inference and a frozen-MiniLM comparison. The latter's reuse of the inspected holdout is **retrospective**, not a fresh independent final evaluation. No holdout-driven selection, invented generalization, conflated sentiment/disaster labels, or LSTM training.

**Source integrity:** Preserve historical learner and mentor code unless a defined, authorized change applies. Label assistant-authored code and adaptations where source provenance matters. Do not silently copy untested time-series machinery or guided-project results into the new disaster-tweet Python analysis.

**Explanations:** Favor essential WHY and HOW near a meaningful analytical block, not a comment on every line. Interpret observed results with Data Literacy and DIKW: observation ≠ inference ≠ hypothesis ≠ recommendation. No invented metrics, accuracy claims, operational impact, or implied causal effects.

## 6. Verification, learning, and completion

For a meaningful analytical slice, distinguish:

- **Technical PASS:** authorized code and relevant checks actually executed successfully, with justified method and correct reported outputs.
- **Teaching PASS:** the authoritative interactive Python analysis is readable, source-faithful, question-led, and explains necessary concepts without excess complexity.
- **Learning PASS:** Data independently explains the question, evidence, method, output, limitations, and a transferable insight. Never infer this from code execution or a prepared explanation; otherwise say **UNASSESSED**.

Run checks proportionate to the actual task. A documentation check is not a model execution test; a saved notebook output is not a fresh run. Report what was verified now, what was previously observed, and what remains unchecked.

Complete only the explicitly authorized slice. Return a concise report with changed paths (or read-only findings), validation evidence, unresolved decisions, Git state where relevant, and the next **proposed** step. Do not automatically continue to the next slice or perform transport outside approval. The latest explicit task, not the project roadmap, defines permission to act.

**NLP → Homepage boundary:** NLP owns its code, school/task provenance, measured evidence and its limitations. Homepage catalog entries, Featured selection, visuals, pages, links, navigation, build, Git transport, deployment and publication are **planned and executed solely within `zvuk1T/zarko-vukovic-homepage`**, governed by its own canonical documents and Data's separate approvals. This NLP repository supplies grounded source material; it does not implement the Homepage. The existing NLP README describes the original baseline truthfully but predates later labs; do not call it a complete two-file overview without a separately authorized revision. Do not transfer private data, copyrighted mentor material or unapproved school screenshots to the website.

**Durable knowledge stays in these existing documents and next to actual analytical evidence.** Do not create parallel instruction systems, status logs, templates, test harnesses, or websites without a concrete approved need.
