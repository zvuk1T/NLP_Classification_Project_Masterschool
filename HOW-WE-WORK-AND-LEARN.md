# How We Work and Learn — MasterSchool DA112

This practical educational guide operates under [AGENTS.md](AGENTS.md), the sole binding repository-level operational contract. [DA112_PROJECT_MAP.md](DA112_PROJECT_MAP.md) owns school requirements, source evidence, approved decisions, status, and the roadmap. This guide does not authorize edits, Git transport, additional coursework, or another work block.

**Understanding before automation. WHY before HOW. Question before method. Evidence before conclusion.**

## 1. What we are trying to learn

The analysis should help a reader understand why an NLP question matters, how relevant data and a sound method provide evidence, and what that evidence can and cannot establish. A working classifier and a polished document do not, by themselves, demonstrate independent understanding.

Data reports having already worked through the school project. This reconstruction develops the Data–Troi–Spock collaboration and an evidence-backed portfolio case study. Work in the map's coherent blocks, with one implementation and one joint review, rather than routine approval stops. **Larger work blocks do not justify larger unexplained jumps inside the code.** Briefly recap familiar mechanics; explain new objects, consequential choices, and actual misunderstandings where needed.

The guided Sprint 2 movie-review classifier supplies teaching references; Sprint 3 Disaster Tweets is our task. Use applicable mentor examples without transferring their labels, cleaning assumptions, or scores. Hugging Face remains an optional post-baseline exercise, not a prerequisite for the core analysis or portfolio.

The authoritative Python file is both the real analysis and a learning resource. Returning readers should be able to follow the problem, choices, transformations, evidence, limitations, and lessons without reconstructing a chat.

## 2. Main Question → Learning → Better Questions

Better Questions is our philosophy of inquiry, drawing on Data's DataCamp Gym work; Main Question is its entry point for a particular analytical unit. This is our synthesis, not a verbatim DataCamp framework. [Forming Analytical Questions](https://www.datacamp.com/courses/forming-analytical-questions) supports clarifying purpose, refining relevant questions, selecting fitting methods, and returning findings to the original need.

```text
PURPOSE AND CONTEXT -> PROJECT QUESTION
                          |
                          v
MAIN QUESTION -> GOAL + RELEVANT GUIDING QUESTIONS
                          |
                          v
RELEVANT DATA -> METHOD + EXPLAINED PYTHON
                          |
                          v
VERIFIED EVIDENCE -> INTERPRETATION + LIMITS
                          |
                          v
LEARNING -> WHAT CHANGED IN OUR UNDERSTANDING?
                          |
                          +--> ETHICAL TAKEAWAY (when relevant)
                          |
                          `--> BETTER QUESTION / DECISION / PAUSE
```

This is a revisable reasoning loop, not a compulsory ladder or an endless research queue. Missing evidence can change a question before coding; results can challenge an assumption. A better question may narrow a claim, expose missing context, or justify not acting yet. It need not be harder or broader.

### Keep three question levels connected

| Level | Purpose |
| --- | --- |
| **Project Question** | What overall problem are we trying to understand, and why? |
| **Main Question** | Which meaningful uncertainty can this unit investigate within scope? |
| **Guiding Questions** | Which smaller observable checks help answer it without losing the larger purpose? |

For Disaster Tweets, a project question concerns trustworthy text classification; a preparation question concerns which examples an evaluation can describe. Counting conflicting groups is a guiding check, not the whole purpose.

Use specific, bounded, relevant, evidence-answerable questions. DataCamp's SMART formulation is **Specific, Measurable, Actionable, Relevant, Time-bound**; A means Actionable. Apply it proportionately, without five required headings, invented time ranges, or fictional business scenarios. A precise question can still be irrelevant.

### Three connected WHYs

1. **Why investigate?** What matters in this context, and what understanding or decision could the answer support?
2. **Why this approach?** Why can the available data and chosen method provide the needed evidence, and what might they miss or distort?
3. **Why learn it?** Which principle transfers to another dataset, analytical question, or AI problem?

These guide teaching and review, not a questionnaire for Data. Answer direct questions directly; do not impose quizzes or Socratic dialogue.

### Frame the task before the solution

Before a meaningful unit, distinguish the source-backed problem, relevant context and assumptions, actual school instructions versus project choices, and the expected kind of evidence. Clarify population, observation grain, labels, units, denominator, comparison, and time where they affect interpretation. Expected output is not an observed result.

Do not reverse-engineer a school requirement from existing code, hide a solution inside an investigative question, or invent a motivation the source does not supply. An algorithm does not make an unsupported question answerable.

### Show how understanding develops

At substantive turning points, connect the starting question to the evidence or limitation, the supported insight, and why it changes the next question or judgment. Keep this explanation beside the analysis, without a separate question log or new IDs.

**Question → evidence → changed understanding → significance → better question or justified decision.**

For example, a raw row total does not establish distinct inputs; normalization changes which strings count as equivalent; excluding conflicting groups changes the population a future score describes. These are connected insights, not merely a sequence of pandas commands.

Label retrospective explanations as retrospective. Do not invent Data's former beliefs, learning history, or mastery. Announcing "next, train a model" is a workflow transition, not a complete learning outcome. A next question never authorizes another task.

## 3. From question to student-readable Python analysis

**Primary format:** Root `disaster_tweets_classification.py` is the single implementation source. Use VS Code Python Interactive Window with `# %% [markdown]` for educational text and `# %%` for executable Python. Prefix every Markdown line with `#`, including blank lines and collapsible-answer content. Verify the actual interpreter and kernel before claiming execution.

Preserve the historical `Notebooks/NLP_Classification_Project_Masterschool.ipynb` unchanged. School/mentor notebooks are references. A derived Colab export from the validated `.py` requires separate authorization and must not become a second independent implementation.

### Teaching functions and emoji vocabulary

A meaningful analytical unit may span several cells. Use these functions naturally, not a full template for every import or operation:

| Element | Placement and purpose |
| --- | --- |
| **❓ Main Question** | Before code: the relevant uncertainty connected to the Project Question. |
| **📋 Context, Source & Task** | Verified source, input meaning, important assumptions, and actual instructions. |
| **🎯 Goal & Approach — What & Why** | Why investigate, why this approach fits, and what evidence is needed. |
| **🔎 Guiding Questions** | Only the useful smaller questions needed for this unit. |
| **💻 Real Code + Inline Walkthrough** | The actual executable cells, with nearby comments explaining important mechanisms. |
| **✅ Evidence & Interpretation** | After execution: verified findings, the supported answer, and its limits. |
| **🧠 Learning → Better Questions** | Transferable insight, changed understanding, significance, and a justified next question, decision, or pause. |
| **🧑‍💼 Recruiter / Transfer Check** | Optional at a meaningful milestone; a reasoning question with a concise answer under `💡 Show answer`. |

Keep the emojis as navigational cues. The code itself fills the 💻 role; a separate decorative Code or Walkthrough Markdown heading must not replace comments inside executable cells. Omit inapplicable elements, such as model metrics in a setup task.

### Explain new objects where they are created

**Every meaningful new table, column, feature, Boolean mask, grouped result, or intermediate object needs an explanation at first use inside its executable cell.** A nearby comment explains what goes in, what changes, what comes out, and why it matters. Include what one row or element represents, and what the index means, where relevant.

Markdown explains WHY before code. Comments explain important WHAT/HOW beside the operation. Markdown after execution explains evidence, meaning, and learning. None substitutes for the others.

One or two short comment lines usually cover a meaningful logical block; a third may clarify a distinct necessary point. One comment can cover related operations if each new object's role is clear. Do not narrate every assignment, repeat method names, or assume that an earlier appearance proves understanding. Explain unfamiliar method/attribute/indexer terminology only when it resolves a real gap.

**Original teaching illustration using our existing objects, not a DataCamp solution or a new policy:**

```python
# Count distinct target labels per prepared text, not observations per group.
# The Series index contains text strings; values of 2 indicate label disagreement.
labels_per_prepared_text = prepared.groupby("prepared_text")["target"].nunique()

# Extract the text keys of conflicting groups, not integer source-row positions.
conflicting_keys = labels_per_prepared_text[labels_per_prepared_text > 1].index

# Map those group keys back to one flag per source row: True means group membership.
conflict_mask = prepared["prepared_text"].isin(conflicting_keys)

# .loc selects whole rows; ~ reverses the mask to keep the other groups.
consistent_rows = prepared.loc[~conflict_mask]
```

The mechanism is the movement from source rows to group-level evidence and back to row selection. "Create a mask" or "use groupby" alone does not explain that movement. Introduce the meaning of `.index`, `isin()`, `.loc`, and `~` at the relevant first use, without repeating the explanation mechanically later.

### Optional aids: ASCII maps and ethics

Use **🗺️ Mental Model**, **⚠️ Watch Out**, and **✅ Notes for Students** near a real conceptual dependency, misconception, or practical need. They are optional aids, not quotas or decoration.

**🗺️ Mental Model:** Use plain ASCII diagrams inside fenced `text` blocks, with characters such as `|`, `-`, `+`, `>` and `v`. Keep emoji headings outside the diagram; do not use Mermaid or Unicode box-drawing characters. Keep all diagram content ASCII, including arrow symbols and labels; English labels fit our documentation language. This restriction applies to diagrams, not prose, emoji headings, or the name **Learning → Better Questions**.

In `.py` Markdown cells, prefix the fence and diagram lines with `#`. Use compact diagrams rather than wide layouts that are hard to read on a phone. Do not add a map to every section or relabel an ordinary paragraph as a diagram.

**⚖️ Ethical Takeaway — optional**

Where a meaningful learning unit raises an ethical issue, end its reflection with one concise, evidence-grounded sentence about the relevant responsibility.

Connect it to the actual data choice, interpretation, affected people, privacy, representation, or possible misuse. Do not add a generic moral message to every exercise or invent harms that the evidence does not establish.

Consider responsibility when choosing and interpreting the analysis; the closing sentence summarizes that reasoning, not a substitute for it. Technical correctness, publication permission, and responsible use are different judgments. A serious unresolved issue needs attention, not just a disclaimer. This aid creates no new analytical task or compulsory section.

**Original DA112 illustration, not an observed impact or an official DataCamp quotation:**

> **⚖️ Ethical Takeaway:** Be transparent about excluded examples, because performance on retained tweets would not establish reliability for all disaster reports.

For this preparation, a brief takeaway beside the interpretation of exclusions or the final learning reflection is useful; repeating it after every technical operation is not.

### Writing and student-facing voice

Write code, comments, and documentation in English; conversation follows Data's language. Usually use one or two focused sentences, adding a third for a necessary distinct point. Aim for roughly 10–20 words per sentence, allowing up to 23 for clarity; these are readability guidelines, not reasons to remove essential reasoning. Split genuinely different teaching points rather than creating arbitrary fragments.

Use a neutral voice useful to another learner. Retain concise provenance and consequential decisions without repeatedly narrating crew actions, approvals, or internal verification. Explain known concepts again only when useful. A written recap or prepared transfer answer never establishes Data's Learning PASS.

## 4. Complexity and the DataCamp bridge

### Actual school code and verified teaching examples

MasterSchool defines the assignment and provides the primary implementation reference. Inspect relevant actual guided/mentor cells, retain sound operation sequences, adapt them to tweets and VS Code, and identify material source errors rather than copying them. `guided.ipynb` is the relevant NLP reference; `untitled41.py` chiefly demonstrates time-series methods. See the map for their provenance and publication boundaries.

DataCamp is the primary teaching reference for accessible explanations and relevant patterns, not an automatic override of task requirements or technical correctness. Before analytical coding or handoffs, inspect relevant instructions and available code via the connector or an accessible official page. Reuse verified sources when still applicable; metadata and catalog descriptions alone are not code reviews.

| Reference | Relevant use |
| --- | --- |
| [Associate Data Scientist in Python](https://www.datacamp.com/tracks/associate-data-scientist-in-python) | Python, pandas, statistics, and basic supervised learning. |
| [Data Scientist in Python](https://www.datacamp.com/tracks/data-scientist-in-python) | Continuation material only where the approved question needs it. |
| [Intermediate Python — Driving right (1)](https://campus.datacamp.com/courses/intermediate-python/logic-control-flow-and-filtering?ex=15) | A Boolean Series, its row labels, and selection of complete records. |
| [Joining Data with pandas — Your first inner join](https://campus.datacamp.com/courses/joining-data-with-pandas/data-merging-basics?ex=3) | Input meaning, keys, row grain, and interpretation of counts. |
| [Feature Engineering for NLP in Python](https://www.datacamp.com/courses/feature-engineering-for-nlp-in-python) | Course-level starting point for relevant text-feature lessons, including TF-IDF. |
| [Sentiment Analysis in Python](https://www.datacamp.com/courses/sentiment-analysis-in-python) | Relevant classification examples; sentiment is not disaster detection. |
| [Supervised Learning with scikit-learn](https://www.datacamp.com/courses/supervised-learning-with-scikit-learn) | Relevant split, classification, pipeline, and evaluation lessons. |

These are source-selection aids, not courses Data must complete. Distinguish official starter code, saved learner attempts, verified official solutions, and assistant-authored explanations. A platform-accepted answer is not automatically an official reference solution. Name inaccessible material; never invent a lesson, code review, or result. Do not repeatedly retry an unreliable endpoint or audit whole tracks unnecessarily.

The private Gym companions and Time Series guidance are comparative teaching references. Transfer the reasoning standard, not protected course content, copied exercises, unrelated implementation, or their environment/authorization rules into this public project.

### Complexity must earn its place

Beginner-level explanation is the baseline; much Intermediate material is familiar, but unfamiliar concepts still need support. Advanced knowledge must not be assumed. Published course levels provide context, not universal difficulty labels for individual operations.

For a needed unfamiliar concept, explain its purpose, inputs, outputs, and important assumptions at first use. Give a verified relevant lesson/chapter link where possible; disclose course-only or unavailable references. A link never replaces explanation or justifies unnecessary complexity.

Prefer direct top-to-bottom operations, descriptive variables, and visible intermediate steps. Break dense chains when they hide changes in shape, index, or meaning. More readable lines can be simpler than one compressed expression. Add a helper, class, dependency, or advanced method only for a concrete requirement or material correctness/clarity benefit; raise a departure that expands scope before dependent work.

Preserve authentic learner and mentor code unless specifically authorized otherwise. The assistant-authored `.py` can be refactored within an explicit task while preserving approved analytical behavior. The Gym's learner-answer preservation rule is not a reason to freeze assistant-authored project code.

**Simplify the solution before adding more commentary.** A long explanation cannot rescue needless complexity.

### Analytical checks versus developer verification

Keep student-visible checks focused on the question: row accounting, remaining duplicates/conflicts, partition sizes, class proportions, and text overlap. Explain each check's meaning and limits; retain essential path and invalid-input safeguards. Do not print unconditional PASS claims.

Do not reintroduce executable assertions, SHA-256 preservation machinery, repeated splitting, or test scaffolding into this teaching analysis by default. Replacing many asserts with many branches does not solve the cognitive-load problem. Any genuinely necessary exception requires a specific justification and Data's decision before changing this agreed approach.

Spock remains responsible for technical correctness. Additional equivalence and preservation checks may run temporarily during authorized verification, outside the permanent teaching flow. Report what was checked, do not conceal failures, and do not create extra repository test files without authorization. For an educational refactor, equal counts alone do not establish equal retained rows, indexes, labels, membership, or order.

## 5. DA112 methodology as a chain of questions

Use the [Project Map](DA112_PROJECT_MAP.md#5-methodology-contract-for-the-first-reliable-baseline) for approved decisions and exact audit facts. Do not reopen them because a new chat starts.

| Learning question | Evidence and method | Trap to understand |
| --- | --- | --- |
| **What is a disaster-related tweet?** | Inspect the recorded task, labels, examples, and quality. | Keywords and negative sentiment are not equivalent to real disasters. |
| **Are these examples and labels reliable?** | Inspect missingness, repeats, and conflicting annotations. | Conflicting labels reveal disagreement, not its cause or the true label. |
| **How can we fairly evaluate a classifier?** | Preserve the agreed preparation and reproducible holdout. | Repeated text, related events, and holdout-guided choices limit evaluation. |
| **How should text become numbers?** | Explain TF-IDF, vocabulary, term frequency, and document frequency. | Fitting on held-out text exposes evaluation-set information. |
| **What does Logistic Regression learn?** | Fit the linear classifier on training features. | Associations and coefficients do not establish causal disaster signals. |
| **What constitutes good performance?** | F1 for `target=1`, precision, recall, accuracy, support, and confusion matrix. | A single score hides error types and limits of the evaluated population. |
| **Would robustness checks or tuning help?** | Only approved training-fold pipeline CV and modest tuning. | More computation is not automatically better understanding or generalization. |
| **What should we claim or do next?** | Return to the purpose with DIKW and bounded judgment. | A school classifier does not establish emergency-response benefit or deployment readiness. |

The agreed first model remains TF-IDF + Logistic Regression. A scikit-learn `Pipeline` binds learned transformations and model; Hugging Face's pretrained Transformer `pipeline` is a different abstraction. Explain that distinction if the optional extension is approved.

Before any separately approved preprocessing change, consider URLs, hashtags, mentions, punctuation, negation, names, and disaster terms. Do not mechanically copy movie-review cleaning. Keep training/inference preparation consistent, fit vocabulary/IDF on training data only, and never tune on the final holdout. In CV, fit the full pipeline independently inside each training fold.

## 6. DIKW, Data Literacy, and storytelling

Use DIKW to explain what reasoning adds, not as labels required in every cell:

```text
DATA: recorded observations and provenance
    |
    v
INFORMATION: organization, context, counts, and comparisons
    |
    v
KNOWLEDGE: supported interpretation, assumptions, and limits
    |
    v
WISDOM: responsible judgment in context
    |
    `--> Act / investigate / narrow the claim / pause
```

Aggregation alone does not establish an explanation, causality, or a recommendation. Before relying on a result, consider missingness, errors, representation, collection bias, and whether the evidence fits the question. Verified provenance and fitness for a purpose are separate judgments; cleaning does not automatically establish representativeness.

**Fact ≠ interpretation ≠ hypothesis ≠ recommendation.** An insight is a supported connection, not merely a number or graph. Judgment considers purpose, uncertainty, and consequences; choosing not to act yet can be justified. Do not invent stakeholders, costs, benefits, or impacts.

For our preparation, excluding groups with conflicting recorded labels does not establish that all ambiguous examples were removed or that retained labels are correct. Keep those claims separate. Explain false positives and false negatives in plain language when a model exists; a change in F1 is not evidence of lives saved or faster response.

Good storytelling makes the path from question to evidence, meaning, limitations, and judgment understandable without hiding inconvenient findings. It starts during analysis, not only in the final presentation. Retrospective documentation does not establish that a historical recording or school submission occurred.

Conceptual sources, not additional assignments:

| Source | Contribution |
| --- | --- |
| [Forming Analytical Questions](https://www.datacamp.com/courses/forming-analytical-questions) | Relevant, answerable questions; method selection; return to purpose. |
| [Introduction to Data Literacy](https://www.datacamp.com/courses/introduction-to-data-literacy) | Data quality, context, meaning, and DIKW transitions. |
| [Communicating Data Insights](https://www.datacamp.com/courses/communicating-data-insights) | Audience-aware interpretation and communication of limits. |
| [Data Storytelling Concepts](https://www.datacamp.com/courses/data-storytelling-concepts) | Coherent evidence-based narrative without distortion. |
| [Introduction to Data Ethics — bias](https://campus.datacamp.com/courses/introduction-to-data-ethics/data-for-good?ex=3) and [accountability](https://campus.datacamp.com/courses/introduction-to-data-ethics/data-for-good?ex=11) | Collection limitations, responsibility, and consequences of data use. |

The ethics references were inspected as selected lessons; this is not a claim that Data completed the course. Our project-specific ethical takeaways are original interpretations, not quoted official course answers.

## 7. How Data, Troi, and Spock collaborate

Data leads choices and acceptance. Troi leads source-aware educational design; Spock leads authorized implementation and technical verification. Both may challenge assumptions with exact evidence and a simpler sound alternative; neither is merely the other's executor.

**Canonical ownership:** Data and Troi maintain `AGENTS.md`, this guide, and `DA112_PROJECT_MAP.md`. Spock reads them and proposes corrections; authoring changes requires an explicit reassignment. Direct remote edits and local synchronization follow the contract's separate authorization boundaries.

For an approved analytical block: ground the question in current sources, agree purpose and material choices, implement the bounded work, verify results and teaching, then review the complete outcome. Do not add repeated approval ceremonies for routine operations. Completion, a roadmap, or a good next question does not authorize another block or publication.

When exact educational wording is approved, identify the file, baseline, anchor, and complete replacement or addition. Spock flags stale anchors, source contradictions, or consequential objections before changing the approved meaning. KEEP / REPLACE / ADD can help precise integration, but they are an optional technique, not a new workflow or template.

Every analytical handoff names the current canonical documents and carries the question, purpose, sources, expected evidence, allowed files/actions, and stopping point. Reference durable rules instead of pasting the entire contract again. Preserve reasoning where it matters; keep routine instructions concise.

## 8. Review and continuity without bureaucracy

Review actual code, prose, source fidelity, and outputs, not just an implementation report. Read a unit's opening without its answer: is the question, input meaning, and expected evidence understandable? Then inspect whether important new objects are explained inside executable cells and findings support the stated insight.

At meaningful milestones, distinguish:

- **Technical PASS:** authorized methods and outputs checked through appropriate execution and analytical verification; distinguish fresh checks, reused evidence, and unverified behavior.
- **Teaching PASS:** purpose before code, mechanisms inside code, and evidence-supported meaning afterwards; source-faithful explanations, proportionate complexity, and useful transfer. Headings, emoji presence, successful execution, and lengthy prose are insufficient.
- **Learning PASS:** Data independently explains the reasoning, assumptions, limits, and transfer to another example; otherwise **UNASSESSED**.

A source review does not reproduce someone else's runtime checks; document what each establishes. Syntax checks, saved notebook outputs, and Git transport are not substitutes for analytical execution. Do not impose routine exams; prepared answers and course completion do not establish independent understanding.

For educational revisions, preserve sound existing work and repair actual gaps rather than rewriting to demonstrate compliance. Durable rules stay in `AGENTS.md`, teaching detail here, requirements and status in the map, and observations beside the `.py` analysis. Do not create extra question logs, answer keys, templates, or verification systems merely to enforce these instructions.

End a bounded task with actual evidence, limitations, and the next proposed question or decision. STOP at its boundary.
