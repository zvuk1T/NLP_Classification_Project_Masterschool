# How We Work and Learn — MasterSchool DA112

This practical educational guide operates **under** [AGENTS.md](AGENTS.md), the sole binding operational contract. The [DA112 Project Map](DA112_PROJECT_MAP.md) supplies the original school requirements, verified evidence, project decisions, and delivery sequence. This guide does not authorize code changes, Git transport, additional requirements, or new courses.

**Understanding before automation. WHY before HOW. Question before method. Evidence before conclusion.**

## 1. What we are trying to learn

Our goal is not to make a notebook that simply predicts a label. We want Data to understand **why a particular NLP question is worth asking**, **how relevant text and a sound method yield evidence**, and **what that evidence can and cannot establish**.

The guided Sprint 2 movie-review classifier explains the fundamentals. The self-paced Sprint 3 disaster-tweet project is where we independently apply them. The mentor's examples enrich our understanding only when they apply. Hugging Face is a **small optional follow-up** after the classical model, not a replacement for learning TF-IDF and Logistic Regression.

The notebook should function as both the **real analysis** and a **readable learning resource**. Months later, Data should be able to revisit it and explain the problem, data, choices, outputs, mistakes, and lessons without reconstructing an earlier chat.

## 2. Better Questions — our learning loop

This Data-authored questioning loop from the DataCamp Gym method is the conceptual center of DA112:

```text
BETTER QUESTIONS
       │
       ▼
Šta želimo razumjeti?
       │
       ▼
Koji kontekst i dokazi su važni?
       │
       ▼
Šta tačno moramo istražiti?
       │
       ▼
Kako ćemo to istražiti?
       │
       ▼
Python / Statistics / NLP
       │
       ▼
Šta nam rezultati stvarno govore?
       │
       ▼
Šta smo naučili?
       │
       ▼
Koje bolje pitanje sada možemo postaviti?
```

This is a **revisable reasoning loop**, not an inflexible algorithm. Discovering ambiguous labels, duplicated tweets, unsupported assumptions, or leakage may require changing the question or evaluation design before writing more code.

### Three distinct WHY questions

1. **Why investigate?** What exactly is being classified, and why does that distinction matter in this task?
2. **Why this approach?** Why can the available data, TF-IDF, and the selected classifier provide useful evidence? What might they miss or distort?
3. **Why learn it?** Which generalizable principle will help Data independently solve another analytical or AI problem?

These questions guide the teaching and the agent's internal self-review, **not a questionnaire Data must fill in before every cell**. Interpretation should suggest a better next question, not a forced claim of certainty.

### Frame the task before the solution

Before a new meaningful learning unit, preserve four distinct items:

| Item | Question to answer |
| --- | --- |
| **Problem / Question** | What is the source-backed problem or decision, and why is it important? |
| **Context** | Which data, labels, observations, constraints, and assumptions define the problem? |
| **Instructions** | What did the actual school task ask, and which implementation choices remain ours? |
| **Expected outcome** | What type of evidence or output is needed, *without pretending it has already been observed*? |

Do not reverse-engineer a school requirement from the code that happens to exist. Distinguish source requirements from assistant-authored choices. Do not reveal a solution within the phrasing of a question that is meant to be investigated.

## 3. From question to student-readable notebook

Work in **one coherent analytical idea or learning slice at a time**. A learning unit may span multiple Markdown and Python cells. The six teaching elements below are a **unit-level structure**, not six sections for every import, line of Python, or routine operation.

1. **📋 Source, Question & Instructions:** briefly name the relevant school source, question, essential context, conditions, and expected evidence.
2. **🎯 Goal & Approach — What & Why:** explain why the selected inputs and method are appropriate, including important assumptions and alternatives.
3. **💻 Real Code + Inline Reasoning:** write readable, executable Python in the existing `.ipynb`. Explain non-obvious operations beside the code without narrating trivial syntax.
4. **✅ Result — What Actually Happened:** give an **observed and verified** output, metric, table, or finding; state how it answers the initial question. Label expected or unexecuted results honestly.
5. **🧠 Recap — What We Learned:** state one transferable principle, meaningful limitation, or better next question. Do not repeat the code or result.
6. **🧑‍💼 Recruiter / Transfer Check:** one relevant conceptual question testing independent reasoning at a meaningful milestone, with a concise model answer where useful. It tests understanding; it does **not** establish Data's Learning PASS.

Use the elements naturally and omit inapplicable parts (for example, a Git setup task has no classification result). Do not duplicate a school rubric or create extra work solely to fill headings. A small **🗺️ Mental Model** or **⚠️ Watch Out** belongs near a genuine conceptual dependency or trap, not as decoration.

**Writing style:** Documentation, code, and code comments are in English; our conversation can use Data's preferred language. Prefer 1–3 focused sentences per meaningful explanatory section. Aim for about 10–20 words per sentence and allow up to 23 for clarity; these are readability guidelines, not a reason to omit essential reasoning. One meaningful idea per paragraph or logical code block; no inflated boilerplate.

## 4. Complexity and the DataCamp bridge

Data knows **Beginner** Python. Much **Intermediate** material is familiar, but not all; **Advanced** techniques need additional explanation. These are teaching needs, **not a ban on an analytically necessary method**.

When an unfamiliar Intermediate or Advanced operation is genuinely required:

- Explain **why** it is needed, **what goes in**, **what comes out**, and any material assumptions or trade-offs, near its first use.
- Prefer a verified, specific **official DataCamp** lesson/chapter link when available. If only a course-level link is verifiable, state that. Do not invent a lesson link or require Data to finish another course before proceeding.
- Prefer an equally correct, simpler, direct approach when it is easier to understand. Add a function, class, abstraction, or dependency only when the requirement or a demonstrated correctness/clarity benefit warrants it.

Do not replace an authentic learner attempt with the assistant's cleverer code. Where changes to starter cells are authorized, explain the reason and preserve relevant provenance. Mentor code should stay intact as a reference unless its alteration is specifically requested.

A working notebook is not automatically a learning notebook. But a long explanation cannot rescue unnecessarily complicated code; **simplify the solution before adding more commentary**.

## 5. DA112 methodology as a chain of questions

Use the [Project Map](DA112_PROJECT_MAP.md#5-methodology-contract-for-the-first-reliable-baseline) for exact source facts, open decisions, and verification requirements. The questions below explain *why* the work is ordered as it is.

| Learning question | Evidence and method | Trap to understand |
| --- | --- | --- |
| **What is a disaster-related tweet?** | Inspect labels, real examples, counts, duplicates, and contradictory labels. | Disaster-related keywords and negative sentiment are not equivalent to actual disasters. |
| **Are these labels and examples reliable?** | Audit missing text, repeated strings, class distribution, and ambiguous examples. | Identical tweets with inconsistent labels cannot be fixed by an arbitrary silent rule. |
| **How can we fairly evaluate a classifier?** | Agree a reproducible training/holdout separation before fitting features. | Text repeated across partitions and tuning against the holdout distort evidence. |
| **How should text become numbers?** | Understand TF-IDF, vocabulary, term frequency, and document frequency. | Fitting the vectorizer on all data exposes test-set information. |
| **What does Logistic Regression learn?** | Train an interpretable linear text classifier on training features. | Coefficients and correlations do not prove causal disaster signals. |
| **What constitutes good performance?** | Use real holdout predictions: F1 for `target=1`, precision, recall, accuracy, confusion matrix. | A single metric hides false positives, false negatives, and dataset limitations. |
| **Will the result generalize reliably?** | Optionally cross-validate a full text pipeline on training folds. | Fitting TF-IDF before CV leaks vocabulary/IDF across folds. |
| **Would tuning meaningfully help?** | Optionally vary a small, explainable set of parameters using training folds. | A larger search is not automatically better learning or generalization. |
| **What should we claim?** | Return from results to the question using DIKW and bounded interpretation. | An educational classifier cannot establish improved emergency-response outcomes. |

The standard first baseline is **TF-IDF + Logistic Regression**. A consistent `Pipeline` can bind the vectorizer and model; this differs from Hugging Face's **pretrained Transformer `pipeline`**, which is a different abstraction. Explain this distinction when introducing the latter.

Before preprocessing, consider URLs, hashtags, mentions, punctuation, negation, names, and disaster terms. Do not mechanically copy a movie-review cleaning function. Where results are counterintuitive, inspect examples rather than automatically adding models or cleaning steps.

## 6. DIKW, Data Literacy, and storytelling

Each major analytical claim should be traceable through:

```text
Observed data and source
        ↓
Information (counts, splits, predictions, metrics)
        ↓
Knowledge (evidence-backed interpretation and limits)
        ↓
Wisdom (proportionate next question or decision)
```

**Data:** exactly what was recorded, and how reliable/provenant it is. **Information:** what our valid operations quantified. **Knowledge:** a plausible explanation that fits the evidence and assumptions. **Wisdom:** a measured next action that acknowledges uncertainty.

Keep these categories separate: **fact ≠ interpretation ≠ hypothesis ≠ recommendation**. Explain the model's false positives and false negatives in plain language. If F1 improves, say what was compared and on which data; do not claim lives saved or faster response times.

Good storytelling is the structure of defensible reasoning: **question → data → method → evidence → meaning → limitations → better question**. It begins long before a final presentation. The original school required a Colab notebook and a live-presentation recording; retrospectively producing documentation does not mean that historical recording or submission occurred.

## 7. How Data, Troi, and Spock collaborate

**Data leads decisions and learning. Troi leads source-aware educational design. Spock leads technical realization and verification when assigned.** Both assistants can challenge each other with evidence. Do not require both assistants to participate in every trivial step, and do not create routine approval loops.

For a meaningful collaborative implementation when Data chooses this workflow:

1. **Ground:** The assigned reader identifies the exact source question, current artifact, learner work, and verified evidence. Missing screenshots, dataset files, or runtime capabilities are reported, not invented.
2. **Design:** Data and Troi identify the question, teaching goal, necessary explanation, and essential correctness constraints. Data owns substantive choices.
3. **Technical review:** Spock assesses whether the proposed change can work, identifies compatibility/data-leakage risks, and proposes the simplest sound implementation. An objection must name the exact issue, evidence, and better alternative.
4. **Integrate:** Spock edits only authorized noncanonical files. Preserve Data-approved educational wording and relevant existing code; do not silently change the learning goal because an implementation shortcut seems convenient.
5. **Verify:** Report actual code execution/checks, diffs, outputs, limitations, and any educational gaps. Data evaluates understanding separately.
6. **Transport:** Stage/commit/push/synchronize only under explicit authorization for those operations; successful local editing is not a standing Git permission.

When Data and Troi send Spock a specific text-edit package, identify base repository/revision, exact target and anchor, complete English text, and one of **KEEP / REPLACE / ADD**. **KEEP** means reviewed and approved unchanged; absence of an instruction does **not** mean approved. Spock flags stale anchors or conflicting source evidence, rather than improvising alternative wording. This is an **optional integration technique**, not a required ceremony for every analytical edit.

**Canonical ownership:** Data + Troi maintain `AGENTS.md` and this guide. Spock handles other files when tasked and can propose canonical revisions without silently implementing them. A clear reassignment from Data changes the owner for that particular task.

**Continuous improvement:** Data develops independent reasoning; Troi improves teaching design from Data's feedback and verified results; Spock improves technical choices through evidence and review. Record only material, reusable lessons in the document that owns them. Do not introduce a new rule because of one harmless exception or treat an agent's earlier answer as unquestionable.

## 8. Review and continuity without bureaucracy

Evaluate substantial learning slices on three independent dimensions:

- **Technical PASS:** code/method/results checked with appropriate actual data, plus reproducibility and leakage safeguards.
- **Teaching PASS:** the question is clear before its answer; explanations are accurate, simple, located near relevant code, and interpreted with evidence limits.
- **Learning PASS:** Data explains the question, input, transformation, evidence, interpretation, and relevant limitation **in their own words**. Otherwise: **UNASSESSED**.

Use a brief conversational understanding check only at meaningful milestones, when Data wants to practice or understanding needs testing. Don't turn every cell into an exam. An answer prepared by Troi or Spock is **not** evidence of Data's independent understanding.

Read relevant current sources; do not reconstruct them from chat memory or duplicate whole repositories. Leave durable approved rules in `AGENTS.md`, how-to guidance here, requirements/roadmap in `DA112_PROJECT_MAP.md`, and actual analytical observations beside the real notebook. Avoid extra status files, standalone answer keys, and unnecessary task-management infrastructure.

**End a bounded work slice with verified evidence and the next proposed question.** Do not use a completed task as permission to begin a new one or publish it.
