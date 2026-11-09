# NLP Classification — Project Learning Context

This document supplements the [shared Master School learning methodology](https://github.com/zvuk1T/zarko-vukovic-homepage/blob/main/docs/MASTER-SCHOOL-HOW-WE-WORK-AND-LEARN.md).

`AGENTS.md` remains the repository's operational authority. `DA112_PROJECT_MAP.md` owns the school requirements, evidence, decisions and roadmap.

## Project Purpose and Source Roles

Master School DA112 Sprint 3 — Disaster Tweets — is the primary school project.

The main analytical sequence is:

**Prepared tweets → TF-IDF → Logistic Regression → fixed holdout → classification errors and limitations.**

`disaster_tweets_classification.py` is the authoritative school analysis and primary learning source.

`disaster_tweets_huggingface.py` is a separate frozen-MiniLM representation experiment. Classical cross-validation, pretrained sentiment inference and frozen-MiniLM comparisons are additional learning, not replacement school requirements or independent final evaluations.

Preserve the historical learner notebook. Guided Sprint 2 and mentor material are references, not interchangeable implementations.

## NLP-Specific Analytical Questions

The analysis must explain:

- What recorded disaster labels represent and what they do not establish.
- How missingness, repeated text and conflicting labels affect preparation.
- Why the fixed train/holdout boundary matters.
- How TF-IDF turns text into numerical features.
- What Logistic Regression learns and how its errors are measured.
- Why precision, recall, F1, support and the confusion matrix answer different questions.
- Why training-only cross-validation is distinct from final holdout evaluation.

The original fixed holdout has 1,497 retained observations. The baseline recorded class-1 F1 of 0.7308 and missed 208 of 637 disaster-labeled examples.

These results describe agreement with retained annotations, not verified real-world disaster-detection performance.

## Source Fidelity and Modeling Constraints

Keep preprocessing consistent between fitting and inference. Fit vocabulary and IDF using training data only. Inside cross-validation, fit the full pipeline independently within each training fold.

Do not mechanically transfer sentiment-cleaning rules to disaster tweets. Review URLs, hashtags, mentions, negation, punctuation and disaster terms when a preprocessing change is authorized.

DistilBERT sentiment inference is not disaster classification. Frozen MiniLM features and their retrospective holdout evaluation must remain clearly separated from the original baseline.

Do not infer causal disaster signals, emergency-response readiness or operational benefit from classification scores.

## Educational Implementation

The accepted source format is VS Code percent-cell Python.

Preserve meaningful, always-visible comments about unfamiliar intermediate tables, features, masks and indexes. The source-owned Explain this code disclosures provide optional deeper reading.

Use actual verified guided and DataCamp examples where relevant. Particularly useful topics include text-feature engineering, supervised classification, Boolean filtering, pandas joins and scikit-learn pipelines.

The project's offline environment depends on prepared local NLTK and Transformer resources. A previously verified execution in the existing environment does not establish reproducibility in a fresh clone.

Keep additional developer verification separate from student-facing explanations unless a check directly helps answer the analytical question.

## Project Story

Present the original Sprint 3 baseline first, followed by its errors and limits. Identify later learning explicitly as **Beyond the Baseline**.

The central Master School document governs shared learning methodology, DIKW, concise explanations, optional transfer questions and learning validation.