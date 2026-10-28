# Disaster Tweets: An Explainable NLP Baseline

**How well does the TF-IDF + Logistic Regression baseline classify the held-out tweets, and where does it fail?**

This MasterSchool DA112 reconstruction follows tweet data from quality checks to a measured classification baseline.
On a fixed 1,497-tweet holdout, positive-class F1 is **0.7308**; the model misses **208 of 637 recorded positives**.
The central lesson is that preparation defines what a score describes, while error analysis reveals what the score hides.

The [full Python analysis](disaster_tweets_classification.py) contains the executable work, explanations, source references, and selected error examples.
It retrospectively documents the analysis without claiming a new school submission, presentation recording, or independent assessment of learner understanding.

## Data and consequential decisions

The supplied [training CSV](DATA/train.csv) contains **7,613 rows**, with `text` and `target` columns: `1` means disaster, `0` non-disaster.
These are recorded annotations, not independently verified event reports.
The project references [Kaggle's Disaster Tweets competition](https://www.kaggle.com/competitions/nlp-getting-started); this two-column teaching copy's exact transformation history is undocumented.

| Decision | What it changes | Important trade-off |
| --- | --- | --- |
| Lowercase, collapse whitespace, trim text ends | Defines comparable text and the model input | Loses case distinctions; original text and source indices remain available |
| Exclude every row in a normalized group with both labels | Removes 56 rows across 18 conflicting groups | Changes the evaluated population without resolving which annotation is correct |
| Keep the first occurrence of each remaining normalized text | Removes 74 redundant rows | Reproducible selection does not establish annotation quality |
| Stratified 80/20 split, `random_state=42` | Preserves approximate class proportions | Reproducibility and label balance do not establish event independence |

These are project choices adapted to tweets, rather than copied movie-review cleaning rules.
They retain **7,483 rows**: **5,986 training** and **1,497 holdout**, with zero shared original or prepared text strings.
The reported scores describe that holdout after exclusions; they do not cover the removed conflicts or establish performance on new events.

## Why this baseline, and how Pipeline works

TF-IDF turns text into numerical term weights, while Logistic Regression learns an inspectable linear classification rule.
This provides a simple first measurement of the school task, without selecting settings to improve holdout scores.
The unchanged defaults use word unigrams, no stop-word list, and an L2-regularized classifier.

During `fit`, the Pipeline learns vocabulary and IDF from training text, then fits the classifier using those features and training labels.
During `predict`, it reuses the learned transformation before applying the fitted classifier; holdout text does not update either step.
Pipeline keeps this ordering consistent, but cannot establish label quality, event independence, or correct selection of training observations.

## Results and what accuracy hides

Evaluation compares predictions with the recorded labels in the unchanged holdout, using `target=1` as positive.
The baseline was published in [commit b106c98](https://github.com/zvuk1T/NLP_Classification_Project_Masterschool/commit/b106c9833d76942b4fc02ee0109c35c0d4af2edb); the analysis computes the following results.

| Measure | Value |
| --- | ---: |
| F1, class 1 — primary metric | **0.7308** |
| Precision, class 1 | 0.7989 |
| Recall, class 1 | 0.6735 |
| Accuracy, both classes | 0.7889 |

**Confusion matrix — rows: recorded label; columns: predicted label.**

| Recorded label | Predicted 0 | Predicted 1 | Support |
| --- | ---: | ---: | ---: |
| 0 — non-disaster | 752 TN | 108 FP | 860 |
| 1 — disaster | 208 FN | 429 TP | 637 |

The 108 false positives predict disaster for tweets labeled non-disaster; the 208 false negatives miss tweets labeled disaster.
**What do 208 misses among 637 recorded positives reveal that overall accuracy does not?**
Despite matching 78.89% of all labels, the model misses **32.65% of recorded positives**; its class-1 precision exceeds its recall.

## What the mistakes teach us

The analysis inspects three false positives and three false negatives, selected by lowest source index for reproducibility.
Figurative fire language appears among false positives, while fire reports and abbreviated context appear among false negatives.
These observations suggest questions about representation and context; they do not establish causes or representative error frequencies.

Single-token features lose word order and phrase context, while unfamiliar vocabulary contributes no learned feature.
That limitation helps frame hypotheses, but the inspected examples do not measure which features caused individual predictions.
Neither recorded labels nor a small illustrative sample can settle those explanations.

## Transferable lessons and limits

Preparation determines the population being evaluated; excluding conflicting groups does not remove every ambiguous or incorrectly labeled example.
Consistent transformation protects the feature-learning sequence, while trustworthy evaluation also needs appropriate labels and independence between observations.
A single fixed holdout provides evidence for this experiment, without demonstrating emergency-response benefit or deployment readiness.

**Better question:** What independent evidence about event overlap and annotation context would justify extending these findings beyond the retained tweets?
The supplied text-only table does not establish those relationships, so broader claims remain unsupported.
Training-only cross-validation or a small approved extension could be future work; neither was performed for this baseline.

**⚖️ Ethical Takeaway:** Disclose exclusions and missed positives so readers do not mistake this educational baseline for reliable detection of every disaster report.

## Reproduce the analysis

The verified local environment uses **Python 3.11.11**, **pandas 2.2.3**, and **scikit-learn 1.5.2**.
The latter two are the direct runtime packages; installed numerical dependencies include **NumPy 1.26.4** and **SciPy 1.10.1**.
VS Code Interactive Window additionally uses **ipykernel 7.4.0** with the Python and Jupyter extensions.

From the repository root, using the existing macOS project environment:

```bash
.venv/bin/python disaster_tweets_classification.py
```

Alternatively, open the repository in VS Code, open the Python file, and select `.venv/bin/python` for the Interactive Window.
Restart the kernel and run all cells from top to bottom; the required input is `DATA/train.csv`.
The `# %%` cells execute Python, and `# %% [markdown]` cells provide the accompanying explanation.

The existing `requirements.txt` includes packages unused by this baseline and is not a minimal installation recipe.
A clean-environment installation has not been tested; the command above assumes the listed packages are already available in `.venv`.

## Sources and provenance

MasterSchool's Sprint 3 assignment defines the task; its Sprint 2 guided NLP example supplies an implementation reference.
DataCamp supports the teaching adaptations below; the project-specific explanations and conclusions are original.
The [historical learner notebook](Notebooks/NLP_Classification_Project_Masterschool.ipynb) is preserved separately from the authoritative Python analysis.

- [Kaggle dataset description](https://www.kaggle.com/competitions/nlp-getting-started/data).
- [DataCamp: TF-IDF vectors for TED talks](https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/tf-idf-and-similarity-scores?ex=3).
- [DataCamp: Pipeline for song genre prediction](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=6).
- [scikit-learn 1.5: Pipeline API](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html).

The Python analysis links additional inspected lessons and distinguishes teaching references from implementation adaptations.
