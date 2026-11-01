# Disaster Tweets: An Explainable NLP Baseline

**How well does the TF-IDF + Logistic Regression baseline classify the held-out tweets, and where does it fail?**

This MasterSchool DA112 reconstruction follows tweet data from quality checks to a measured classification baseline.
On a fixed 1,497-tweet holdout, positive-class F1 is **0.7308**; the model misses **208 of 637 recorded positives**.
The central lesson is that preparation defines what a score describes, while error analysis reveals what the score hides.

The [school Python analysis](disaster_tweets_classification.py) contains the executable baseline, explanations, source references, and selected error examples.
It retrospectively documents the analysis without claiming a new school submission, presentation recording, or independent assessment of learner understanding.

| Authoritative source | Role |
| --- | --- |
| [disaster_tweets_classification.py](disaster_tweets_classification.py) | School baseline and original holdout; supplementary Plotly EDA, NLTK teaching, six training-only CV configurations and one DistilBERT sentiment example |
| [disaster_tweets_huggingface.py](disaster_tweets_huggingface.py) | **Beyond the Baseline:** frozen MiniLM embeddings with Logistic Regression and LinearSVC, training-only CV and a retrospective holdout comparison |

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
Later training-only CV and Transformer examples extend the learning; they do not change this original baseline or make its holdout fresh again.

**⚖️ Ethical Takeaway:** Disclose exclusions and missed positives so readers do not mistake this educational baseline for reliable detection of every disaster report.

## Beyond the Baseline

The main file compares Logistic Regression, LinearSVC and MultinomialNB at two n-gram settings using the same ten training folds.
Every fold fits its own complete TF-IDF Pipeline. Its separate DistilBERT example predicts movie-review sentiment, not disaster labels.

The MiniLM file keeps the encoder frozen and fits two classical classifiers to its 384-feature vectors.
Its explanations and Plotly comparisons distinguish current measurements from historical TF-IDF references, including the false-positive / false-negative trade-off.
External pretraining and retrospective reuse of the inspected holdout limit the claims; no encoder fine-tuning or operational deployment is demonstrated.

## Run both analyses in the existing environment

### Interpreter and packages

Preserve the verified project-local **Python 3.11.11 `.venv`**; no rebuild is needed.
[requirements.txt](requirements.txt) records the installed analysis, model-loading and Interactive Window packages, with selected numerical/tokenizer compatibility pins.
It is not a full transitive lockfile, and a clean-environment installation has not been tested.

The scripts use pandas, NumPy, scikit-learn, Plotly, NLTK, PyTorch, Transformers and huggingface-hub.
SciPy supports the numerical work; safetensors and tokenizers support model loading.
VS Code's Python/Jupyter extensions use **ipykernel 7.4.0**; **nbformat 5.11.1** supports inline Plotly display.
The MiniLM repository name does not require installing the `sentence-transformers` Python package: this study uses Transformers directly.

### Project-local resources

NLTK data and pretrained weights are separate from Python packages. The ignored `.cache/` directory contains copies of existing approved resources.
Both scripts require their relevant files here and stop with a setup error if they are missing; they do not download replacements.

```text
.cache/
  nltk_data/
    tokenizers/punkt_tab/english/
    taggers/averaged_perceptron_tagger_eng/
    corpora/wordnet.zip
    corpora/stopwords/english
  huggingface/hub/
    models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/<revision>/
    models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<revision>/
```

| Model | Immutable revision | Required snapshot files |
| --- | --- | --- |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `714eb0fa89d2f80546fda750413ed43d93601a13` | `config.json`, `model.safetensors`, `tokenizer_config.json`, `vocab.txt` |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | `1_Pooling/config.json`, `config.json`, `model.safetensors`, `modules.json`, `sentence_bert_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` |

The snapshot files are independent copies, without links back to the global cache; the original global resources remain untouched.
NLTK lookup is restricted to the project directory, and both models use explicit cache paths with `local_files_only=True`.
Missing setup requires a separately approved copy or provisioning step; installing requirements alone does not supply these resources.
Neither `.venv` nor `.cache` is included in Git, so a fresh checkout is not ready to run without that setup.

### Execution

From the repository root, run each complete script separately with the existing interpreter (`-B` avoids writing Python bytecode):

```bash
.venv/bin/python -B disaster_tweets_classification.py
.venv/bin/python -B disaster_tweets_huggingface.py
```

The main script includes all 60 classical CV fits and the sentiment inference; MiniLM includes encoding, 20 CV fits and two final classifier fits.
Both use the unchanged `DATA/train.csv`; the second script reconstructs the same approved preparation and split independently.

For VS Code, open this repository folder and select its `.venv/bin/python` kernel for each Python file's Interactive Window.
Restart the kernel before running cells from top to bottom, especially after earlier sessions loaded global NLTK corpora or model objects.
Interactive sessions use their current folder to locate data and caches; terminal scripts resolve these paths beside the source file.
The `# %%` cells execute Python; `# %% [markdown]` cells supply explanations and Mermaid diagrams.
Plotly display depends on the active renderer; successful terminal execution does not verify Interactive Window graphics or website export.

### Verification and limits

Local verification on **2026-11-01** executed both complete files through `runpy` in the existing `.venv`, with Hub offline mode and network connections blocked.
All imports, the 13 requirement pins and `pip check` passed; both resource locations and missing/incomplete-cache errors were checked.
Preparation, split membership/order and all ten folds matched across scripts. The baseline, six classical CV summaries and MiniLM holdout results matched their documented precision.
The 20 recorded Naive Bayes and 20 MiniLM fold scores also matched; the other 40 historical fold values are not recorded individually in the source.
DistilBERT again returned `POSITIVE` with score `0.999887`; this remains a sentiment demonstration.
All 14 Plotly figures were constructed and serialized using the terminal JSON renderer.

A fresh VS Code Interactive Window verified the actual interpreter/folder, current resource-loading cells, both model inferences and the original class-balance Plotly chart.
This was a focused UI check, not a second full analysis run in the GUI or a website-export check.
The kernel emitted an optional `IProgress not found` warning; model loading, inference and Plotly display still succeeded without installing widgets.
The existing environment and global resources were preserved; no clean-install reproduction or new independent holdout evaluation was performed.

## Sources and provenance

MasterSchool's Sprint 3 assignment defines the task; its Sprint 2 guided NLP example supplies an implementation reference.
DataCamp supports the teaching adaptations below; the project-specific explanations and conclusions are original.
The [historical learner notebook](Notebooks/NLP_Classification_Project_Masterschool.ipynb) is preserved separately from the authoritative Python analysis.

- [Kaggle dataset description](https://www.kaggle.com/competitions/nlp-getting-started/data).
- [DataCamp: TF-IDF vectors for TED talks](https://campus.datacamp.com/courses/feature-engineering-for-nlp-in-python/tf-idf-and-similarity-scores?ex=3).
- [DataCamp: Pipeline for song genre prediction](https://campus.datacamp.com/courses/supervised-learning-with-scikit-learn/preprocessing-and-pipelines-4?ex=6).
- [scikit-learn 1.5: Pipeline API](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html).
- [NLTK data locations](https://www.nltk.org/data.html) and [resource lookup](https://www.nltk.org/api/nltk.data.html).
- [Hugging Face cache layout](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) and [local snapshot lookup](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download).

The Python analysis links additional inspected lessons and distinguishes teaching references from implementation adaptations.
