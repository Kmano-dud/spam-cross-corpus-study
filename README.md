spam-cross-corpus-study

This repository contains the datasets, code, supplementary materials, and experimental outputs used to reproduce the results reported in the manuscript.

================================================================================
DATASETS
================================================================================

The Datasets directory contains the exact datasets used during the experimental evaluation.

dataset_a_cross_train_27000_ids.pkl
- Stratified 27,000-record subset derived from Dataset A's original 30,000-record sample.
- Used as the source corpus during cross-corpus training.

dataset_b_cross_train_27000_ids.pkl
- Stratified 27,000-record subset derived from Dataset B's original 30,000-record sample.
- Used as the source corpus during cross-corpus training.

dataset_a_to_b_target_9956_ids.pkl
- Stratified 9,956-record subset of Dataset A.
- Used as the target corpus in the Dataset B → Dataset A cross-corpus experiment.

dataset_b_to_a_target_9956_ids.pkl
- Stratified 9,956-record subset of Dataset B.
- Used as the target corpus in the Dataset A → Dataset B cross-corpus experiment.

dataset_a_within_30000_ids.pkl
- Stratified 30,000-record sample used for the within-corpus experiments on Dataset A.

dataset_b_within_30000_ids.pkl
- Stratified 30,000-record sample used for the within-corpus experiments on Dataset B.

datasetC_9956_ids.pkl
- Processed Dataset C corpus containing 9,956 emails after preprocessing and quality-control filtering of the original 10,000-record LLM-generated dataset.
- Used as the target corpus in the Dataset A → Dataset C and Dataset B → Dataset C experiments.

================================================================================
SUPPLEMENTARY
================================================================================

cross_corpus_bootstrap_CIs
- Contains the 95% bootstrap confidence intervals (10,000 resamples) for:
  • F1-score
  • Matthews Correlation Coefficient (MCC)
  • Relative Retention Score (RRS)
  • Absolute performance degradation
- Includes all 60 cross-corpus experiments.

pairwise_corrected
- Contains the pairwise Nadeau-Bengio corrected resampled t-tests.

reconstructed_confusion_metrics
- Contains the reconstructed confusion matrices used during the uncertainty analysis.

manifest.csv
- Machine-readable manifest containing:
  • filenames
  • SHA256 hashes
  • record counts
  • class totals
  • source-target experiment directions

overlap_manifest.csv
- Machine-readable manifest containing the reported overlap counts corresponding to the final deposited dataset identifiers and preprocessing stage.

Within Corpus Folds Dataset A
Within Corpus Folds Dataset B

These directories contain all fold-level evaluation results for every within-corpus experiment and feature configuration.

They additionally include:

- Fold scores from the duplicate-removal sensitivity analysis for the Text and Text+Metadata configurations.

- Fold scores from the exploratory Logistic Regression sensitivity analysis on Dataset A using the Text+Metadata configuration with standardized metadata features (StandardScaler).

================================================================================
CROSS CORPUS UNROUNDED RESULTS
================================================================================

This directory contains the original full-precision cross-corpus evaluation metrics used throughout the study.

Results are organized according to the following source-target transfer directions:

A-B Target
- Models trained on Dataset A and evaluated on Dataset B.

B-A Target
- Models trained on Dataset B and evaluated on Dataset A.

A-C Target
- Models trained on Dataset A and evaluated on Dataset C (LLM-generated corpus).

B-C Target
- Models trained on Dataset B and evaluated on Dataset C (LLM-generated corpus).

Each subdirectory contains three result files corresponding to the evaluated feature configurations.

results_(source)_(target)_textonly.csv
- Cross-corpus classification results obtained using the Text Only representation.
- Features consist exclusively of TF-IDF textual features.

results_(source)_(target)_metadata.csv
- Cross-corpus classification results obtained using the Metadata Only representation.

results_(source)_(target)_text+metadata.csv
- Cross-corpus classification results obtained using the combined Text + Metadata representation.
- TF-IDF textual features are augmented with the engineered metadata features.

The CSV files preserve the original full-precision evaluation metrics generated during experimentation and were subsequently used for confusion-matrix reconstruction and uncertainty analysis.

================================================================================
CODE
================================================================================

cross_corpus_pipeline.ipynb
- Cross-corpus evaluation pipeline.

Email_Preprocessing_Pipeline.ipynb
- Raw email preprocessing pipeline used to generate the processed datasets.

Within_Corpus_Pipeline.ipynb
- Within-corpus evaluation pipeline.

================================================================================
IMPORTANT REPRODUCIBILITY NOTES
================================================================================

Evaluation Workflow

The reported classification experiments are executed using the deposited preprocessed datasets.

The preprocessing pipeline applied to the original raw datasets is documented separately in Email_Preprocessing_Pipeline.ipynb and is not executed automatically as part of the evaluation workflow.

Reproducing the preprocessing stage therefore requires independent access to the original raw datasets, whereas all reported classification results can be reproduced directly from the deposited processed datasets.

Prediction Storage

The prediction-saving functionality (SAVE_PREDICTIONS) available in cross_corpus_pipeline.ipynb was introduced after the original experiments had been completed and was therefore NOT used to generate the reported manuscript results.

Consequently, prediction-level outputs are unavailable for the original cross-corpus experiments.

Confidence Intervals and Confusion Matrices

The reported confidence intervals were not derived from the rounded values appearing in the manuscript tables.

Instead, the confusion matrices were recovered from the original full-precision accuracy, precision, and recall values preserved in the experiment result files together with the known target-class counts of the deposited evaluation datasets.

The reported 95% confidence intervals should therefore be interpreted as approximate conditional bootstrap intervals based on the recovered confusion matrices.

These intervals quantify target-sample uncertainty for fixed trained models and do not incorporate training-sample variability, model-training variability, or paired uncertainty between classifiers.