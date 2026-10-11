# spam-cross-corpus-study

File "Datasets" contains =

-dataset_a_cross_train_27000_ids.pkl - the stratified 27,000 record subsample from Dataset A's initial 30,000 sample for cross-corpus training

-dataset_b_cross_train_27000_ids.pkl - the stratified 27,000 record subsample from Dataset B's initial 30,000 sample for cross-corpus training

-dataset_a_to_b_target_9956_ids.pkl - the 9,956 portion of Dataset A's 27,000 stratified subsample for cross-corpus testing

-dataset_b_to_a_target_9956_ids.pkl the 9,956 portion of Dataset A's 27,000 stratified subsample for cross-corpus testing


-dataset_a_within_30000_ids.pkl - the stratified 30,000 record sample from Dataset A's initial size for within-corpus training

-dataset_b_within_30000_ids.pkl - the stratified 30,000 record sample from Dataset B's initial size for within-corpus training

-datasetC_9956_ids.pkl - the 9,956 records of LLM-generated Dataset C's 10,000 size


File "Supplementary" contains = 

-cross_corpus_bootstrap_CIs - 95% bootstrap CIs (10,000 resamples) for F1, MCC, RRS and absolute
% degradation of all 60 cross-corpus experiments.

-pairwise_corrected - the pairwise Nadeau-Bengio corrected 60
resampled t-tests.

-reconstructed_confusion_metrics - the reconstructed confusion matrices

File "Within Corpus Folds Dataset B and "Within Corpus Folds Dataset B" which contain all fold scores for each dataset and feature configuration. It also included the fold scores for the rerun no duplicate experiments for Text and Text+Metadat configs.

File "Code" contains =

-cross_corpus_pipeline.py - the cross corpus evaluation pipeline
-Email_Preprocessing_Pipeline.ipynb - the raw email preprocessing pipeline
-Within_Corpus_Reorganized.ipynb - the within-corpus evaluation pipeline