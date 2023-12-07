# brain_age_prediction
Repository for the exam project of the "Computing Methods for Experimental Physics and Data Analysis" course

[![Unit tests](https://github.com/zaffo1/brain_age_prediction/actions/workflows/unittests.yml/badge.svg)](https://github.com/zaffo1/brain_age_prediction/actions/workflows/unittests.yml)
[![Documentation Status](https://readthedocs.org/projects/brain-age-prediction/badge/?version=latest)](https://brain-age-prediction.readthedocs.io/en/latest/?badge=latest)

# Synopsis
The aim of this project is to create a regressor able to predict the age of healthy subjects, based on data from structural and functional MRI scans. In particular, the idea is to train a machine learning model on data from typically developing control subjects (TD), and then apply it to predict the age of subjects diagnosed with autism spectrum disorder (ASD).
By comparing chronological age with the predicted age we can estimate a PAD (Predicted Age Difference) score. We would like to study this score, investigating whether it can be an informative marker for the ASD.

In particular, we build three different models: a structural model, a functional model, and a joint model that takes in input both structural and functional features in a multi-modal fashion.
We would like to compare the results of these three models, in order to understand the impact of different data on our task.

A detailed explanation of the performed anlysis can be found [here](https://drive.google.com/drive/folders/1AVXB8rGO54TP7sK_EjqeZvzmXexWMssN?usp=drive_link).

# Documentation
The documentation for this project is hosted on [readthedocs](https://brain-age-prediction.readthedocs.io/en/latest/?badge=latest).

# Installation
Refer to this brief [installation guide](https://brain-age-prediction.readthedocs.io/en/latest/installation.html#installation-guide).

# Note on the Dataset
For this analysis, data are taken from the ABIDE I and II datasets. In particular I had acess to already harmonized data. Since these data are "confidential", only limited samples are here reported in the folder `dataset-ABIDE-I-II`  (we report only data relative to the first 10 of the 1383 subjects considered).
