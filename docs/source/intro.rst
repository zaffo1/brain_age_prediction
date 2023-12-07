Introduction
============



The aim of this project is to create a regressor able to predict the age of healthy subjects,
based on data from structural and functional MRI scans.
The idea is to train a machine learning model on data from typically developing
control subjects (TD), and then apply it to predict the age of subjects diagnosed with autism
spectrum disorder (ASD).
By comparing chronological age with the predicted age we can estimate
a PAD (Predicted Age Difference) score. We would like to study this score,
investigating whether it can be an informative marker for the ASD.

In particular, we build three different models: a structural model, a functional model, and
a joint model that takes in input both structural and functional features in a multi-modal fashion.
We would like to compare the results of these three models, in order to understand the impact of different data
on our task.


