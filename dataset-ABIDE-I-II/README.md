# DATASET used for the project
Data are taken from the ABIDE I and II datasets. In particular I had acess to already harmonized data.
These data are "confidential", therefore only limited samples are here reported (we report only data relative to the first 10 of the 1383 subjects considered).

- The file `sample_Harmonized_structural_features.csv` contains structural features of the subjects
- The file `sample_Harmonized_functional_features.csv` contains functional features of the subjects
  - In particular, it contains 5253 functional connectivity features corresponding to a numberical label from '0' to '5252'.
  - The file `Functional-conn.xlsx` serves to understand what those labels mean. Each number correspond to the two ROIs (Regions of Interest) between which the connectivity is computed. A total of N = 103 ROIs were considered, corresponding to $\frac{N(N-1)}{2}$ non-redundant feature values, equal to 5253 indeed.

