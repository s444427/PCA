# PCA
Project realising implementation of Primal Component Analysis for the dataset of iris

For given data with unlimited (yet countable) number of features return data limited to only those d- features that provide the highest variance of it.
In other words delete data that doesn't provide value (diversity).

Purpouse of algorythm: preprocessing large datasets

########################
HOW TO USE
########################
1. In main: change iris_data to any matrix where data is stacked in columns (samples are in rows)
2. In main: Set number of features d that you want to have in a final set (not more then columns!!!) in both pca calls

result:
Y - processed data
V - list of eigenvectors of processed data (matrix)
Lambda - list of eigenvalues of processed data (vector)
