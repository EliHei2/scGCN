# scGCN

**s**ingle-**c**ell **G**raph **C**onvolutional **N**etwork (scGCN) is a single cell annotation tool. scGCN takes gene-gene interactions into account by constructing an interaction network (as a proxy to the true gene regulatory network) on a subset of genes and uses a graph deep learning approach to classify single cells. The pipeline is an integrated data flow, with tunable options to achieve the most accurate cell annotation. Moreover, we provide multiple visualization tools including t-SNE, UMAP1 as well as a few statistical measures at different stages of the model to facilitate interpretation.

Look at `analysis` for input prepration and post-analyais in R, also find the deep learning models in `code/python`. The functions used in R are integrated in package `scGCNUtils`. Find it in `code/R`.

Elyas (eheidari[at]student[dot]ethz[dot]ch)

Cheers
