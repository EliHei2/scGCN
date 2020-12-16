#  2020-07-20 09:10 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/redim_sce.R

#' A wrapper for `scater`'s dimension reduction functions.
#'
#' @description
#' Implements PCA, t-SNE and UMAP, as a wrapper for `scater`'s dimension reduction functions.
#'
#' @param sce A SingleCellExperiment to perform dimension reduction on. 
#' @param method A string indicating the method to be used for dimension reduction.
#' @param label Which labels to be used (should be column name from the `colData`) for annotating the cells in the visualization.
#' @param plot A logical, whether to plot the low-dimensional embedding.
#' @param ... Any additional arguments.
#'
#'
#' @author Elyas Heidari
#'
#'
#' @return A SingleCellExperiment object with the reduced dimention matrix in `reducedDims(..., name)`.
#' @export
#'
#'
#'
#' @importFrom  scater runPCA runUMAP runTSNE
#' @importFrom  SingleCellExperiment SingleCellExperiment
#' @importFrom  Matrix dgCMatrix
#' @importFrom  tictoc toc tic.log
#' @importFrom  tidyverse %>% map

redim_sce <- function(sce, method = 'umap', label = c('cell_type', 'batch'), plot = FALSE, ...){
    plot_list = list()
    if('pca' %in% method){
        sce = runPCA(sce, ncomponents=2, ...)
        if(plot)
            plot_list = append(plot_list, plot_sce(sce, 'PCA', label))
    }
    if('umap' %in% method){
        sce = runUMAP(sce, ...)
        if(plot)
            plot_list = append(plot_list, plot_sce(sce, 'UMAP', label))
    }
    if('tsne' %in% method){
        sce = runTSNE(sce, ...)
        if(plot)
            plot_list = append(plot_list, plot_sce(sce, 'TSNE', label))
    }
    if(plot){        to.plot = plot_list %>% purrr::reduce(`+`)
        to.plot = to.plot + plot_layout(ncol=length(label), guides='collect')
        metadata(sce)$vis$redim_plot = to.plot
    }
    # log
    toc(log=TRUE, quiet=TRUE)
    messagef(tic.log(format = TRUE)[[1]], verbose=params$verbose, log=params$log)
    sce
}

