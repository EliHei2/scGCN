#  2020-07-17 07:35 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/redim_mtx.R


#' Perform low-dimensional embedding of a matrix
#'
#' @description
#' Performs low-dimensional embedding of a matrix and visualize the result.
#'
#' @param mtx The matrix to be embedded in low dimension.
#' @param ... Any additional arguments.
#'
#'
#' @author Elyas Heidari
#'
#' @section Additional arguments:
#' \describe{
#' \item{method}{A string indicating the dimensionality reduction method (possible values = `c('pca', 'tsne', 'umap')`).}
#' \item{n_neighbors}{Only if `method == 'umap'`, `n_neighbors` passed to `uwot::umap`.}
#' \item{perplexity}{Only if `method == 'tsne'`, `perplexity` passed to `Rtsne::Rtsne`.}
#' \item{colors}{A vector of the same length as the number of columns, indicating point colors in the low-dimensional visualization.}
#' \item{shapes}{A vector of the same length as the number of columns, indicating  point shapes in the low-dimensional visualization.}
#' \item{plot}{A logical indicating if the low-dimensional embedding should be visualized.}
#' \item{plot_args}{Visualization arguments to be passed to `plot_redim_mtx`.}
#' \item{log}{A logical indicating whether to log the computation times.}
#' \item{verbose}{A logical indicating whether to print out the computation steps.}
#' }
#' 
#' @return A list including the low-dimensional transformed dataset and the plot if `plot==TRUE`.
#' @export
#'
#'
#'
#' @importFrom  Rtsne Rtsne
#' @importFrom  uwot umap
#' @importFrom  Matrix dgCMatrix
#' @importFrom  tictoc toc tic.log
#' @importFrom  tidyverse %>% map



redim_mtx <- function(mtx, ...){
    # TODO: message
    # set params
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    params %<>% set_params('redim_mtx', 2)
    # remove null observations
    to.rm  = which(rowSums(mtx) == 0)
    mtx    %<>% .[-to.rm, -to.rm]
    params$colors %<>% .[-to.rm]
    params$shapes %<>% .[-to.rm]
    # create redim dt
    to.ret = list()
    if(params$method == 'umap'){ # uwot::umap
        if(is.null(params$n_neighbors))
            params$n_neighbors = floor(dim(mtx)[1] / 10)
        umap_dt = uwot::umap(mtx, n_neighbors=params$n_neighbors)
        to.ret$redim_dt = data.table(UMAP_1=umap_dt[,1], UMAP_2=umap_dt[,2],
                                     color=params$colors, shape=params$shapes)
    }
    if(params$method == 'tsne'){ # Rtsne
        if(is.null(params$perplexity))
            params$perplexity = floor(dim(mtx)[1] / 4)
        tsne_dt = Rtsne(mtx, perplexity=params$perplexity) %>% .$Y %>% as.data.table
        to.ret$redim_dt = data.table(TSNE_1=tsne_dt[,1], TSNE_2=tsne_dt[,2],
                                     color=params$colors, shape=params$shapes)
    }
    if(params$method == 'pca'){ # pca
        pca_dt = prcomp(data.frame(mtx), scale=TRUE) %>% .$x %>% as.data.table
        to.ret$redim_dt = data.table(PC_1=pca_dt[,1], PC_2=pca_dt[,2],
                                     color=params$colors, shape=params$shapes)
    }
    # visualize
    if(params$plot)
        to.ret$plot = plot_redim_mtx(to.ret$redim_dt, ...=params$plot_args)
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    to.ret
}
