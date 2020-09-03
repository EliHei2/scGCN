#  2020-07-17 07:35 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/graph_const.R


#' construct and visualize Gaussian Graphical Models.
#'
#' @description
#' Fit a Gaussian Graphical Model to continuous-valued dataset employing a subset of methods from stepwise AIC, stepwise BIC, stepwise significance test, partial correlation thresholding, edgewise significance test, or glasso.
#' Also visualizes the fitted Graphical Model.
#'
#' @param data A normalized dataframe or matrix with no missing data of continuous measurements.
#' @param methods A string or list of strings indicate methods used to construct the model. See the details for more information. (default = 'glasso')
#' @param ... Any additional arguments.
#'
#' @details The function combines the methods to construct the model, that is, the edge set is the intersection of all edge sets each of which is found by a method. The package gRim is used to implement AIC, BIC, and stepwise significance test. The method glasso from the package glasso is used to provide a sparse estimation of the inverse covariance matrix.
#'
#' @references  Højsgaard, S., Edwards, D., & Lauritzen, S. (2012). Graphical Models with R. Springer US. \url{https://doi.org/10.1007/978-1-4614-2299-0}
#' @references  Friedman, J., Hastie, T., & Tibshirani, R. (2007). Sparse inverse covariance estimation with the graphical lasso. Biostatistics, 9(3), 432–441. \url{https://doi.org/10.1093/biostatistics/kxm045}
#' @references  Abreu, G. C. G., Edwards, D., & Labouriau, R. (2010). High-Dimensional Graphical Model Search with thegRapHDRPackage. Journal of Statistical Software, 37(1). \url{https://doi.org/10.18637/jss.v037.i01}
#'
#'
#' @author Elyas Heidari
#'
#' @section Additional arguments:
#' \describe{
#' \item{threshold}{A threshold for partial correlation thresholding method (default = 0.05). To be used only when the method 'threshold' is used.}
#' \item{significance}{A cutoff for edge significance (default = 0.05). To be used only when the method 'significance' is used.}
#' \item{rho}{(Non-negative) regularization parameter for glasso (default = 0.1). To be used only when the method 'glasso' is used.}
#' }
#'
#' @return an igraph object of the graphical model.
#' @export
#'
#' @examples
#' glasso_ggm = ggm(data = iris[1:4])
#'
#'
#' @importFrom  gRbase cov2pcor stepwise
#' @importFrom  purrr map
#' @importFrom  gRim cmod
#' @importFrom  graph graphNEL
#' @importFrom  SIN sinUG getgraph
#' @importFrom  glasso glasso
#' @importFrom  igraph cluster_louvain betweenness membership V intersection
#' @importFrom  stats C cov.wt
#' @importFrom  methods as
#' @importFrom  graph nodes
#' @importFrom  dplyr %>%
#' 



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
