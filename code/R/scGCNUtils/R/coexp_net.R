# 2020-07-21 09:34 
# elihei  [<eheidari@student.ethz.ch>]
#/Volumes/Projects/scGCN/code/R/scGCNUtils/R/coexp_net.R

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


coexp_net <- function(mtx, ...) {
    # TODO: message
    # set params
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    params %<>% set_params('coexp_net', 2)
    genes = colnames(mtx)
    mtx %<>% t # transpose, consistent with ggm_net
    # compute mi values
    partial_mi <- function(vec1, vec2){ # function to compute mi of two binarized vectors
        vec_union = ifelse(vec1 + vec2 > 0 , 1, 0) # marginalize to non-zero positions
        idx = which(vec_union == 1)
        vec1     %<>% .[idx]
        vec2     %<>% .[idx]
        freq_tab  = table(vec1, vec2) / length(idx)
        mi = 0
        for(i in 1:dim(freq_tab)[1])
            for(j in 1:dim(freq_tab)[2]){
                if(freq_tab[i, j] == 0)
                    next 
                mi = mi + freq_tab[i, j] * 
                        log2(freq_tab[i, j] / 
                            (sum(freq_tab[i,]) * sum(freq_tab[,j])))
            }
        mi
    }
    # compute marginalized mi values
    mi_vals = mclapply(1:(dim(mtx)[1]-1), 
            function(i) lapply((i+1):dim(mtx)[1], 
                            function(j) partial_mi(mtx[i,], mtx[, j])),
        mc.cores=params$mc.cores)
    mi_vals = unlist(mi_vals)
    # get binarized adj matrix based on cut-offs
    if(is.null(params$num_edges)){
        params$min_mi = max(params$min_mi, params$min_rel_prop * max(mi_vals))
    }else{
        params$min_mi = sort(mi_vals, decreasing=T)[params$num_edges]
    }
    mi_vals = ifelse(mi_vals >= params$min_mi, 1, 0)
    adj  = matrix(0, nrow=dim(mtx)[1], ncol=dim(mtx)[1])
    adj[lower.tri(adj, diag=FALSE)] = mi_vals
    dimnames(adj) = list(genes, genes)
    # symmetrize and return the adj matrix
    adj = (adj + t(adj)) %>% as('dgCMatrix')
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    adj
}