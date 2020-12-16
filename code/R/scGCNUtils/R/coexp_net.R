# 2020-07-21 09:34 
# elihei  [<eheidari@student.ethz.ch>]
#/Volumes/Projects/scGCN/code/R/scGCNUtils/R/coexp_net.R

#' Coexpression network based on the mutual information
#'
#' @description
#' Computes mutual information of pairs of features (e.g., genes), to construct an interaction network based on high mutual informations.
#'
#' @param mtx An expression or count matrix with rows as features (e.g., genes) and columns as observations (e.g., cells).
#' @param ... Any additional arguments.
#'
#' @details Running time might be long and using multiple cores is recommended.
#'
#'
#' @author Elyas Heidari
#'
#' @section Additional arguments:
#' \describe{
#' \item{mc.cores}{Number of cores to be used for parallelization.}
#' \item{num_edges}{A numeric indicating the final number of edges (top n edges to be picked).}
#' \item{min_mi}{Minimum of the mutual information tolerable (if `num_edges = NULL`).}
#' \item{min_rel_prop}{Minimum of the mutual information tolerable relative to the maximum among all edges(if `num_edges = NULL`).}
#' \item{log}{A logical indicating whether to log the computation times.}
#' \item{verbose}{A logical indicating whether to print out the computation steps.}
#' }
#'
#' @return The adjacency matrix as a sparse matrix of type `dgCMatrix`.
#' @export
#'
#'
#'
#' @importFrom  Matrix dgCMatrix
#' @importFrom  tictoc toc tic.log
#' @importFrom  tidyverse %>% map


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