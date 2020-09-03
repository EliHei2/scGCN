# 2020-04-15 10:05
# elihei [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/ggm_net.R 

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

ggm_net <- function(mtx, ...) {
    # TODO: message
    # set args
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    params %<>% set_params('ggm_net', 2)
    genes = colnames(mtx)
    data  = mtx %>% as.data.table
    # design and stats
    model    = gRim::cmod(~ . ^ ., data=data)
    S        = stats::cov.wt(data, method='ML')$cov
    g_models = list() # keeps the models
    if('aic' %in% tolower(params$method)) # aic
        g_models$aic = gRbase::stepwise(model) %>% 
            as('igraph') %>%
            as_adj(type='both') 
    if('bic' %in% tolower(params$method)) # bic
        g_models$bic = gRbase::stepwise(model, k=log(nrow(data))) %>% 
            as('igraph') %>%
            as_adj(type='both') 
    if('test' %in% tolower(params$method)) # test
        g_models$test = gRbase::stepwise(model, criterion='test') %>% 
            as('igraph') %>%
            as_adj(type='both') 
    if('threshold' %in% tolower(params$method)) { # threshold on PCs
        PC                 = gRbase::cov2pcor(S)
        Z                  = abs(PC)
        Z[Z < params$threshold]   = 0
        diag(Z)            = 0
        Z[Z > 0]           = 1
        g.thresh           = as(Z, 'graphNEL')
        g_models$thresh    = gRim::cmod(g.thresh, data=data)  %>% 
            as('igraph') %>%
            as_adj(type='both') 
    }
    if('sin' %in% tolower(params$method)) { # significance tests
        psin             = sinUG(S, n=nrow(data))
        g_models$gsin    = method::as(getgraph(psin, params$significance), 'graphNEL')  %>% 
            as('igraph') %>%
            as_adj(type='both') 
    }
    if('glasso' %in% tolower(params$method)) {
        glasso_rho <- function(rho_val){ # a function to compute glasso for a specefic rho
            C         = stats::cov2cor(S)
            res.lasso = glasso::glasso(C, rho=rho_val)
            AM        = abs(res.lasso$wi) > params$threshold
            diag(AM)  = F
            1 * AM
        }
        if(!is.null(params$stability)){ # stability analysis 
            params$rho_vals = rexp(params$stability, 1/params$rho_vals)
            if(is.null(params$stability_cut)) # set 0.9 cut-off for significance of params$stability test
                params$stability_cut = 0.9
        }
        if(length(params$rho_vals) < params$mc.cores){
            warning('#processes < mc.cores, sets mc.cores = #processes')
            params$mc.cores = length(params$rho_vals)
        }
        # aggregate glasso graphs
        if(params$mc.cores == 1)
            sum_adjs = params$rho_vals %>% lapply(glasso_rho) %>% purrr::reduce(`+`) 
        else
            sum_adjs = params$rho_vals %>% mclapply(glasso_rho, mc.cores=params$mc.cores) %>% purrr::reduce(`+`) 
        sum_adjs = sum_adjs / max(sum_adjs)
        # binarize adj matrix based on stability cut_off
        sum_adjs = ifelse(sum_adjs > params$stability_cut, 1, 0)
        g_models$glasso = sum_adjs
    }
    # aggeregate all models
    adj = g_models %>% purrr::reduce(`*`)
    dimnames(adj) = list(genes, genes)
    # symmetrize and return
    adj = (adj + t(adj))
    adj = adj %>% `>`(0) %>% `*`(1) %>% as('dgCMatrix')
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    adj
}


# 11:22


