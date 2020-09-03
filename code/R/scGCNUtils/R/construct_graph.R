#  2020-07-17 07:35 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/construct_graph.R


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

construct_graph <- function(sce, ...){
    # TODO: message
    # set args
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    params %<>% set_params('construct_graph', 1)
    adj_train = list()
    adj_test  = list()
    # get matrices
    mtx_train  = logcounts(sce[, sce$tag == 'train']) %>% t
    mtx_test   = logcounts(sce[, sce$tag == 'test'])  %>% t
    # compute adj matrices
    ## ggm network
    if('ggm' %in% params$method){
        # TODO: message
        adj_train$ggm = ggm_net(mtx=mtx_train, ...=params$ggm)
        adj_test$ggm  = ggm_net(mtx=mtx_test,  ...=params$ggm)
    }
    ## TF reg. network
    if('tf' %in% params$method){
        # TODO: message
        params$tf$tf_subset = rownames(sce)
        adj_test$tf = adj_train$tf = tf_net(...=params$tf)
    }
    ## MI/coexp network
    if('coexp' %in% params$method){
        # TODO: message
        adj_train$coexp = coexp_net(mtx=mtx_train, ...=params$coexp)
        adj_test$coexp  = coexp_net(mtx=mtx_test,  ...=params$coexp)
    }
    ## aggregate adj. matrices
    # TODO: message
    if(params$agg_fun == 'intersect'){
        adj_train %<>% purrr::reduce(`*`)
        adj_test  %<>% purrr::reduce(`*`)
    }else{
        adj_train %<>% purrr::reduce(`+`) %>% `>`(0) %>% `*`(1)
        adj_test  %<>% purrr::reduce(`+`) %>% `>`(0) %>% `*`(1)     
    }
    # convert adj. matrices to usable lists
    metadata(sce)$train$graph = adj_train %>% adj2graph(...=params$vis) 
    metadata(sce)$test$graph  = adj_test  %>% adj2graph(...=params$vis)
    # set metadata
    metadata(sce)$params$construct_graph = params 
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    # return the modified sce
    sce
}
