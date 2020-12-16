#  2020-07-17 07:35 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/construct_graph.R


#' Construct and visualize the interaction networks for train and test.
#'
#' @description
#' Constructs a the interaction networks for train and test separately based on Gaussian graphical models, mutual information, or prior knowledge.
#'
#' @param sce A SingleCellExperiment object in which train and test subsets are tagged in the `colData`.
#' @param ... Any additional arguments.
#'
#'
#' @author Elyas Heidari
#'
#' @section Additional arguments:
#' \describe{
#' \item{method}{The method to be used for the network construction (possible values = `c('ggm', 'tf', 'coexp')`).}
#' \item{ggm}{Only used if `'ggm'` is specified as a method. Parameters to be passed to `ggm_net` function.}
#' \item{tf}{Only used if `'tf'` is specified as a method. Parameters to be passed to `tf_net` function.}
#' \item{coexp}{Only used if `'coexp'` is specified as a method. Parameters to be passed to `coexp_net` function.}
#' \item{agg_fun}{If multiple methods are specified, how to aggregate them? if `'intersect'` intersects the edge sets of all methods, otherwise, takes a union.}
#' \item{vis}{Visualization parameters to be passed to `adj2graph` function.}
#' \item{log}{A logical indicating whether to log the computation times.}
#' \item{verbose}{A logical indicating whether to print out the computation steps.}
#' }
#'
#' @return A SingleCellExperiment object with interaction networks for train (`metadata(.)$train$graph`) and test (`metadata(.)$test$graph`).
#' @export
#'
#'
#' @importFrom  SingleCellExperiment SingleCellExperiment
#' @importFrom  tidyverse map %>%
#' @importFrom  tictoc toc tic.log

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
