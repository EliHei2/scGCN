# 2020-07-23 17:59 
# elihei  [<eheidari@student.ethz.ch>]
#/Volumes/Projects/scGCN/code/R/scGCNUtils/R/compare_networks.R


#' Compare train and test interaction networks assigned to a SingleCellExperiment.
#'
#' @description
#' Compares train and test interaction networks to check their coherency.
#'
#' @param sce A SingleCellExperiment with reconstructed interaction networks for train and test data.
#' @param ... Any additional arguments.
#'
#'
#' @author Elyas Heidari
#'
#' @section Additional arguments:
#' \describe{
#' \item{log}{A logical indicating whether to log the computation times.}
#' \item{verbose}{A logical indicating whether to print out the computation steps.}
#' }
#'
#' @return A SingleCellExperiment object with visualized comparison of graphs (`$comparegraphs$vis`) and a coherency score (`$comparegraphs$com_score`).
#' @export
#'
#'
#'
#' @importFrom  SingleCellExperiment SingleCellExperiment
#' @importFrom  tidyverse map %>%
#' @importFrom  tictoc toc tic.log


compare_networks <- function(sce, ...){
    # TODO: message
    # set params
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    params %<>% set_params('compare_networks', 1)
    adj_train  = metadata(sce)$train$graph$adj
    adj_test   = metadata(sce)$test$graph$adj
    adj_all    = rbind(adj_train, adj_test)
    colors     = c(rep(metadata(sce)$train$graph$community, 2))
    shapes     = c(rep('train', dim(adj_train)[2]), rep('test', dim(adj_test)[2])) 
    redim_obj  = redim_mtx(as.matrix(adj_all), ...=append(params$redim_mtx, 
                            list(colors=colors, shapes=shapes)))
    comp_score = (adj_train * adj_test) %>% sum %>% `/`(min(sum(adj_train), sum(adj_test)))
    print(comp_score)
    metadata(sce)$compare_graphs$vis = redim_obj
    metadata(sce)$compare_graphs$comp_score = comp_score
    metadata(sce)$params$compare_networks = params
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    sce
}