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

adj2graph <- function(adj, ...){
    # TODO: message
    # set params
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    params %<>% set_params('adj2graph', 2)
    # the returned object
    to.ret = list(adj=adj)
    # create the igraph object
    ig = graph_from_adjacency_matrix(adj, mode=ifelse(params$directed, 'directed', 'undirected'))
    to.ret$igraph  = ig 
    # extract graph communities
    if(params$community){
        # TODO: message
        if (!is.null(params$groups)) { # use user's communities
            V(ig)$community = params$groups
        }else{  # compute communities with louvain
            fc = cluster_louvain(as.undirected(ig))
            V(ig)$community = membership(fc)
        }
    }else{
        V(ig)$community = 1
    }
    com             = V(ig)$community
    names(com)      = names(V(ig))
    to.ret$community = com
    # visualize graph with qgraph
    if(params$plot){
        gg = qgraph::qgraph(
            as_adjacency_matrix(ig),
            bidirectional = !params$directed,
            groups        = as.factor(V(ig)$community),
            layout        = 'layout.kamada.kawai',
            palette       = "pastel",
            label.norm    = "OOOOOOO",
            labels        = names(V(ig)),
            vsize         = max(1, 0.5 + 320 / (length(V(ig)) + 50)),
            legend        = FALSE,
            edge.color    = '#b9c2d9',
            esize         = .2,
            label.prop    = 1.3,
            borders       = FALSE,
            curve         = .01,
            curveAll      = TRUE,
            title         = params$title
        )
        V(ig)$color = gg$graphAttributes$Nodes$color
        to.ret$graph_vis = gg
    }
    # visualize graph with visNetwork
    if(params$vis_net){
        if(params$betweenness){ # compute betweenness values 
            not_sort_bt = betweenness(ig, V(ig), directed=params$directed)
            bt          = sort(not_sort_bt, decreasing=T)
            nodes_title = paste0("<p> Degree = ", degree(ig), "</br> Betweenness = ", not_sort_bt, "</p>")
        }else{
            bt = NULL
            nodes_title = paste0("<p> Degree = ", degree(ig), "</p>")
        }   
        data = toVisNetworkData(ig) 
        data$nodes$group = V(ig)$community
        node_value       = degree(ig)
        nodes            = data$nodes
        nodes[, "title"] = nodes_title
        nodes[, "value"] = node_value
        vs               = visNetwork(nodes=nodes, edges=data$edges)  %>%
                           visOptions(highlightNearest=list(enabled=TRUE, degree=1, hover=TRUE))
        if(params$directed)
            vs = vs %>% visEdges(arrows="to", smooth=F)
        to.ret$vis_net = vs
    }
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    to.ret
}
