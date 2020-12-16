#  2020-07-17 07:35 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/graph_const.R


#' Visualize a graph from its adjacency matrix
#'
#' @description
#' Visualizes a graph from its adjacency matrix.
#'
#' @param adj An adjacency matrix (a matrix with rows and columns representing nodes and arguments representing an edge or edge weight between two specific nodes).
#' @param ... Any additional arguments.
#'
#' @author Elyas Heidari
#'
#' @section Additional arguments:
#' \describe{
#' \item{directed}{A logical indicating if the graph is directed or not.}
#' \item{community}{A logical indicating if the node communities should be found and the nodes should be colored by them.}
#' \item{plot}{A logical showing if the graph should be visualized statically (with qgraph).}
#' \item{vis_net}{A logical showing if the graph should be visualized interactively (with visNetwork).}
#' \item{betweenness}{While visualizing interactively a logical showing if the node sizes should be proportional to their betweenness centrality.}
#' \item{log}{A logical indicating whether to log the computation times.}
#' \item{verbose}{A logical indicating whether to print out the computation steps.}
#' }
#'
#' @return A list including the desired features of a graph.
#' @export
#'
#'
#'
#' @importFrom  visNetwork toVisNetworkData visOptions visEdges
#' @importFrom  igraph cluster_louvain betweenness membership V intersection graph_from_adjacency_matrix
#' @importFrom  tidyverse %>%

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
