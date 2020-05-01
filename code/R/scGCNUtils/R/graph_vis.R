# 2020-04-15 11:49
# elihei [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/graph_vis 

#' graph visualization and community detection
#'
#'
#' @description
#' Converts the graph to an igraph object, finds communities and plots it using qgraph package.
#'
#' @param graph An arbitrary graph object in R.
#' @param directed Set it TRUE when the graph is directed. (default = FALSE)
#' @param community A logical value to show if the node communities should be detected and colored in the returned graph. (default = TRUE)
#' @param betweenness A logical value to show if the node betweenness measurements should be computed and returned from the function. (default = TRUE)
#' @param plot A logical value to show if the graph should be plotted. (default = FALSE)
#' @param ... Any additional arguments described below.
#'
#'
#' @author  Elyas Heidari, Vahid Balazadeh
#'
#' @return If plot = TRUE it plots the non-interactive graph (If plot.community = TRUE plots communities too) also returns a list contains:
#' \item{graph}{an igraph object.}
#' \item{betweenness}{betweenness measurements of each node.}
#' \item{network}{a visNetwork plot of the graph.}
#' \item{communities}{a named vector indicating the community of each node.}
#'
#' @export
#'
#' @examples
#' require(datasets)
#' data("Harman23.cor")
#' gv = graph_vis(Harman23.cor$cov, plot = TRUE, plot.community = TRUE, community.list = c(1, 2))
#'
#' @section Additional arguments:
#' \describe{
#' \item{groups}{A list that indicates which community each node is. The automatic community detection will be ignored when it is set.}
#' \item{plot.community}{Logical indicating if communities should be plotted. Defaults to FALSE.}
#' \item{filename}{Name of the plot file without extension. (qgraph function argument)}
#' \item{filetype}{A character indicates the file type to save the plots in. (qgraph function argument)}
#' \item{community.list}{A list indicates which communities should be plotted. When is not set, will plot all the communities.}
#' }
#'
#' @importFrom  visNetwork toVisNetworkData visNetwork visOptions
#' @importFrom  igraph cluster_louvain betweenness membership V layout_with_fr induced.subgraph as_adjacency_matrix degree as.undirected
#' @importFrom  methods as
#' @importFrom  dplyr %>%
#' @importFrom  qgraph qgraph

graph_vis <- function(graph, directed = F, community = T, betweenness = T, plot = F, title = NULL, ...) {

    arguments      = list(...)
    usr_groups     = arguments$groups
    plot.community = arguments$plot.community
    community.list = arguments$community.list

    if ( is.null(plot.community) )
        plot.community = F

    .plot_community <- function(graph, community_num) {
        idx       = V(graph)$community == community_num
        v         = V(graph)[idx]
        sub_graph = induced.subgraph(graph, v)
        qgraph::qgraph(
            as_adjacency_matrix(sub_graph),
            groups        = as.factor(V(sub_graph)$community),
            layout        = "spring",
            bidirectional = T,
            color         = V(sub_graph)$color,
            label.norm    = "OOOOOOO",
            labels        = names(V(sub_graph)),
            vsize         = max(1, 0.5 + 320 / (length(V(sub_graph)) + 50)),
            filename      = paste(arguments$filename, community_num, sep=""),
            filetype      = arguments$filetype,
            legend        = FALSE,
            esize         = 1,
            label.prop    = 1.3,
            borders       = FALSE
        )

    }

    ig = methods::as(graph, "igraph")

    if ( betweenness ) {
        not_sort_bt = betweenness(ig, V(ig), directed=directed)
        bt          = sort(not_sort_bt, decreasing=T)
    } else {
        bt = NULL
    }

    community_n = 1

    if ( community ) {
        fc              = cluster_louvain(as.undirected(ig))
        V(ig)$community = membership(fc)
        community_n     = length(unique(V(ig)$community))
    }

    if ( !is.null(usr_groups) ) {
        V(ig)$community = usr_groups
        community_n = length(unique(usr_groups))
    }

    if ( betweenness )
        nodes_title = paste0("<p> Degree = ", degree(ig), "</br> Betweenness = ", not_sort_bt, "</p>")
    else
        nodes_title = paste0("<p> Degree = ", degree(ig), "</p>")

    data = toVisNetworkData(ig)
    if ( community )
        data$nodes$group = V(ig)$community

    node_value       = degree(ig)
    nodes            = data$nodes
    nodes[, "title"] = nodes_title
    nodes[, "value"] = node_value
    vs               = visNetwork(nodes=nodes, edges=data$edges)  %>%
                       visOptions(highlightNearest=list(enabled=TRUE, degree=1, hover=TRUE))

    if ( directed )
        vs = vs %>% visEdges(arrows="to", smooth=F)

    if ( plot ) {
        gg = qgraph::qgraph(
            as_adjacency_matrix(ig),
            bidirectional = T,
            groups        = as.factor(V(ig)$community),
            layout        = "spring",
            palette       = "pastel",
            label.norm    = "OOOOOOO",
            labels        = names(V(ig)),
            vsize         = max(1, 0.5 + 320 / (length(V(ig)) + 50)),
            filetype      = arguments$filetype,
            filename      = arguments$filename,
            legend        = FALSE,
            esize         = .2,
            label.prop    = 1.3,
            borders       = FALSE,
            curve         = .01,
            curveAll      = TRUE,
            title         = title
        )
        V(ig)$color = gg$graphAttributes$Nodes$color
    }

    if ( community ){
        if ( plot && plot.community )
            if ( is.null(community.list) )
                community.list = c(1:community_n)
            for ( i in community.list )
                .plot_community(ig, i)
        com        = V(ig)$community
        names(com) = names(V(ig))
        list(graph=ig, betweenness=bt, vis_net=vs, communities=com, q_net=gg)
    }else
        list(graph=ig, betweenness=bt, network=vs, q_net=gg)
}

# 12:32
