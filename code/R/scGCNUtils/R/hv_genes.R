# 2020-04-16 09:13
# elihei [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/hv_genes.R 

#' graph visualization and community detection
#'
#'
#' @description
#' Converts the graph to an igraph object, finds communities and plots it using qgraph package.
#'
#' @param df An arbitrary graph object in R.
#' @param n Set it TRUE when the graph is directed. (default = FALSE)

#' @author  Elyas Heidari
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
#' @importFrom  genefilter rowSds
hv_genes <- function(df, n = 100) {
    sds        = rowSds(df, na.rm = T)
    means      = rowMeans(df, na.rm = T)
    sds        = unlist(sds)
    means      = unlist(means)
    lm         = lm(sds~means)
    res        = residuals(lm)
    names(res) = 1:length(res)
    down       = tail(order(res), n)
    down
}

# 09:14