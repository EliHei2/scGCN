#  2020-07-20 09:10 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/plot_sce.R

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


plot_sce <- function(sce, method = 'PCA', label = c('cell_type', 'batch')){
    col_data = colData(sce) %>% as.data.table %>% .[, .SD, .SDcols=c('tag', label)]
    redim_dt  = reducedDim(sce, method) %>% 
        as.data.table %>%
        setnames(c('dim1', 'dim2')) %>%
        cbind(col_data)
    plot_label <- function(label){
        if(label == 'cell_type')
            palette = 'Dark2'
        else
            palette = 1
        redim_dt %>% 
            ggplot +
            aes_string(x='dim1', y='dim2', col=label)+
            geom_point() +
            theme_bw() +
            theme(axis.text.x=element_blank(), axis.text.y=element_blank()) +
            facet_grid(tag ~ .) +
            scale_color_brewer(palette=palette) +
            labs(x=paste0(method, '_1'), y=paste0(method, '_2'))
    }
    label %>% map(~plot_label(.x))
}