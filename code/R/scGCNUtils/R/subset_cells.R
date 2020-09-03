#  2020-07-19 20:39 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/select_celltypes.R


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


subset_cells <- function(sce, ...) {
    # set params
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    params %<>% set_params('subset_cells', 1)
    # colData to as.data.table
    col_data = colData(sce) %>%
        as.data.table(keep.rownames=T) %>%
        setnames('rn', 'id') 
    # select cell types
    if(params$cell_types == 'all')
        params$cell_types = unique(col_data$cell_type)
    col_data_list = unique(col_data$tag) %>%
        map(~col_data[tag==.x])
    names(col_data_list) = unique(col_data$tag)
    # subset to shared cell types
    params$cell_types = col_data_list %>%
        map(~.x$cell_type) %>%
        purrr::reduce(intersect) %>%
        intersect(params$cell_types)
    col_data_list %<>% map(~.x[cell_type %in% params$cell_types])
    names(col_data_list) = unique(col_data$tag)
    stopifnot(length(col_data_list) <= 2)
    # subset based on cut-offs
    type_freqs = col_data_list[['train']] %>% .$cell_type %>% table
    type_props = type_freqs / max(type_freqs)
    params$cell_types = names(which(type_freqs > params$min_num & type_props > params$min_rel_prop))
    col_data_list %<>% map(~.x[cell_type %in% params$cell_types])
    # balance cell type abaundances
    if(params$balanced){
        type_freqs = col_data_list[['train']] %>% .$cell_type %>% table
        min_freq   = min(type_freqs)
        col_data_list[['train']]  %<>% .[, .SD[sample(1:dim(.SD)[1], min_freq)], by = 'cell_type']
    }
    cells = col_data_list %>% map(~.x$id) %>% purrr::reduce(c)
    sce = sce[,cells]
    # compute gene vars for the subset sce
    rowData(sce)$gene_var = modelGeneVar(sce)
    # add metadata fields
    metadata(sce)$params$subset_cells = params
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    sce
}
