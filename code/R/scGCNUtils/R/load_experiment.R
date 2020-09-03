# 2020-07-19 08:29 
# elihei  [<eheidari@student.ethz.ch>]
#/Volumes/Projects/scGCN/code/R/scGCNUtils/R/load_experiment.R

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


load_experiment <- function(tag, ...) {
    # set params
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    # initialization
    exp_dir    = file.path('data_raw', tag)
    assay_tags = c('train', 'test')
    sce_tag    = paste(format(Sys.Date(), '%Y%m%d'), format(Sys.time(), "%X"), sep='_') %>%
        gsub(':', '', .) %>% 
        paste(tag, ., sep='_')
    sce_state  = 1
    log_tag    = NULL
    if(params$log){
        log_tag = sce_tag
        log_dir = sprintf('.logs/%s.log', log_tag)
        log_dir %>% file.create
        params$log = log_tag
    }
    params %<>% set_params('load_experiment', 1)
    # load a single assay
    load_assay <- function(assay_tag){
        # load data
        assay_dir  = file.path(exp_dir, assay_tag) 
        counts_mat = file.path(assay_dir, 'counts.txt')  %>% readMM
        cells      = file.path(assay_dir, 'cells.txt')   %>% fread %>% c %>% .[[1]]
        genes      = file.path(assay_dir, 'genes.txt')   %>% fread %>% c %>% .[[1]]
        colnames(counts_mat) = cells
        rownames(counts_mat) = genes
        # matrix + metadata --> SCE 
        colData    = file.path(assay_dir, 'colData.txt') %>% fread %>% DataFrame
        rownames(colData) = colData$id
        colData    %<>% .[,setdiff(colnames(colData), 'id')]
        stopifnot(dim(colData)[1] == dim(counts_mat)[2])
        colData$tag = assay_tag
        sce         = SingleCellExperiment(
                        assays=list(counts=counts_mat), 
                        colData=DataFrame(colData))
        rownames(sce) %<>% tolower
        # message
        sprintf('--      read %s with %d cells and %d genes', assay_tag, dim(sce)[2], dim(sce)[1]) %>%
            messagef(verbose=params$verbose, log=log_tag)
        sce 
    }
    # load and merge sces
    sce_list = assay_tags %>% future_map(~load_assay(.x))
    genes    = sce_list   %>% map(rownames) %>% purrr::reduce(intersect)
    sce      = sce_list   %>% map(~.x[genes,]) %>% purrr::reduce(cbind)
    # add rowData
    sce      = logNormCounts(sce)
    tfs      = sprintf(file.path('data_raw','prior/homo_genes_%s.txt'), params$spec) %>%
        fread(header=F) %>% .$V1
    rowData(sce)$is_TF    = rownames(sce) %in% tfs
    rowData(sce)$gene_var = modelGeneVar(sce)
    # define metadata fields
    metadata(sce)$input  = list(train=list(), test=list())
    metadata(sce)$output = list(train=list(), test=list())
    metadata(sce)$vis    = list()
    metadata(sce)$tag    = sce_tag
    metadata(sce)$params$load_experiment = params
    # message
    sprintf('--      return sce %s with %d cells and %d genes', sce_tag, dim(sce)[2], dim(sce)[1]) %>%
        messagef(verbose=params$verbose, log=log_tag)
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    sce
}
