# 2020-07-23 08:59 
# elihei  [<eheidari@student.ethz.ch>]
#/Volumes/Projects/scGCN/code/R/scGCNUtils/R/set_params.R


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
#' 


set_params <- function(params, name, level){
    if(is.null(params$verbose))
        params$verbose = TRUE
    prefix = ifelse(level == 1, '+       ', '--      ')
    sprintf('%srun %s', prefix, name) %>% messagef(params$verbose, params$log)
    if(level == 1)
        tic.clearlog()
    sprintf('%s%s', prefix, name) %>% tic
    if(name == 'load_experiment'){
        # TODO: message
        if(is.null(params$spec))
            params$spec = 'human'
        return(params)
    }
    if(name == 'subset_cells'){
        # TODO: message
        if(is.null(params$cell_types))
            params$cell_types   = 'all'
        if(is.null(params$min_rel_prop))
            params$min_rel_prop = 0
        if(is.null(params$min_num))
            params$min_num      = 0
        if(is.null(params$balanced))
            params$balanced     = TRUE
        return(params)
    }
    if(name == 'subset_genes'){
        # TODO: message
        if(is.null(params$method))
            params$method = 'hvg'
        if(is.null(params$n))
            params$n = 100
        return(params)
    }
    if(name == 'construct_graph'){
        if(is.null(params$method))
            params$method  = 'ggm'
        if(is.null(params$agg_fun))
            params$agg_fun = 'intersect'
        if(!is.null(params$log))
            params$ggm$log  = params$tf$log = params$coexp$log = params$vis$log = params$log
        return(params)
    }
    if(name == 'ggm_net'){
        if(is.null(params$threshold))
            params$threshold = 0.05
        if(is.null(params$method))
            params$method = 'glasso'
        if(is.null(params$significance))
            params$significance = 0.05
        if(is.null(params$mc.cores))
            params$mc.cores = detectCores()/2
        if(is.null(params$rho))
            params$rho_vals = 0.1
        else
            params$rho_vals = params$rho
        return(params)
    }
    if(name == 'tf_net'){
        if(is.null(params$spec))
            params$spec = 'human'
        return(params)
    }
    if(name == 'coexp_net'){
        if(is.null(params$num_edges))
            if(is.null(params$min_rel_prop) & is.null(params$min_mi))
                params$num_edges = dim(mtx)[2] * 5 # TODO: rationalize this
        if(is.null(params$min_rel_prop))
            params$min_rel_prop = 0
        if(is.null(params$min_mi))
            params$min_mi       = 0 
        if(is.null(params$mc.cores))
            params$mc.cores     = detectCores()/2
        return(params)
    }    
    if(name == 'adj2graph'){
        if(is.null(params$directed))
            params$directed    = FALSE
        if(is.null(params$community))
            params$community   = TRUE
        if(is.null(params$betweenness))
            params$betweenness = FALSE
        if(is.null(params$plot))
            params$plot        = TRUE
        if(is.null(params$vis_network))
            params$vis_network = FALSE
        return(params)
    }
    if(name == 'compare_networks'){
        if(!is.null(params$log))
            params$redim_mtx$log = params$log
        return(params)
    }  
    if(name == 'redim_mtx'){
        if(is.null(params$method))
            params$method      = 'umap'
        if(is.null(params$plot))
            params$plot        = TRUE
        if(!is.null(params$log))
            params$plot_args$log = params$log
        if(params$verbose)
            params$plot_args$verbose = TRUE  
        return(params)
    }  
    if(name == 'plot_redim_mtx'){
        if(length(params$colors) <= 1)
            params$colors = '#b9c2d9'
        if(length(params$shapes) <= 1)
            params$shapes = 1
        if(is.null(params$col.label))
            params$col.label = 'color'
        if(is.null(params$shape.label))
            params$shape.label = 'shape'
        if(is.null(params$legend))
            params$legend = FALSE
        if(is.null(params$palette))
            params$palette = 'Dark2'
        return(params)
    }      
}