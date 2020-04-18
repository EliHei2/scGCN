# 2020-04-15 10:05
# elihei [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/ggm.R 

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

ggm <- function(data, methods = c('glasso'), ...) {

    arguments    = list(...)
    threshold    = arguments$threshold
    significance = arguments$significance
    rho          = arguments$rho

    if (is.null(threshold))
        threshold = 0.05

    if (is.null(significance))
        significance = 0.05

    if (is.null(rho))
        rho = 0.1

    model        = gRim::cmod(~ . ^ ., data=data)
    S            = stats::cov.wt(data, method='ML')$cov
    g_models     = list()

    if ('aic' %in% tolower(methods))
        g_models$aic = gRbase::stepwise(model)

    if ('bic' %in% tolower(methods)) 
        g_models$bic = gRbase::stepwise(model, k=log(nrow(data)))

    if ('test' %in% tolower(methods)) 
        g_models$test = gRbase::stepwise(model, criterion='test')

    if ('threshold' %in% tolower(methods)) {
        PC                 = gRbase::cov2pcor(S)
        Z                  = abs(PC)
        Z[Z < threshold]   = 0
        diag(Z)            = 0
        Z[Z > 0]           = 1
        g.thresh           = as(Z, 'graphNEL')
        g_models$thresh    = gRim::cmod(g.thresh, data=data)
    }

    if ('sin' %in% tolower(methods)) {
        psin             = sinUG(S, n=nrow(data))
        g_models$gsin    = methods::as(getgraph(psin, significance), 'graphNEL')
    }

    if ('glasso' %in% tolower(methods)) {
        C                     = stats::cov2cor(S)
        res.lasso             = glasso::glasso(C, rho=rho)
        AM                    = abs(res.lasso$wi) > threshold
        diag(AM)              = F
        g.lasso               = as(AM, 'graphNEL')
        graph::nodes(g.lasso) = colnames(data)
        g_models$glasso       = g.lasso
    }

    g_models     = g_models %>% map(as, 'igraph')
    commonedges  = do.call(igraph::intersection, g_models)

    commonedges
}


# 11:22


