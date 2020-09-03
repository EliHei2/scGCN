#  2020-07-19 20:37 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/verbose.R

#' construct and visualize Gaussian Graphical Models.
#'
#' @description
#' Fit a Gaussian Graphical Model to continuous-valued dataset employing a subset of methods from stepwise AIC, stepwise BIC, stepwise significance test, partial correlation thresholding, edgewise significance test, or glasso.
#' Also visualizes the fitted Graphical Model.
#'
#' @param txt A normalized dataframe or matrix with no missing data of continuous measurements.
#' @param verbose A string or list of strings indicate methods used to construct the model. See the details for more information. (default = 'glasso')
#'
#' @details The function combines the methods to construct the model, that is, the edge set is the intersection of all edge sets each of which is found by a method. The package gRim is used to implement AIC, BIC, and stepwise significance test. The method glasso from the package glasso is used to provide a sparse estimation of the inverse covariance matrix.
#'
#' @references  Højsgaard, S., Edwards, D., & Lauritzen, S. (2012). Graphical Models with R. Springer US. \url{https://doi.org/10.1007/978-1-4614-2299-0}
#' @references  Friedman, J., Hastie, T., & Tibshirani, R. (2007). Sparse inverse covariance estimation with the graphical lasso. Biostatistics, 9(3), 432–441. \url{https://doi.org/10.1093/biostatistics/kxm045}
#' @references  Abreu, G. C. G., Edwards, D., & Labouriau, R. (2010). High-Dimensional Graphical Model Search with thegRapHDRPackage. Journal of Statistical Software, 37(1). \url{https://doi.org/10.18637/jss.v037.i01}
#'
#'
#' @author Elyas Heidari




messagef <- function(txt, verbose=TRUE, log=NULL) {
    if(verbose)
        message(txt)
    if(!is.null(log)){
        log_file = sprintf('.logs/%s.log', log)
        cat(txt, file=log_file, sep='\n', append=TRUE)
    }
}
