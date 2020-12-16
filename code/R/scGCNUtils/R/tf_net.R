# 2020-07-21 09:34 
# elihei  [<eheidari@student.ethz.ch>]
#/Volumes/Projects/scGCN/code/R/scGCNUtils/R/tf_net.R

#' Construct the transcription factor network
#'
#' @description
#' Constructs the transcription factor interaction network based on RegNetwork.
#'
#' @param ... Any additional arguments.
#'
#'
#' @references  Zhi-Ping Liu, Canglin Wu, Hongyu Miao and Hulin Wu (2015). RegNetwork: an integrated database of transcriptional and posttranscriptional regulatory networks in human and mouse. Database 2015. doi: 10.1093/database/bav095 
#'
#'  @author Elyas Heidari
#'
#' @section Additional arguments:
#' \describe{
#' \item{spec}{A string indicating the sapience. possible values = `c('cattle', 'chicken','chimpanzee', 'dog_domestic', 'frog_western clawed', 'human', 'mouse_laboratory','rat', 'zebrafish')`.}
#' \item{tf_subset}{A vector of strings indicating the transcription factors to be selected.}
#' \item{log}{A logical indicating whether to log the computation times.}
#' \item{verbose}{A logical indicating whether to print out the computation steps.}
#' }
#'
#' @return The adjacency matrix as a sparse matrix of type `dgCMatrix`.
#' @export
#'
#'
#'
#' @importFrom  Matrix dgCMatrix
#' @importFrom  tictoc toc tic.log
#' @importFrom  tidyverse %>% map


tf_net <- function(...) {
    # TODO: message
    # set params
    params = list(...)
    if(is.list(params[[1]]))
        params = params[[1]]
    params %<>% set_params('tf_net', 2)
    # load tfs
    tfs = sprintf('data_raw/prior/homo_genes_%s.txt', params$spec) %>%
        fread(header=F) %>% .$V1
    # load reg network
    adj = sprintf('data_raw/prior/homo_adj_%s.txt', params$spec) %>%
            readMM
    dimnames(adj) = list(tfs, tfs)
    if(!is.null(params$tf_subset))
        adj = adj[params$tf_subset, params$tf_subset]
    # symmetrize and return
    adj = (adj + t(adj))
    diag(adj) = 0
    adj = ((adj > 0) * 1) %>% as('dgCMatrix')
    # log
    toc(log=TRUE, quiet=TRUE)
    tic_log = tic.log(format = TRUE)
    messagef(tic_log[[length(tic_log)]], verbose=params$verbose, log=params$log)
    adj
}