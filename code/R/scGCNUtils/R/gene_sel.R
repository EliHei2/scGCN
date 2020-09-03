# 2020-07-16 09:02 
# elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/scGCNUtils/R/gene_sel.R

#' Select genes for the network construction
#'
#' @description
#' Subsets genes to be used as the network nodes
#' 
#'
#' @param data the count matrix with gene symbols as rownames.
#' @param method a string or list of strings indicate methods used to select genes. See the details for more information. (default = 'hvg')
#' @param n number of genes to be selected.
#' @param spec the species, used just when 'tf' is used as a method.
#'
#' @details for method, 'hvg': highly variable genes, 'tf': transcription factors, 'rand': random subsetting.
#'
#'
#' @author Elyas Heidari
#'
#'
#' @return a vector of gene symbols as strings.
#' @export
#'
#'
#'
#' @importFrom  scran modelGeneVar getTopHVGs
#' @importFrom  data.table fread
#' @importFrom  dplyr %>%

sel_genes <- function(data, method = 'hvg', n = 300, spec = 'human'){
    load_tfs <- function(){
        f = sprintf('data_raw/prior/homo_genes_%s.txt', spec) %>% fread(header=F)
        f$V1
    }
    genes = rownames(data)
    if(n > length(genes)){
        warning('n > #genes, sets n = #genes')
        n = length(genes)
    }
    if('hvg' %in% method){
        gene_vars = modelGeneVar(data)
        genes = getTopHVGs(gene_vars) 
    }
    if('tf' %in% method){
        tf = load_tfs()
        if(n > length(tf)){
            warning('n > #TFs, sets n = #TFs')
            n = length(tf)
        }
        genes = genes[genes %in% tf]
    }
    if('rand' %in% method){
        return(sample(genes, n, replace=F))
    }
    genes %>% .[1:n]
}

