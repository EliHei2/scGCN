#  2020-07-20 10:40 
#  elihei  [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/01-pipeline-test.R

# Initialize values                =================================
message('   0-initialization')
suppressMessages(source('code/R/dep.R'))
## dirs
tag     = 'test'
verbose = 2
log     = TRUE
spec    = 'human'
cell_types = 'all'
min_num = 500
min_rel_prop = 0.5
balanced = TRUE
ref_tag = 'assay_pbmc1'
gene_sel_method = 'hvg'
n_genes = 500
vis_all = c('pca', 'tsne', 'umap')
plot_all = TRUE
read_cache  = TRUE
write_cache = TRUE

## verbose
verbose_1 = FALSE
verbose_2 = FALSE
if(verbose == 1)
    verbose_1 = TRUE
if(verbose == 2){
    verbose_1 = verbose_2 = TRUE
}

## cache
cache_state = 0
cache_file = file.path('cache', sprintf('%s.RData', tag))
if(read_cache){
    if(!file.exists(cache_file))
        warning('no cache found!')
    else{
        load(cache_file)
    }
}

if(write_cache){
    if(!file.exists(cache_file)){
        warning('no cache found! creates one.')
        save.image(cache_file)
    }
}

if(cache_state == 0){
    tic.clearlog()
    tic('+ load experiment')
    library('tictoc')
    library('furrr')
    sce = load_experiment(
        tag=tag, 
        spec=spec, 
        verbose=verbose_2,
        log=TRUE)
    log_tag = metadata(sce)$tag
    toc(log=TRUE, quiet=TRUE)
    messagef(tic.log(format = TRUE)[[1]], verbose=verbose_1, log=log_tag)
    cache_state = 1
    save.image(cache_file)
}

if(cache_state == 1){
    tic.clearlog()
    tic('+ subset cells')
    sce = subset_cells(
        sce=sce,
        cell_types=cell_types, 
        min_rel_prop=min_rel_prop, 
        min_num=min_num, 
        balanced=balanced, 
        verbose=verbose_2)
    toc(log=TRUE, quiet=TRUE)
    messagef(tic.log(format = TRUE)[[1]], verbose=verbose_1, log=log_tag)
    cache_state = 2
    save.image(cache_file)
}

if(cache_state == 2){
    tic.clearlog()
    tic('+ subset genes')
    genes = subset_genes(
        sce=sce, 
        method=gene_sel_method, 
        n=n_genes, 
        verbose=verbose_2)
    sce = sce[genes, ]
    toc(log=TRUE, quiet=TRUE)
    messagef(tic.log(format = TRUE)[[1]], verbose=verbose_1, log=log_tag)
    cache_state = 3
    save.image(cache_file)
}

if(cache_state == 3){
    tic.clearlog()
    tic('+ dimentionality reduction')
    sce = redim_sce(sce, method=vis_all, plot=plot_all)
    toc(log=TRUE, quiet=TRUE)
    if(plot_all)
        metadata(sce)$redim_plot
    messagef(tic.log(format = TRUE)[[1]], verbose=verbose_1, log=log_tag)
    cache_state = 4
    save.image(cache_file)
}

