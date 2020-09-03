message('   0-initialization')
suppressMessages(source('code/R/dep.R'))
list.files('code/R/scGCNUtils/R') %>% file.path('code/R/scGCNUtils/R', .) %>% map(source)
library('tictoc')
library('furrr')
## dirs
tag     = 'test'
verbose = 2
log     = TRUE
read_cache  = FALSE
write_cache = FALSE

## verbose
verbose_1 = FALSE
verbose_2 = FALSE
if(verbose == 1)
    verbose_1 = TRUE
if(verbose == 2){
    verbose_1 = verbose_2 = TRUE
}

params = list()
params$load_experiment = list( 
    spec='human', 
    verbose=verbose_1, 
    log=TRUE)
params$subset_cells    = list(
    cell_types='all', 
    min_num=500, 
    min_rel_prop=0.5,
    verbose=verbose_1,  
    balanced=TRUE)
params$subset_genes    = list(
    n=300, 
    verbose=verbose_1, 
    method=c('tf', 'hvg'))
# TODO: redim sce
params$construct_graph = list(
    method=c('ggm'), 
    agg_fun='union', 
    verbose=verbose_1, 
    ggm=list(method='glasso', rho=0.05, verbose=verbose_2, mc.cores=3),
    vis=list(plot=TRUE))
params$compare_networks = list(
    verbose=verbose_1, 
    redim_mtx=list(
        method='umap', 
        plot=TRUE, 
        verbose=verbose_2,
        plot_args=list(
            title='Node Embeddings, train & test', 
            col.label='community',
            shape.label='dataset',
            verbose=verbose_2,
            legend=TRUE)))


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
    sce = load_experiment(tag=tag, ...=params$load_experiment)
    log_tag = metadata(sce)$tag
    cache_state = 1
    if(write_cache)
        save.image(cache_file)
}

if(cache_state == 1){
    sce = subset_cells(sce=sce, ...=append(params$subset_cells, list(log=log_tag)))
    cache_state = 2
    if(write_cache)
        save.image(cache_file)
}

if(cache_state == 2){
    sce = subset_genes(sce=sce, ...=append(params$subset_genes, list(log=log_tag)))
    cache_state = 3
    if(write_cache)
        save.image(cache_file)
}

# if(cache_state == 3){
#     tic.clearlog()
#     tic('+      dimentionality reduction')
#     sce = redim_sce(sce, method=vis_all, plot=plot_all)
#     toc(log=TRUE, quiet=TRUE)
#     if(plot_all)
#         metadata(sce)$redim_plot
#     messagef(tic.log(format = TRUE)[[1]], verbose=verbose_1, log=log_tag)
#     cache_state = 4
#     save.image(cache_file)
# }

if(cache_state == 3){
    sce = construct_graph(sce, ...=append(params$construct_graph, list(log=log_tag)))
    cache_state = 4
    if(write_cache)
        save.image(cache_file)
}

if(cache_state == 4){
    sce = compare_networks(sce, ...=append(params$compare_networks, list(log=log_tag)))
    cache_state = 5
    if(write_cache)
        save.image(cache_file)
}

