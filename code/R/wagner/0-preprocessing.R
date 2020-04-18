# 2020-04-17 10:03
# elihei [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/wagner/0-preprocessing.R 


## Load Dependencies                =================================
suppressMessages(source('code/R/dep.R'))

## Preprocess wagner et al.'s data =================================
message('   wagner: 0-preprocessing.R')

message('   0-initialization')
data_raw  = 'data_raw/wagner'
data_tidy = 'data_tidy/wagner'
fit_mem   = 10000 # how many cells in each split?
rho       = 0.1 
output    = sprintf('output/wagner/rho_%s', rho)

message('   1- load data')
files_wagner = list.files(data_raw)
# load count matrix
count_matrix = files_wagner[grep('hpf.csv$', files_wagner)]
count_matrix = file.path(data_raw, count_matrix)
count_matrix = count_matrix %>%
    map(fread) %>%
    map(setnames, 'Row', 'GENE') %>%
    map(function(x) x[, GENE := tolower(GENE)]) %>%
    map(setkey, 'GENE') %>%
    purrr::reduce(merge)
gc()
# load meta data
## cluster meta data
cluster_meta = fread(file.path(data_raw, 'GSE112294_ClusterNames.csv'))
cluster_meta %>% 
    setnames(c('TimePoint(hpf)', 'ClusterID', 'ClusterName'), c('hpf', 'id', 'name')) %>%
    setkey(id) %>%
    .[, id := gsub('1\\..*', '1', as.character(id))] %>%
    .[, name := gsub(".*hpf-", "", name)] %>%
    .[, c('type', 'subtype', 'subsubtype') := tstrsplit(tolower(name), " - ")] %>%
    .[, name:= NULL]
## cell meta data
cell_meta = files_wagner[grep('.txt$', files_wagner)]
cell_meta = file.path(data_raw, cell_meta)
cell_meta = unlist(cell_meta %>% map(fread))
## create integrated meta_data
meta_data = cluster_meta[cell_meta] %>%
    .[, barcode := names(count_matrix[, -c('GENE')])] %>%
    setkey(barcode)
rm(cluster_meta)
rm(cell_meta)
gc()

message('   2- find highly variable genes')
# subset desired cell types
cell_types = unique(meta_data$type)
# filter low quality genes 
## split data to fit into memory ## TODO: run on bigger ram 
row.sums = 0
j = k = 1
for ( i in 1:ceiling(dim(count_matrix)[2] / fit_mem) ) {
    j        = k + 1
    k        = min(i * 10000, dim(count_matrix)[2])
    row.sums = row.sums + rowSums(as.matrix(count_matrix[, j:k]))
}
count_matrix     = count_matrix[row.sums > 200, ]
rm(row.sums)
gc()
# select higly variable genes
hvg             = hv_genes(as.matrix(count_matrix[,-c('GENE')]), 300)
ggm_data        = data.table(t(as.matrix(count_matrix[hvg, -c('GENE')])))
names(ggm_data) = count_matrix[hvg, GENE]
gc()

message('   3- construct graph')
# learn ggm structure from data
ggm_graph = ggm(ggm_data, rho=rho)
isolated  = which(degree(ggm_graph)==0)
ggm_graph = delete.vertices(ggm_graph, isolated)
# visualize graph. Set plot to FALSE if running on server
dummy_var = graph_vis(ggm_graph, plot=T, filetype='pdf', filename=paste0(output, 'network'))
rm(dummy_var)
gc()

message('   4- save outputs')
# save count matrix
fwrite(count_matrix, file.path(data_tidy, 'count_matrix.txt'))
saveRDS(ggm_data, file.path('output/wagner', 'ggm_data.rds'))
# save meta data
fwrite(meta_data, file.path(data_tidy, 'meta_data.txt'))
# save graph object
saveRDS(ggm_graph, paste0(output, 'graph.rds'))
# save adj matrix
adj = as_adj(ggm_graph)
fwrite(as.matrix(adj), file.path(data_tidy, sprintf('rho_%s_adj.txt', rho)))
# save features data table
fwrite(count_matrix[names(ggm_data), ], file.path(data_tidy, sprintf('rho_%s_features.txt', rho)))
##TODO: save cell labels

message('   *- wagner: Preprocessing finished!')
rm(list = ls())
gc()
# 17 April 2020 (Friday)
# 12:59