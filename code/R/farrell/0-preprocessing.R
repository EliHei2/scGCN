# 2020-04-15 17:57
# elihei [<eheidari@student.ethz.ch>]
# /Volumes/Projects/scGCN/code/R/farrell/0-preprocessing.R 


## Load Dependencies                =================================
suppressMessages(source('code/R/dep.R'))

## Preprocess farrell et al.'s data =================================
message('   farrell: 0-preprocessing.R')

message('   0-initialization')
data_raw  = 'data_raw/farrell'
data_tidy = 'data_tidy/farrell'
fit_mem   = 10000 # how many cells in each split?
rho       = 0.1 
output    = sprintf('output/farrell/rho_%s_', rho)

message('   1- load data')
# load count matrix
count_matrix = fread(file.path(data_raw, 'URD_Dropseq_Expression_Log2TPM.txt'))
count_matrix[, GENE := tolower(GENE)]
setkey(count_matrix, 'GENE')
# load meta data
meta_data    = fread(file.path(data_raw, 'URD_Dropseq_Meta.txt'))
meta_data    = meta_data[-1]
lineage_cols = grep('Lineage', names(meta_data))
meta_data %>%
    .[, (lineage_cols):=lapply(.SD, as.logical), .SDcols = lineage_cols] %>%
    setnames(lineage_cols, gsub('Lineage_', '', names(.)[lineage_cols])) %>%
    setnames(lineage_cols, gsub('_', ' ', names(.)[lineage_cols])) %>%
    setnames(lineage_cols, tolower(names(.)[lineage_cols])) %>%
    setnames('NAME', 'barcode') %>%
    setkey('barcode')
    
gc()

message('   2- find highly variable genes')
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
saveRDS(ggm_data, file.path('output/farrell', 'ggm_data.rds'))
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

message('   *- farrell: Preprocessing finished!')
rm(list = ls())
gc()
# 17 April 2020 (Friday)
# 10:09