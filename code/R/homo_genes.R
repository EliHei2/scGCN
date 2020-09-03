# 2020-07-16 15:43 
# elihei  [<eheidari@student.ethz.ch>]
#/Volumes/Projects/scGCN/code/R/homo_genes.R

## Load dependencies                =================================
suppressMessages(source('code/R/dep.R'))

## Initialize values                =================================
message('   0-initialization')
data_raw = 'data_raw/prior'
# read
homo_f = file.path(data_raw, 'HOM_AllOrganism.rpt.txt') 
human_regNet_f = file.path(data_raw, 'human/human.source')
# write
homo_dt_f = file.path(data_raw, 'homo_dt.txt')
homo_adj_mat_f = file.path(data_raw, 'homo_adj_%s.txt')
homo_genes_f = file.path(data_raw, 'homo_genes_%s.txt')

## Load data                        =================================
message('   1- load data')
# homologus genes
homo_dt = homo_f %>% fread %>% .[, c(1,2,4, 12)] %>%
    setnames(c('ID', 'organism', 'symbol', 'Synonyms'))
# human regulatory network of TFs from regNetwork
human_regNet = human_regNet_f %>% fread %>% .[, .(V1, V3)] %>%
    .[, lapply(.SD, tolower)] %>%
    .[V3 %in% V1]

## Preprocess data                  =================================
message('   2- preprocess data')
# add synonym gene symbols to have complete data
tmp = dim(homo_dt)[1]
to.bind = 1:tmp %>% # takes a while!
    map(function(x) (strsplit(homo_dt[x]$Synonyms, '|', fixed=T) %>%
        map(function(y) homo_dt[x, .(ID, organism, symbol=y)]))[[1]]) %>%
    purrr::reduce(rbind)
homo_dt = rbind(homo_dt[, 1:3], to.bind) %>% .[!is.na(symbol)]
# cast to a usable version
homo_cast_dt = unique(homo_dt) %>% 
    dcast(ID ~ organism, value.var='symbol', fun.aggregate=function(x) paste(x[1])) %>%
    as.data.table %>%
    .[, lapply(.SD, tolower)]
# create sparse adj matrix for human regNet
genes = union(human_regNet$V1, human_regNet$V3)
adj_human_regNet = matrix(0, nrow=length(genes), ncol=length(genes))
dimnames(adj_human_regNet) = list(genes, genes)
i = match(human_regNet$V1, genes)
j = match(human_regNet$V3, genes)
adj_human_regNet = sparseMatrix(i = i, j = j, x= 1, dims=c(length(genes),
                            length(genes)), dimnames=list(genes, genes))
# create homologus regNets
homo_adjs = colnames(homo_cast_dt)[-1] %>% 
    map(function(x) intersect(c(homo_cast_dt[,..x])[[1]], genes)) %>%
    map(~(adj_human_regNet[.x, .x])) 
names(homo_adjs) = gsub(', ', '_', colnames(homo_cast_dt)[-1])

## Save results                  =================================
message('   3- save results')
homo_dt %>% fwrite(homo_dt_f)
names(homo_adjs) %>% map(~writeMM(homo_adjs[[.x]], sprintf(homo_adj_mat_f, .x)))
names(homo_adjs) %>% map(~write(rownames(homo_adjs[[.x]]), sprintf(homo_genes_f, .x)))