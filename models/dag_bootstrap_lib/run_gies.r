suppressMessages(library(pcalg))
suppressMessages(library(gRbase))

run_gies_multitarget = function(data, all_targets, target_index_per_sample,path,i){
  data = as.data.frame(data)
  p = ncol(data)
  colnames(data) = as.character(1:p)
  gie_score_fn <- new("GaussL0penIntScore", data, all_targets, target_index_per_sample) # BIC score
  gies.fit <- gies(gie_score_fn)
  W = gies.fit$repr$weight.mat()
  colnames(W) = as.character(1:p)
  write.csv(W, paste(path, i, sep=''), row.names=FALSE)
}

args = commandArgs(trailingOnly=TRUE)
data_path = args[1]
unique_targets_path = args[2]
target_indices_path = args[3]
path = args[4]
boot_step = as.numeric(args[5])

data = as.data.frame(read.table(data_path))

unique_targets = read.csv(unique_targets_path, col.names=paste0("V",seq_len(dim(data)[2])+1), header=FALSE)
unique_targets_list = list()
for (i in 1:nrow(unique_targets)){
    temp = unique_targets[i, ][!is.na(unique_targets[i, ])]
    if (any(temp == -1)){
        temp = integer(0)
    }
    unique_targets_list[[i]] = as.integer(temp)
}
target_indices = read.csv(target_indices_path, header=FALSE)[, 1]
run_gies_multitarget(data,unique_targets_list,target_indices,path,boot_step)
