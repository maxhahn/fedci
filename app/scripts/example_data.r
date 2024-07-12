library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(rIOD)

n_cores <- 8
plan("multicore", workers = n_cores, gc=TRUE)



# create datasets for 3 servers separately

# They are saved in data[[1]], data[[2]], data[[3]]

get_example_data <- function(dag_type, num_clients, num_samples, num_vars) {

  adag_out <- getDAG(dag_type)
  truePAG <- getTruePAG(adag_out$dagg)
  trueAdjM <- truePAG@amat
  renderAG(trueAdjM)

  data <- list()
  for (i in 1:num_clients) {
    adat_out <- FCI.Utils::generateDataset(adag = adag_out$dagg, N=num_samples, type = "continuous")
    cur_full_dat <- adat_out$dat
    data[[i]] <-  cur_full_dat[, sample(1:ncol(cur_full_dat), size = num_vars)] #generated datasets
  }
  data
}

