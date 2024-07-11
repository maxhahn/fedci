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

adag_out <- getDAG("pdsep_g")
truePAG <- getTruePAG(adag_out$dagg)
trueAdjM <- truePAG@amat
renderAG(trueAdjM)

# create datasets for 3 servers separately

# They are saved in data[[1]], data[[2]], data[[3]]

get_example_data <- function(c, n, f) {

  data <- list()
  for (i in 1:c) {
    adat_out <- FCI.Utils::generateDataset(adag = adag_out$dagg, N=n, type = "continuous")
    cur_full_dat <- adat_out$dat
    data[[i]] <-  cur_full_dat[, sample(1:ncol(cur_full_dat), size = f)] #generated datasets
  }
  data
}

