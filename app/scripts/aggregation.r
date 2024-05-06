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

aggregate_ci_results <- function(ci_data) {
    citestResultsList <- list()
    index <- 1
    for (cur_dat in ci_data) {
      citestResultsList[[index]] <- cur_dat
      index <- index + 1
    }

    suffStat <- list()
    suffStat$citestResultsList <- citestResultsList
    # call IOD.
    alpha <- 0.05
    iod_out <- IOD(suffStat, alpha)
    iod_out
}



# # show the output.
# iod_out$Gi_PAG_list # list of PAGs generated from each dataset
# lapply(iod_out$Gi_PAG_list, renderAG)

# iod_out$G_PAG_List # list of possible merged PAGs
# lapply(iod_out$G_PAG_List, renderAG)