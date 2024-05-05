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

###
# GENERATE RANDOM DATA
###

# # They are saved in data[[1]], data[[2]], data[[3]]
# adag_out <- getDAG("pdsep_g")
# truePAG <- getTruePAG(adag_out$dagg)
# trueAdjM <- truePAG@amat
# renderAG(trueAdjM)

# data <- list()
# for (i in 1:3) {
#   adat_out <- FCI.Utils::generateDataset(adag = adag_out$dagg, N=100000, type = "continuous")
#   cur_full_dat <- adat_out$dat
#   data[[i]] <-  cur_full_dat[, sample(1:ncol(cur_full_dat), size = 3)] #generated datasets
#   write.csv(data[[i]], paste0('./random-data-', i, '.csv'), row.names=FALSE)
# }

###
# PREPARE DATA
###

run_ci_test <- function(data) {
  x <- list()
  labels <- colnames(data)
  indepTest <- mixedCITest
  suffStat <- getMixedCISuffStat(dat = data,
                                 vars_names = labels,
                                 covs_names = c())

  citestResults <- getAllCITestResults(data, indepTest, suffStat)
  result <- list(citestResults=citestResults, labels=labels)
  x[[1]] <- result
  print(result)
  print(x)
  result
}

# # run the citests separately
# citestResultsList <- list()
# index <- 1
# for (cur_dat in data) {
#   #this is how to run CI Tests for a dataset cur_dat
#   cur_labels <- colnames(cur_dat)
#   indepTest <- mixedCITest
#   suffStat <- getMixedCISuffStat(dat = cur_dat,
#                                  vars_names = cur_labels,
#                                  covs_names = c())

#   citestResults <- getAllCITestResults(cur_dat, indepTest, suffStat)
#   citestResultsList[[index]] <- list(citestResults=citestResults, labels=cur_labels)
#   index <- index + 1
# }

# create the list of the suffstat
#suffStat <- list()
#suffStat$citestResultsList <- citestResultsList