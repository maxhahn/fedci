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

run_ci_test <- function(data) {
  labels <- colnames(data)
  indepTest <- mixedCITest
  suffStat <- getMixedCISuffStat(dat = data,
                                 vars_names = labels,
                                 covs_names = c())

  citestResults <- getAllCITestResults(data, indepTest, suffStat)
  result <- list(citestResults=citestResults, labels=labels)
  result
}
