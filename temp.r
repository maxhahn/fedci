library(FCI.Utils)
library(pcalg)
library(igraph)
library(RBGL)
library(rje)
library(graph)
library(doFuture)
library(gtools)
library(rIOD)

#n_cores <- 8
#plan("multicore", workers = n_cores, gc=TRUE)

run_ci_test <- function(data, max_cond_set_cardinality, filedir, filename) {
  labels <- colnames(data)
  indepTest <- mixedCITest
  suffStat <- getMixedCISuffStat(dat = data,
                                 vars_names = labels,
                                 covs_names = c())
  citestResults <- getAllCITestResults(data,
                                      indepTest,
                                      suffStat,
                                      m.max=max_cond_set_cardinality,
                                      saveFiles=FALSE,
                                      fileid=filename,
                                      citestResults_folder=filedir)
  result <- list(citestResults=citestResults, labels=labels)
  result
}


labelList <- list()

aggregate_ci_results <- function(labelList_, ci_data, alpha) {
    labelList <<- labelList_

    suffStat <- list()
    suffStat$citestResultsList <- ci_data
    suffStat$labelList <- labelList

    # call IOD.
    #alpha <- 0.05
    iod_out <- IOD(labelList, suffStat, alpha)
    index <- 1
    iod_out$G_PAG_Label_List <- list()
    for (gpag in iod_out$G_PAG_List) {
      iod_out$G_PAG_Label_List[[index]] <- colnames(gpag)
      index <- index + 1
    }
    index <- 1
    iod_out$Gi_PAG_Label_List <- list()
    for (gipag in iod_out$Gi_PAG_List) {
      iod_out$Gi_PAG_Label_List[[index]] <- colnames(gipag)
      index <- index + 1
    }
    iod_out
}

iod_on_ci_data <- function(labelList_, suffStat, alpha) {
    labelList <<- labelList_
    suffStat$labelList <- labelList
    iod_out <- IOD(labelList, suffStat, alpha)

    index <- 1
    iod_out$G_PAG_Label_List <- list()
    for (gpag in iod_out$G_PAG_List) {
      iod_out$G_PAG_Label_List[[index]] <- colnames(gpag)
      index <- index + 1
    }

    index <- 1
    iod_out$Gi_PAG_Label_List <- list()
    for (gipag in iod_out$Gi_PAG_List) {
      iod_out$Gi_PAG_Label_List[[index]] <- colnames(gipag)
      index <- index + 1
    }
    iod_out
}



load_pags <- function() {
    load("100randomPAGs.RData")
    #c(truePAGs, subsetsList)
    #tuple <- list(A, B)
    return(list(truePAGs = truePAGs, subsetsList = subsetsList))
}
