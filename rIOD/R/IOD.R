#' @export IOD
IOD <- function(suffStat, alpha) {

  initSkeletonOutput <- initialSkeleton(suffStat, alpha)
  G <- initSkeletonOutput[[1]]
  IP <- initSkeletonOutput[[2]]
  sepsetList <- initSkeletonOutput[[3]]
  Gi_list <- initSkeletonOutput[[4]]

  n_datasets <- length(suffStat$citestResultsList)

  # lapply(initSkeletonOutput[[4]], renderAG)
  # renderAG(G)

  # Algorithm 3:

  p <- length(colnames(G))

  possSepList <- setOfPossSep(G,p)

  existingEdges <- adjPairsOneOccurrence(G)

  RemEdges <- getRemEdges(existingEdges,G, possSepList,n_datasets,suffStat)


  power_RemEdges <- powerSet(unique(RemEdges))
  # one_edge_list <- which(lapply(power_RemEdges, length) == 1)

  index_possImmList <- 1

  #G_PAG_List <- list()
  #for (E in power_RemEdges) {
  G_PAG_List <- foreach (E = power_RemEdges, .verbose=TRUE) %dofuture% {
    H <- induceSubgraph(G,E)
    labelsG <- colnames(G)
    PossImm <- getPossImm(H, n_datasets, suffStat, sepsetList, labelsG)

    listAllHt <- list()
    if (length(PossImm) > 0) {
      power_possImm <- all_combinations(PossImm)
      for (t in power_possImm) {
        H_t <- H
        # orient Colliders
        for (tau in t) {
          if (!is.null(tau)) {
            H_t[tau[1], tau[2]] <- 2
            H_t[tau[3], tau[2]] <- 2
          }
        }
        listAllHt[[length(listAllHt)+1]] <- H_t
      }
    } else {
      listAllHt[[length(listAllHt)+1]] <- H
    }

    G_PAG <- applyRulesOnHt(unique(listAllHt))
    # For each possible G in the power set of graphs you are creating, make sure
    # to update sepset accordingly.
    G_PAG <- unique(G_PAG)

    violation_List <- validatePossPags(G_PAG, sepsetList, suffStat, IP)
    G_PAG <- G_PAG[!violation_List]

    return(G_PAG)
  }

  G_PAG_List <- unique(unlist(G_PAG_List, recursive=F))

  return(list(G_PAG_List=G_PAG_List, Gi_PAG_list=Gi_list))
}
