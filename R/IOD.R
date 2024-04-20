IOD <- function(suffStat, alpha) {

  initSkeletonOutput <- initialSkeleton(suffStat, alpha)
  G <- initSkeletonOutput[[1]]
  IP <- initSkeletonOutput[[2]]
  sepsetList <- initSkeletonOutput[[3]]
  n_datasets <- length(suffStat$citestResultsList)

  #Algorithm 3:

  p <- length(colnames(G))

  possSepList <- setOfPossSep(G,p)

  existingEdges <- adjPairsOneOccurrence(G)

  RemEdges <- getRemEdges(existingEdges,G, possSepList,n_datasets,suffStat)

  power_RemEdges <- powerSet(unique(RemEdges))
  index_possImmList <- 1

  G_PAG_List <- list()
  for (E in power_RemEdges) {
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

    G_PAG_List <- validatePossPags(G_PAG, G_PAG_List, sepsetList, suffStat, IP)
  }
  return(unique(G_PAG_List))
}
