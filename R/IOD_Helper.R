# PS: never transfer/combine data, only statistics
# Idea:
#   1) compute p-values for each dataset and save in a citestResults
#   2) write a iodCITest that combines p-values based on the citestResults of each dataset

#' @param suffStat a list with the following entries:
#'      cur_labels: these are the names of the variables in the dataset to
#'                  which the test is being applied.
#'      citestResultsList: This is a list where each entry is also a list
#'                         containing both citestResults and the corresponding
#'                         labels for each dataset.
iodCITest <- function(x, y, S, suffStat) {
  xname <- suffStat$cur_labels[x]
  yname <- suffStat$cur_labels[y]
  snames <- suffStat$cur_labels[S]

  citestResultsList <- suffStat$citestResultsList
  k <- 0
  n_datasets <- length(citestResultsList)
  p <- rep(1, n_datasets)

  for (i in 1:n_datasets) {
    labels_i <- citestResultsList[[i]]$labels
    required_labels <- c(xname, yname, snames)
    if (all(required_labels %in% labels_i)) {

      citestResults_i <- citestResultsList[[i]]$citestResults
      required_results <- extractValidCITestResults(citestResults_i,
                                                    labels_i, required_labels)
      colnames(required_results) <- c("ord", "X", "Y", "S", "pvalue")
      required_results[, c(1:3,5)] <- lapply(required_results[, c(1:3,5)], as.numeric)

      x <- which(required_labels == xname) # should be always 1
      y <- which(required_labels == yname) # should be always 2
      S <- which(required_labels %in% snames) # should be always 3:length(required_labels)
      sStr <- getSepString(S)
      resultsxys <- c()
      if (!is.null(required_results)) {
        # the test should be the symmetric for X,Y|S and Y,X|S
        resultsxys <- subset(required_results, X == x & Y == y & S == sStr)
        resultsyxs <- subset(required_results, X == y & Y == x & S == sStr)
        resultsxys <- rbind(resultsxys, resultsyxs)
      }
      p[i] <- resultsxys[1, "pvalue"]
      k <- k+1
    } else {
      p[i] <- 1
    }
  }
  test_statistic <- -2 * sum(log(p))
  df <- 2*k

  p_value <- pchisq(test_statistic, df, lower.tail = FALSE) # H0: Independency

  return(p_value)
}

# Initialize G with edges between all nodes
initG <-function(suffStat){
  labelsG <- collectLabelsG(suffStat)
  print(labelsG)
  p <- length(labelsG)
  print(p)
  G <- matrix(1, nrow = p, ncol = p)
  G <- G - diag(p)
  colnames(G) <- rownames(G) <- labelsG
  return(G)
}

collectLabelsG <- function(suffStat){
  n_datasets <- length(suffStat$citestResultsList)
  lists <- suffStat$citestResultsList
  labelsG <- list()
  for (i in 1:n_datasets){
    labelsG[[i]] <- lists[[i]]$labels
  }
  return(unique(unlist(labelsG)))
}

#adjust G with information from skeleton alg
remEdgesFromG <- function(sepset, G, cur_labels){
  n_sepset <- length(sepset)
  for (i in 1:n_sepset){
    for (j in 1:n_sepset){
      if (!is.null(sepset[[i]][[j]])){
        labelX <- as.character(cur_labels[i])
        labelY <- as.character(cur_labels[j])
        G[labelX, labelY] <- 0
        G[labelY, labelX] <- 0
      }
    }
  }
  return(G)
}

adjPairsOneOccurrence <- function(G){
  labelsG <- colnames(G)
  neighbours <- list()
  G_copy <- G
  renderAG(G)
  index <- 1

  for (j in 1:length(labelsG)) {
    cur_adj <- getAdjNodes(G_copy, labelsG[j])
    if(length(cur_adj) > 0){
      for(adj in cur_adj){
        neighbours[[index]] <- c(labelsG[j],adj)
        index <- index + 1
        #only one occurrence
        G_copy[adj,labelsG[j]] <- 0
        G_copy[labelsG[j],adj] <- 0
      }
    }
  }
  return(neighbours)
}

setOfPossSep <- function(G, p, pdsep.max= Inf, m.max = Inf){
  labelsG <- colnames(G)
  possSepG <- lapply(seq_len(p), function(.) vector("list", p))
  #this outputs all PossSep(x,G) which are the nodes where a path exists from x
  #that they are possible separating
  allPdsep <- lapply(1:p, qreach, amat = G)
  for(x in seq_len(p)) {
    cur_adj_x <- getAdjNodes(G,x)
    for(y in seq_len(p)){
      cur_adj_y <- getAdjNodes(G,y)
      if(y %in% allPdsep[[x]]){

        X <- labelsG[x]
        Y <- labelsG[y]

        allPdsepLabels_x <- labelsG[allPdsep[[x]]]
        allPdsepLabels_y <- labelsG[allPdsep[[y]]]

        pdsep_x <- setdiff(allPdsepLabels_x, list(Y, cur_adj_x))
        pdsep_y <- setdiff(allPdsepLabels_y, list(X, cur_adj_y))

        #this gets the paths between x and y for which the condition is true
        unionOfSetXY <-  intersect(pdsep_x,  pdsep_y)

        possSepG[[x]][[y]] <- unionOfSetXY
      }
    }
  }
  return(possSepG)
}

induceSubgraph <- function(G, edges){
  H <- G
  for (edge in edges) {
    H[edge[1], edge[2]] <- H[edge[2], edge[1]] <- 0
  }

  return(H)
}

all_combinations <- function(lst) {
  result <- list(list()) # also empty set
  for (i in 1:length(lst)) {
    result <- c(result, combn(lst, i, simplify = FALSE))
  }
  return(result)
}


#https://rdrr.io/cran/pcalg/src/R/pcalg.R
#They say minDiscrPath is in pcalg, but I couldnt find it
minDiscrPath <- function(pag, a,b,c, verbose = FALSE){
  ## Purpose: find a minimal discriminating path for a,b,c.
  ## If a path exists this is the output, otherwise NA
  ## ----------------------------------------------------------------------
  ## Arguments: - pag: adjacency matrix
  ##            - a,b,c: node positions under interest
  ## ----------------------------------------------------------------------
  ## Author: Diego Colombo, Date: 25 Jan 2011; speedup: Martin Maechler

  p <- as.numeric(dim(pag)[1])
  visited <- rep(FALSE, p)
  visited[c(a,b,c)] <- TRUE # {a,b,c} "visited"
  ## find all neighbours of a  not visited yet
  indD <- which(pag[a,] != 0 & pag[,a] == 2 & !visited) ## d *-> a
  if (length(indD) > 0) {
    path.list <- updateList(a, indD, NULL)
    while (length(path.list) > 0) {
      ## next element in the queue
      mpath <- path.list[[1]]
      m <- length(mpath)
      d <- mpath[m]
      if (pag[c,d] == 0 & pag[d,c] == 0)
        ## minimal discriminating path found :
        return( c(rev(mpath), b,c) )

      ## else :
      pred <- mpath[m-1]
      path.list[[1]] <- NULL


      ## d is connected to c -----> search iteratively
      if (pag[d,c] == 2 && pag[c,d] == 3 && pag[pred,d] == 2) {
        visited[d] <- TRUE
        ## find all neighbours of d not visited yet
        indR <- which(pag[d,] != 0 & pag[,d] == 2 & !visited) ## r *-> d
        if (length(indR) > 0)
          ## update the queues
          path.list <- updateList(mpath[-1], indR, path.list)
      }
    } ## {while}
  }
  ## nothing found:  return
  NA
} ## {minDiscrPath}

updateList <- function(path, set, old.list){
  ## Purpose: update the list of all paths in the iterative functions
  ## minDiscrPath, minUncovCircPath and minUncovPdPath
  ## ----------------------------------------------------------------------
  ## Arguments: - path: the path under investigation
  ##            - set: (integer) index set of variables to be added to path
  ##            - old.list: the list to update
  ## ----------------------------------------------------------------------
  ## Author: Diego Colombo, Date: 21 Oct 2011; Without for() by Martin Maechler
  c(old.list, lapply(set, function(s) c(path,s)))
}

#Note: There is no sepset for G, so we create both graphs
#maybe document what we put in the sepset of G
newRule4 <- function(pag, p) {
  applied <- FALSE
  #orig_pag_objs <- list(pag, sepset)
  #out <- list(orig_pag_objs)
  out_pags <- list(pag)
  verbose <- TRUE
  seq_p <- seq_len(p)

  # #initialize sepset
  # sepset <- lapply(seq_p, function(.) vector("list", p))
  jci ='0' #no knowledge

  ind <- which((pag != 0 & t(pag) == 1), arr.ind = TRUE)## b o-* c
  while (length(ind) > 0) {
    b <- ind[1, 1]
    c <- ind[1, 2] # pag[b,c] != 0, pag[c,b] == 1
    ind <- ind[-1,, drop = FALSE]
    ## find all a s.t. a -> c and a <-* b
    indA <- which((pag[b, ] == 2 & pag[, b] != 0) &
                    (pag[c, ] == 3 & pag[, c] == 2))
    # pag[b,a] == 2, pag[a,b] != 0
    # pag[c,a] == 3, pag[a,c] == 2
    ## chose one a s.t. the initial triangle structure exists and the edge hasn't been oriented yet
    while (length(indA) > 0 && pag[c,b] == 1) {
      a <- indA[1]
      indA <- indA[-1]
      ## path is the initial triangle
      ## abc <- c(a, b, c)
      ## Done is TRUE if either we found a minimal path or no path exists for this triangle
      Done <- FALSE
      ### MM: FIXME?? Isn't  Done  set to TRUE in *any* case inside the following
      ### while(.), the very first time already ??????????
      while (!Done && pag[a,b] != 0 && pag[a,c] != 0 && pag[b,c] != 0) {
        ## find a minimal discriminating path for a,b,c
        md.path <- minDiscrPath(pag, a,b,c, verbose = verbose)
        ## if the path doesn't exists, we are done with this triangle
        if ((N.md <- length(md.path)) == 1) {
          Done <- TRUE
        } else {
          applied <- TRUE
          pag1 <- pag2 <- pag
          ## a path exists
          ## if b is in sepset

          #NOTE: we dont know about the sepset
          #create both graphs
          #b could be v from the lecture

          # if ((b %in% sepset[[md.path[1]]][[md.path[N.md]]]) ||
          #   (b %in% sepset[[md.path[N.md]]][[md.path[1]]])) {
          #   if (verbose)
          #     cat("\nRule 4",
          #           "\nThere is a discriminating path between",
          #           md.path[1], "and", c, "for", b, ",and", b, "is in Sepset of",
          #           c, "and", md.path[1], ". Orient:", b, "->", c, "\n")

          #FOUND DISCRIMINATING PATH, DO THE THINGS MENTIONED IN THE PAPER

          # TODO for Alina: we should think about updating the sepsets in the entire code
          #sepset1 <- sepset
          #sepset1[[md.path[1]]][[md.path[N.md]]] <- b
          pag1[b, c] <- 2
          pag1[c, b] <- 3
          # }
          # else {
          #   ## if b is not in sepset
          #   if (verbose)
          #     cat("\nRule 4",
          #         "\nThere is a discriminating path between",
          #         md.path[1], "and", c, "for", b, ",and", b, "is not in Sepset of",
          #         c, "and", md.path[1], ". Orient:", a, "<->", b, "<->",
          #         c, "\n")

          #FOUND DISCRIMINATING PATH, DO THE THINGS MENTIONED IN THE PAPER
          pag2[b,c] <- pag2[c,b] <- 2
          if( pag2[a,b] == 3 ) { # contradiction with earlier orientation!
            if( verbose )
              cat('\nContradiction in Rule 4b!\n')
            if( jci == "0" ) {
              pag2[a,b] <- 2 # backwards compatibility
            }
          } else { # no contradiction
            pag2[a,b] <- 2
          }
          out_pags <- list(pag1, pag2)
        }
        Done <- TRUE
      }
    }
  }
  return(out_pags)
}

computePossImm <- function(PossImm){
  new_PossImm <- list()
  for(triplet in PossImm){

    if(!is.null(triplet)){
      flipped_triplet <- c(triplet[3],triplet[2],triplet[1])
      if(!any(sapply(new_PossImm, function(x) identical(x, triplet))) &&
         !any(sapply(new_PossImm, function(x) identical(x, flipped_triplet)))){
        new_PossImm[[length(new_PossImm)+1]] <- triplet
      }
    }
  }
  return(new_PossImm)
}

colliderOrientation <- function(amat,G, sepset){

  labels_Gi <- colnames(amat)

  unshieldedTr <- find.unsh.triple(amat,check=TRUE)
  if (length(unshieldedTr$unshVect > 0)) {
    for (j in 1:ncol(unshieldedTr[[1]])) {
      unshTripl <-  unshieldedTr[[1]][,j] # (unshieldedTr[1]$unshTripl)
      start_node <- unshTripl[1]  #rownames(amat)[unshTripl[1]]
      middle_node <- unshTripl[2] #rownames(amat)[unshTripl[2]]
      end_node <- unshTripl[3]    #rownames(amat)[unshTripl[3]]

      cur_sep <- sepset[[start_node]][[end_node]]
      if (!(is.null(cur_sep))) {
        # check if middle_node belongs to sepset[[start_node]][[[end_node]]]
        # and proceed only if not.
        if (!(middle_node %in% cur_sep)) {
          #used Coding for type amat.pag from pcalg page 12
          if (G[labels_Gi[start_node],  labels_Gi[middle_node]] != 0) { #AND EDGE EXISTS IN G,
            cat("(1) Adding an arrowhead using dataset with variables ", paste0(labels_Gi, collapse=","), "\n")
            G[labels_Gi[start_node],  labels_Gi[middle_node]] <- 2 # start_node o--> middle_node
          }
          if (G[labels_Gi[end_node],  labels_Gi[middle_node]] != 0) { #AND EDGE EXISTS IN G
            cat("(2) Adding an arrowhead using dataset with variables ", paste0(labels_Gi, collapse=","), "\n")
            G[labels_Gi[end_node],  labels_Gi[middle_node]] <- 2
          }
        }
      }
    }
  }
  return(G)
}

getSubsets <- function(nodes) {
  return(powerSet(nodes))
}

# This is Algorithm 2, note that it applies the skeleton + pdsep + collider orientation
# in each dataset using the iodCITest.
# We can run this only requiring the suffStat of the iodCITest
# cur_labels does not need to be defined in the input
initialSkeleton <- function(suffStat, alpha){
  G <- initG(suffStat)
  n_datasets <- length(suffStat$citestResultsList)
  listGi <- list()
  IP <- list()
  index <- 1
  sepsetList <- list()
  listGiRaw <- list()

  for (i in 1:n_datasets) {
    cur_labels <- suffStat$citestResultsList[[i]]$labels
    suffStat$cur_labels <- cur_labels
    #skeleton
    skel.fit <- skeleton(suffStat = suffStat,
                         indepTest = iodCITest,
                         method = "stable",
                         alpha = alpha, labels = cur_labels,
                         verbose = TRUE, NAdelete = FALSE)
    # skeleton removes edges from independent nodes in Gi
    # orients Colliders of order 0

    sepset <- skel.fit@sepset
    sepset <- fixSepsetList(sepset)

    amat.Gi <-  as(skel.fit@graph, "matrix")
    renderAG(amat.Gi, add_index = TRUE)

    G <- remEdgesFromG(sepset, G, cur_labels) # here the edges that are removed
    # before in Gi are removed from G

    p <- length(cur_labels)
    pdsepRes <- pdsep(skel=skel.fit@graph, suffStat, indepTest = iodCITest,
                      p = p, sepset = sepset, alpha = alpha,
                      pMax = skel.fit@pMax,
                      m.max = Inf, pdsep.max = Inf,
                      NAdelete = FALSE,
                      verbose = TRUE)
    sepset <- pdsepRes$sepset # sepset from the final skeleton
    sepsetList[[i]] <- sepset

    # removing from G the edges removed from Gi after calling pdsep
    G <- remEdgesFromG(sepset, G, cur_labels)

    # adding the remaining edges to IP
    n_sepset <- length(sepset)
    for (k in 1:(n_sepset-1)) {
      for (j in (k+1):n_sepset) {
        sepset <- sepsetList[[i]]
        #if the Sepset is NULL there is no Seperator and the edge is not removed
        if(is.null(sepset[[k]][[j]])){
          labelX <- as.character(cur_labels[k])
          labelY <- as.character(cur_labels[j])
          IP[index] <- list(c(labelX, labelY, cur_labels))
          index <- index + 1
        }
      }
    }

    listGi[[i]] <- pdsepRes$G
    #pdsepRes$G is the final skeleton of Gi
    G <- colliderOrientation(amat.Gi, G, sepset)
  }

  listGi <- lapply(1:length(listGi), function(x) {
    udag2pag(listGi[[x]], rules = rep(TRUE, 10),
             orientCollider = TRUE, sepset = sepsetList[[x]]) })

  return(list(G, IP, sepsetList, listGi))
}

getPossImm <- function(H, n_datasets,suffStat,sepsetList, labelsG){
  PossImm <- list()
  for (z in colnames(H)) {
    adj_z <- which(H[z, ] != 0)
    for (x in adj_z) {
      for (y in adj_z) {
        if (x != y) {
          #Can X,Z,Y be made an immorality? -> if H[X,Y]=0 (unshielded, symmetric)
          if (H[x,y] == 0) {

            conditionsforAllVi<- list()
            for (i in 1:n_datasets) {

              conditionsforAllVi[i] <- FALSE
              cur_labels <- suffStat$citestResultsList[[i]]$labels
              sepsetGi <- sepsetList[[i]]
              #check if x,y in GI
              # Labels of G are the same as labels of H

              x_label <- labelsG[x]
              y_label <- labelsG[y]

              #check if Sepset is undefined,i.e. X,Y are not both observed in Gi
              if(!(x_label %in% cur_labels) | !(y_label %in% cur_labels)){
                conditionsforAllVi[i] <- TRUE
              }else{
                # check if Z is not in Vi
                if(!(z %in% cur_labels)){
                  conditionsforAllVi[i] <- TRUE
                }
              }
            }

            if(all(unlist(conditionsforAllVi))){
              PossImm[[length(PossImm)+1]] <- c(x_label,z,y_label)
            }
          }
        }
      }
    }
  }
  PossImm <- computePossImm(PossImm) #removes flipped occurencies
  return(PossImm)
}

getRemEdges <- function(existingEdges,G, possSepList,n_datasets,suffStat){
  RemEdges <- list()
  for (pair in existingEdges) {
    X <- pair[1]
    Y <- pair[2]
    labelsG <- colnames(G)
    index_X <- which(labelsG == X)
    index_Y <- which(labelsG == Y)

    AdjX <- getAdjNodes(G, X)
    AdjY <- getAdjNodes(G, Y)

    cur_possSep <- c(possSepList[[index_X]][[index_Y]])

    if (length(cur_possSep) > 0) {
      set1 <- unique(unlist(c(pair, AdjX, cur_possSep)))
      set2 <- unique(unlist(c(pair, AdjY, cur_possSep)))
    }
    else{
      set1 <- unique(unlist(c(pair, AdjX))) #currPossSep can be char(0)
      set2 <- unique(unlist(c(pair, AdjY)))
    }

    flagRemEdge <- TRUE
    for (i in 1:n_datasets) {
      cur_labels <- suffStat$citestResultsList[[i]]$labels

      #setdiff outputs elements that are in vector1 and not in vector2
      if (length(setdiff(set1, cur_labels)) == 0) {
        flagRemEdge <- FALSE
      }
      if (length(setdiff(set2, cur_labels)) == 0) {
        # enters here onl< if the possSep was already entirely observed in some Vi
        flagRemEdge <- FALSE
      }
    }

    if (flagRemEdge) {
      RemEdges[[length(RemEdges)+1]] <- pair
    }
  }
  return(RemEdges)
}

applyRulesOnHt <- function(listAllHt){

  rules <- rep(TRUE,10)
  rules[4] <- FALSE
  orientCollider = FALSE
  sepset <- list()

  done_flags <- rep(FALSE, length(listAllHt))
  graphs_done <- list()
  G_PAG <- list()

  while (!all(done_flags)) {

    listPagsLength <- length(listAllHt)

    for (j in 1:listPagsLength) {
      if (!done_flags[j]) {
        new_graph <- udag2pag(listAllHt[[j]], rules = rules, orientCollider = FALSE, sepset = list())
        has_rule4_change <- newRule4(new_graph, length(colnames(new_graph)))

        if (length(has_rule4_change) == 2) {
          done_flags[j] <- TRUE
          listAllHt[[length(listAllHt)+1]] <- has_rule4_change[[1]]
          listAllHt[[length(listAllHt)+1]] <- has_rule4_change[[2]]
          done_flags[length(done_flags)+1] <- FALSE
          done_flags[length(done_flags)+1] <- FALSE
        }
        else {
          graphs_done[[length(graphs_done)+1]] <- new_graph
          done_flags[j] <- TRUE
        }
      }
    }
    G_PAG <- c(G_PAG, graphs_done)
  }
  return(G_PAG)
}

#validatePossPags <- function(G_PAG, G_PAG_List, sepsetList, suffStat, IP){
validatePossPags <- function(G_PAG, sepsetList, suffStat, IP){
  violates_list <- c()
  for (J in G_PAG) {

    #(i)

    violates <- FALSE
    # Here we check whether the PAG is valid by checking whether
    # the canonical MAG is ancestral
    validPAG <- isValidPAG(J, verbose=FALSE) # returns a boolean
    if (! validPAG) {
      violates <- TRUE
    }

    #(ii)

    if(!violates) {

      verbose = TRUE

      for (n in 1:length(sepsetList)) {
        for(i in 1:length(sepsetList[n])) {
          for(j in 1:length(sepsetList[[n]][[i]])) {
            labels <- suffStat$citestResultsList[[n]]$labels
            Sij <- sepsetList[[n]][[i]][[j]]

            if(!is.null(Sij) & i!=j){
              xname <- labels[i]
              yname <- labels[j]
              snames <- labels[unlist(Sij)]

              if (verbose) {
                cat(paste("Checking if {", paste0(labels[unlist(Sij)], collapse={","}),
                          "} m-separates", labels[i], "and", labels[j],"\n"))
              }
              msep <- isMSeparated(J, xname, yname, snames, verbose=verbose)
              violates <- violates || !msep # because we want the m-separation
            }
          }
        }
      }

      if(!violates) {
        #(iii)
        # IP saves the labels
        for(xyS in IP) {

          X <- xyS[1]
          Y <- xyS[2]
          Vi <- xyS[3:length(xyS)]

          nodes <- setdiff(Vi, list(X,Y))
          subsets <- getSubsets(nodes) # this returns list of all possible V'

          for(subset in subsets){
            if(!violates){
              msep <- isMSeparated(J, X, Y, subset,
                                   verbose=verbose)
              violates <- violates || msep # because we want the m-connection
            }
          }
        }
        # if(!violates){
        #   G_PAG_List[[length(G_PAG_List)+1]] <- J
        # }
      }
    }
    violates_list <- c(violates_list, violates)
  }
  #return(G_PAG_List)
  return(violates_list)
}

