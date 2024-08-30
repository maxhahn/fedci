library(MXM)

independence_test <- function(data, x, y, z = NULL) {
  # Ensure x and y are column names or indices
  if (is.character(x)) {
    xIndex <- which(colnames(data) == x)
  } else {
    xIndex <- x
  }
  
  if (is.character(y)) {
    yIndex <- which(colnames(data) == y)
  } else {
    yIndex <- y
  }
  
  # Handle conditioning set Z
  if (!is.null(z)) {
    if (is.character(z)) {
      csIndex <- which(colnames(data) %in% z)
    } else {
      csIndex <- z
    }
  } else {
    csIndex <- NULL
  }
  
  # Determine variable types
  var_types <- sapply(data, class)
  
  # Choose appropriate test based on variable types
  if (var_types[xIndex] %in% c("numeric") && 
      var_types[yIndex] %in% c("numeric")) {
    test <- "testIndFisher"
  } else if (var_types[yIndex] %in% c("numeric")) {
    test <- "testIndMMReg"
  
  #} else if (var_types[xIndex] %in% c("factor", "ordered", "character") || 
  #           var_types[yIndex] %in% c("factor", "ordered", "character")) {
  #  test <- "testIndMultinom" #"testIndLogistic"
  } else if (var_types[yIndex] %in% c("factor", "ordered", "character")) {
    if (length(unique(data[[y]])) == 2) {
      data[[y]] = data[[y]] == 2
      test <- "testIndLogistic"
    } else {
      test <- "testIndMultinom" #"testIndLogistic"
    }
    
  } else if (var_types[yIndex] %in% c("integer")) {
    test <- "testIndMultinom" #"testIndLogistic"
  } else {
    test <- "testIndSpearman"
  }
  
  # Perform the independence test
  result <- do.call(test, list(target = data[, yIndex], 
                               dataset = data, 
                               xIndex = xIndex, 
                               csIndex = csIndex))
  
  # Return the p-value
  return(exp(result$pvalue))
}