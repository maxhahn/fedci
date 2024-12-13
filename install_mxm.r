url <- "https://cran.r-project.org/src/contrib/Archive/MXM/MXM_1.5.5.tar.gz"
pkgFile <- "MXM_1.5.5.tar.gz"
download.file(url = url, destfile = pkgFile)

# From DESCRIPTION file
dependencies = c('methods', 'stats', 'utils', 'survival', 'MASS', 'graphics', 'ordinal', 'nnet', 'quantreg', 'lme4', 'foreach', 'doParallel', 'parallel', 'relations', 'Rfast', 'visNetwork', 'energy', 'geepack', 'knitr', 'dplyr', 'bigmemory', 'coxme', 'Rfast2', 'Hmisc')
install.packages(dependencies)

# Install package
install.packages(pkgs=pkgFile, type="source", repos=NULL)

# Delete package tarball
unlink(pkgFile)
