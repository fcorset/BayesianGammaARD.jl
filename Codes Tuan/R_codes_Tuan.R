##########################################
##########################################
### TRANSLATION OF TUAN'S MATLAB CODES ###
##########################################
##########################################

XEval <- function(x, h) {
  # Return : X and p
  # This function create a degradation vector for evaluate the stationary law
  # x is the threshold vector
  # h is the discritization step (constant)
  # X is the degradation vector containing the thresholds in x
  # p is the position of the thresholds x in degradation vector X
  
  n <- length(x)-1
  X <- seq(from = 0, by = h, to = x[1])
  p <- length(X)
  # x[1] non divisible par h ... on rajoute un "bout de pas"
  if ((x[1]>h) & (X[p[1]]!=x[1])) {
    p[1] <- p[1]+1
    X[p[1]] <- x[1]
  }
  for (i in 2:(n+1)) {
    # car on va jusqu'a L
    if (x[i] >= x[i-1]+h) {
      X <- c(X, seq(from = x[i-1]+h, by = h, to = x[i]))
      p[i] <- length(X)
      # x(i)-x(i-1) non divisible par h ... on rajoute un "bout de pas"
      if (X[p[i]]!=x[i]) {
        p[i] <- p[i]+1
        X[p[i]] <- x[i]
      }
      else if (x[i]>x[i-1]) {
        p[i] <- p[i-1]+1
        X[p[i]] <- x[i]
      } else {
        p[i] <- p[i-1]
      }
    }
  }   
  res <- list(X, p)
  return(res)
}

TransDens <- function(X, a, b, DelT, N) {
  # This function compute the matrix of transition probability density
  # y = NxN matrix having the following form 
  #     | [ | f(0)||     0    ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]     
  #     | [ |  .  ||    f(0)  ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]
  #     | [ |  .  ||     .    ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]
  # y = | [ |  .  ||     .    ||  .  ||    f(0)  ||  .  ||  .  ||  .  ||  .  |]
  #     | [ |  .  ||f(x1i-y1k)||  .  ||f(x1i-y1i)||  .  ||  .  ||  .  ||  .  |]
  #     | [ |f(x1)||     .    ||  .  ||     .    ||  .  ||  .  || f(0)||  .  |]
  #     | [ |  .  ||     .    ||  .  ||f(x1j-y1i)||  .  ||  .  ||  .  || f(0)|]
  
  # horizontal axis = y, vertical axis = x
  # if x < y, then 0, else different than 0
  
  y <- matrix(data = 0, nrow = N, ncol = N)
  for (i in 1:N) {
    y[i,1:i] <- dgamma(x = X[i]-X[1:i], shape = a*DelT, scale = 1/b)
  }
  res <- y
  return(res)
}  

TransDensD <- function(X, a, b, N, Ropt, Qopt, idx) {
  # This function compute the matrix of transition probability density when
  # the waiting time interval D is function of degradation level
  # a, b are parameters of degradation process
  # N is the length of X
  # Ropt, Qopt are reliability and safety thresholds to compute D
  # idx is the index indicating the way to determine D: 
  # * idx=0 => D is based on the system reliability
  # * idx=1 => D is based on the system MRL
  # y = NxN matrix having the following form 
  #    | [ | f(0)||     0    ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]     
  #    | [ |  .  ||    f(0)  ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]
  #    | [ |  .  ||     .    ||  .  ||     0    ||  .  ||  .  ||  .  ||  .  |]
  #y = | [ |  .  ||     .    ||  .  ||    f(0)  ||  .  ||  .  ||  .  ||  .  |]
  #    | [ |  .  ||f(x1i-y1k)||  .  ||f(x1i-y1i)||  .  ||  .  ||  .  ||  .  |]
  #    | [ |f(x1)||     .    ||  .  ||     .    ||  .  ||  .  || f(0)||  .  |]
  #    | [ |  .  ||     .    ||  .  ||f(x1j-y1i)||  .  ||  .  ||  .  || f(0)|]
  # horizontal axis = y, vertical axis = x
  # if x < y, then 0, else different than 0

  # load data to compute the waiting time ?????
  # load DataForStaLawA1p3B1p3L15 % a=1/3, b=1/3, L=15
  
  # vector of possible values of waiting time
  if (idx==0) {
    # reliability-based waiting time
    nX <- length(vX)
    vD <- matrix(data = 0, nrow = 1, ncol = nX)
    for (p in 1:nX) {
      vD[p] <- interp1(x = R[,p], y = u, xi = Ropt, method = "nearest")
    }
  } else {
    # MRL-based waiting time
    vD <- mrl-Qopt
    vD[vD<0] <- 0
  }
  y <- matrix(data = 0, ncol = N, nrow = N)
  for (i in 1:length(X)) {
    # compute the waiting time D
    D <- interp1(x = vX, y = vD, xi = X[1:i], method ="nearest")
    # compute the transition probability density
    y(i,1:i) <- dgamma(x = X[i]-X[1:i], shape = a*D, scale = 1/b)
  }
  res <- y
  return(res)
}

InpDegLevDStra <- function(a, b, L, M, delT, Dopt, Ropt, Qopt, nI, idx) {
  # a,b are parameter of gamma process
  # L is the failure threshold
  # M is the degradation associated with the prediction accuracy
  # delT is the inspection period
  # D is the waiting time before a replacement
  # nI is the number of inspections
  # vXTAnte and vXTPost are vectors of degradation at just before and just after an inspection or a replacement time
  # vXIAnte is vector of degradation levels at just before an inspection
  # This function simulate the degradation level of the system at the inspection
  # and replacement times according to the (\delta,\xi,\lambda) strategy

  # Pre-allocation
  vXTAnte <- matrix(data = 0, nrow = 1, ncol = nI)
  vXTPost <- vXTAnte
  # vXIAnte <- nan*matrix(data = 1, nrow = 1, ncol = nI) ### WHAT IS nan ???
  vXIAnte <- NaN*matrix(data = 1, nrow = 1, ncol = nI) ### WHAT IS nan ???
  vXIAnte[1] <- 0

  # load data to compute the waiting time
  # load DataForStaLawA1p3B1p3L15 % a=1/3, b=1/3, L=15
  
  # vector of possible values of waiting time
  if (idx==0) {
    # reliability-based waiting time
    nX <- length(vX) 
    vD <- matrix(data = 0, nrow = 1, ncol = nX)
    for (p in 1:nX) {
      vD[p] <- interp1(x = R[,p], y = u, xi = Ropt, method = "nearest")
    }
  }
  if (idx==1) {
    # MRL-based waiting time
    vD <- mrl-Qopt
    vD[vD<0] <- 0
  }
  for (k in 2:nI) {
    if (vXTPost[k-1]<M) {
      vXTAnte[k] <- vXTPost[k-1] + rgamma(n = 1, shape = a*delT, rate = 1/b)
      vXIAnte[k] <- vXTAnte[k]
    } else {
      # compute the waiting time D
      if (idx==2) {
        D <- Dopt
      } else {
        D <- interp1(x = vX, y = vD, xi = vXTPost[k-1], method = 'nearest')
      }
      vXTAnte[k] <- vXTPost[k-1] + rgamma(n = 1, shape = a*D, rate = 1/b)
    }
    # if the system fails
    if (vXTAnte[k]>=L) {
      # Replace immediately at Tk
      vXTPost[k] <- 0
    } else if (vXTAnte[k]>=M) {
      # if the RUL prediction is anough accurate
      # decide depending on vXTPost[k-1]
      if (vXTPost[k-1]>=M) {
        # Replace immediately at Tk
        vXTPost(k) <- 0
      } else {
        # Replace D unit latter
        vXTPost[k] <- vXTAnte[k]
      }
    } else {
      # if the RUL prediction is inaccurate
      # state is still as it is
      vXTPos[k] <- vXTAnte[k]        
    }
  }
  vXIAnte <- vXIAnte[is.finite(vXIAnte)]
  res <- list(vXTAnte, vXTPost, vXIAnte)
  return(res)
}










#########################
#########################
### LODADING PACKAGES ###
#########################
#########################

list.of.packages <- c("pracma")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if (length(new.packages)>0) {
  install.packages(new.packages)
}

library(pracma)





