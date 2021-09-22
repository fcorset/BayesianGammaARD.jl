##############################
### PREPARATION DU SCRIPTS ###
##############################

rm(list = ls())

source("simu1.R")
source("plot1.R")

##################
### PARAMETRES ###
##################

alpha <- 1
beta <- 1
b <- 1
rho <- 0.2
tau <- 1
L <- 5
M <- 10
tps.final <- 100
pas <- 0.01

##############################
### SIMULATION DES DONNEES ###
##############################

set.seed(123)
D <- simu1(alpha, beta, b, rho, tau, L, M, tps.final, pas)
plot1(D$obs, D$GammaNM, D$GammaM)

#################################
### ESTIMATION DES PARAMETRES ###
#################################

tps <- seq(from = 0, by = tau, to = tps.final)
obs <- D$obs

DeltaImperf <- ((L <= obs) & (obs < M))
DeltaPerf <- (obs >= M)

nb.inspections <- floor(tps.final/tau)

obs <- c(0,obs)
Obs <- list()

idx <- which(DeltaPerf)
idx.start <- c(0,idx)
idx.end <- c(idx,nb.inspections)

### A REVOIR
nb.cycles <- 2

for (i in 1:nb.cycles) {
  nb <- idx.end[i] - idx.start[i] +1
  aux_tps <- tau*(0:(idx.end[i] - idx.start[i]))
  aux_obs <- obs[1 + idx.start[i]:idx.end[i]]
  aux_obs[1] <- 0
  action <- vector(mode = "logical", length = nb)
  action[aux_obs >= L] <- TRUE 
  action[nb] <- FALSE
  Obs[[i]] <- data.frame(aux_tps, aux_obs, action)
  colnames(Obs[[i]]) <- c("tps", "obs", "action")
}

if (nrow(Obs[[nb.cycles]])==1) {
  Obs <- Obs[-nb.cycles]
  nb.cycles <- nb.cycles - 1
}

AccrObs <- data.frame()
for (i in 1:nb.cycles) {
  nb <- nrow(Obs[[i]])
  tps1 <- Obs[[i]]$tps[1:(nb-1)]
  tps2 <- Obs[[i]]$tps[2:nb]
  accr <- vector(mode = "numeric", length = nb-1)
  for (j in 2:nb) {
    if (!Obs[[i]]$action[j]) {
      accr[j-1] <- Obs[[i]]$obs[j] - Obs[[i]]$obs[j-1]
    } else {
      accr[j-1] <- Obs[[i]]$obs[j] - (1-rho)*Obs[[i]]$obs[j-1]
    }
  }
  aux <- data.frame(tps1, tps2, accr)
  AccrObs <- rbind(AccrObs, aux)
}

Log.Lik <- function(parms) {
  # Parameters
  alpha <- parms[1]
  beta <- parms[2]
  b <- parms[3]
  rho <- parms[4]
  # Pseudo-observations
  AccrObs <- data.frame()
  for (i in 1:nb.cycles) {
    nb <- nrow(Obs[[i]])
    tps1 <- Obs[[i]]$tps[1:(nb-1)]
    tps2 <- Obs[[i]]$tps[2:nb]
    accr <- vector(mode = "numeric", length = nb-1)
    for (j in 2:nb) {
      if (!Obs[[i]]$action[j-1]) {
        accr[j-1] <- Obs[[i]]$obs[j] - Obs[[i]]$obs[j-1]
      } else {
        accr[j-1] <- Obs[[i]]$obs[j] - (1-rho)*Obs[[i]]$obs[j-1]
      }
    }
    aux <- data.frame(tps1, tps2, accr)
    AccrObs <- rbind(AccrObs, aux)
  }
  # Computation of the likelihood
  nb.acc <- nrow(AccrObs)
  res <- vector(mode = "numeric", length = nb.acc)
  for (i in 1:nb.acc) {
    res[i] <- dgamma(x = AccrObs$accr[i], shape = alpha*AccrObs$tps2[i]^beta - alpha*AccrObs$tps1[i]^beta, scale = b, log = TRUE)
  }
  return(-sum(res))
}


Eff <- NULL
for (i in 1:nb.cycles) {
  idx <- which(Obs[[i]]$action)
  if (length(idx)>0) {
    for (j in idx) {
      Eff <- c(Eff, (Obs[[i]]$obs[j] - Obs[[i]]$obs[j+1])/Obs[[i]]$obs[j])
    }    
  }
}
rho.guess <- max(0,max(Eff))

parms <- c(alpha, beta, b, rho)
parms <- c(alpha, 1, b, rho.guess)
optim(par = parms, fn = Log.Lik, method = "SANN")
# optim(par = parms, fn = Log.Lik, method = "L-BFGS-B", lower = rep(1e-1, 4), upper = c(2,2,2,1))



