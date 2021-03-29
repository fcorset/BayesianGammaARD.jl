rm(list=ls())
library(ggplot2)
library(tidyverse)

##################
##################
### PARAMETRES ###
##################
##################

set.seed(123)

tau <- 0.5 # intervalle inter-inspection
rho <- 0.5 # parametre ARDinf
L <- 3 # seuil pour MP
M <- 4 # seuil pout MC
tps.final <- 18 # fenêtre d'observation du processus

# Simulation d'un processus gamma jusqu'au temps final
pas <- 0.01 # pas de temps pour simuler le processus

alpha <- 1 # paramètre de forme de Gamma a = alpha (t)^beta
beta <- 1 # paramètre de forme  de Gamma
b <- 1   # paramètre d'échelle du Gamma
temps <- seq(from = 0,to = tps.final,by = pas)

id.newcycle <- FALSE # Initialisation du nb de cycle de renouvellement


n <- length(temps)

nb.inspections<- floor(tps.final/tau) # nombre d'inspections max pendant la fenêtre d'observation.
obs <- numeric(nb.inspections) # données observées (pendant les inspections)

##############################
##############################
### SIMULATION DES DONNEES ###
##############################
##############################

j <- 1 # indicateur prochaine inspection
j.newcycle <- 0 # identifier le j où nouveau cycle
nb.cycles <- 1 # compteur de cycles

x <- matrix(nrow=nb.inspections,ncol = n) # processus Gamma simulé, nb.lignes = nb.cycles
x[nb.cycles,1] <- 0     # initialisation du processus Gamma = 0 à t=0
for(i in 2:n){
  x[nb.cycles,i] <- x[nb.cycles,i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta)-temps[i-1]^beta)
}
y <- numeric(n) # processus Gamma maintenu
y[1] <- 0

while (j<=nb.inspections) {
  # boucle sur un cycle de renouvellement
  if (id.newcycle){
    # on resimule un x depuis 0
    x[nb.cycles,1]  <- 0     # initialisation du processus Gamma = 0 à t=0
    for(i in 2:n){
      x[nb.cycles,i]  <- x[nb.cycles,i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta)-temps[i-1]^beta)
    }
    id.newcycle <- FALSE
  }
  
  y[((j-1)*tau/pas+2):(j*(tau/pas)+1)] <- y[(j-1)*tau/pas+1]+ x[nb.cycles,((j-j.newcycle-1)*tau/pas+2):((j-j.newcycle)*(tau/pas)+1)]- x[nb.cycles,(j-j.newcycle-1)*tau/pas+1]
  
  obs[j] <- y[j*tau/pas+1] # état dégradation à l'inspection (t_j^-)
  
  if (obs[j] < L) {
    print("no action")
  } else {
    if (obs[j]<M) {
      # on écrase (ARD infini)
      y[j*tau/pas+1] <- (1-rho)*y[j*tau/pas+1] 
    } else {
      y[j*tau/pas+1]<-0
      id.newcycle <- TRUE
      j.newcycle <- j
      nb.cycles <- nb.cycles+1
    }
  }
  j <- j+1
}

##################
##################
### GRAPHIQUES ###  
##################
##################

DeltaImperf <- ((L <= obs) & (obs < M))
DeltaPerf <- (obs >= M)

plot(temps,x[1,],type="l")
plot(temps,x[2,],type="l")

df <- data.frame(temps,x,y)
mygraph <- ggplot(df,aes(x = temps)) +  
  geom_line(aes(y = y), color = "darkred") +
  geom_line(aes(y = x), color="steelblue", linetype="twodash") +
  geom_abline(slope = 0,intercept = L,color="blue") +
  geom_abline(slope = 0,intercept = M,color="red")

plot(temps,x[1,],col="red",type="l",ylim = c(0,max(x[1,])),xlim = c(0,8))
par(new=T)
plot(temps,x[2,],col="green",type="l",ylim = c(0,max(x[1,])),xlim = c(0,8))
par(new=T)
plot(temps,y,type="l",ylim=c(0,max(x[1,])),xlim = c(0,8))
par(new=T)
clr <- rep(x = "black", length = nb.inspections)
clr[DeltaImperf] <- "blue"
clr[DeltaPerf] <- "red"
plot(tau*(1:nb.inspections), numeric(length = nb.inspections), ylim=c(0,max(x[1,])),
     type="p", xlim = c(0,8), col = clr)
abline(h=L,col="blue")
abline(h=M,col="red")



##################
##################
### ESTIMATION ###
##################
##################

obs <- c(0,obs)
Obs <- list()

idx <- which(DeltaPerf)
idx.start <- c(0,idx)
idx.end <- c(idx,nb.inspections)

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
  rho <- pnorm(parms[4])
  # Pseudo-observations
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
  # Computation of the likelihood
  nb.acc <- nrow(AccrObs)
  res <- 0
  for (i in 1:nb.acc) {
    res = res + dgamma(x = AccrObs$accr[i], shape = alpha*AccrObs$tps2[i]^beta - alpha*AccrObs$tps1[i]^beta, scale = b, log = TRUE)
  }
  return(-res)
}

parms <- c(alpha, beta, b, 0)
optim(par = parms, fn = Log.Lik, method = "SANN")
optim(par = parms, fn = Log.Lik, method = "CG")

