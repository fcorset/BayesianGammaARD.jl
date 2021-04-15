rm(list=ls())
library(ggplot2)
library(tidyverse)

##################
##################
### PARAMETRES ###
##################
##################

set.seed(123)

tau <- 0.1 # intervalle inter-inspection
rho <- 0.5 # parametre ARDinf
L <- 1 # seuil pour MP
M <- 3 # seuil pout MC
tps.final <- 10 # fenêtre d'observation du processus

# Simulation d'un processus gamma jusqu'au temps final
pas <- 0.01 # pas de temps pour simuler le processus

alpha <- 0.8 # paramètre de forme de Gamma a = alpha (t)^beta
beta <- 2 # paramètre de forme  de Gamma
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
  x[nb.cycles,i] <- x[nb.cycles,i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
}
y <- numeric(n) # processus Gamma maintenu
y[1] <- 0

while (j<=nb.inspections) {
  # boucle sur un cycle de renouvellement
  if (id.newcycle){
    # on resimule un x depuis 0
    x[nb.cycles,1]  <- 0     # initialisation du processus Gamma = 0 à t=0
    for(i in 2:n){
      x[nb.cycles,i]  <- x[nb.cycles,i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
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

temps.cycle<-c(0,tau*which(obs>M),tps.final)
indice.temps.cycle <- c(1,tau/pas*(which(obs>M))+1,length(temps))


for (k in 1:nb.cycles){
  plot(temps[indice.temps.cycle[k]:indice.temps.cycle[k+1]],x[k,1:(indice.temps.cycle[k+1]-indice.temps.cycle[k]+1)],col="red",type="l",ylim = c(0,2*M),xlim = c(0,tps.final),xlab="",ylab="")
  par(new=T)
}

plot(temps,y,type="l",ylim=c(0,2*M),xlim = c(0,tps.final),ylab="",xlab="")


par(new=T)
plot(tau*(1:nb.inspections),numeric(length = nb.inspections),ylim=c(0,2*M),type="p",xlim = c(0,tps.final),xlab = "temps",ylab="Dégradation")
abline(h=L,col="blue")
abline(h=M,col="red")
abline(v=tau*(1:nb.inspections), lty =3, col = grey(0.1), lwd = 0.45)


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
# optim(par = parms, fn = Log.Lik, method = "SANN")
# optim(par = parms, fn = Log.Lik, method = "L-BFGS-B", lower = rep(1e-1, 4), upper = c(2,2,2,1))



