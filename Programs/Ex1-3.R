##############################
### PREPARATION DU SCRIPTS ###
##############################

rm(list = ls())

source("simu1.R")
source("plot1.R")
source("Estim1bis.R")

##################
### PARAMETRES ###
##################

Id.Cas <- 4

alpha <- rep(x = 1, times = 9)
beta <- c(1, 1, 1, 0.75, 0.5, 0.5, 1.2, 1.2, 1.2)
b <- rep(x = 1, times = 9)
rho <- c(0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8)
TabCas <- cbind(alpha, beta, b, rho)

alpha <- TabCas[Id.Cas,1]
beta <- TabCas[Id.Cas,2]
b <- TabCas[Id.Cas,3]
rho <- TabCas[Id.Cas,4]

tau <- 1
L <- 5
M <- 10
tps.final <- 100
pas <- 0.01
parms <- c(alpha, beta, b, rho)

set.seed(123)

B <- 3

# Code avec parallélisation

# library(slurmR)
library(doParallel)
library(foreach)
library(doRNG)


#NbCores <- detectCores()
cl <- makeCluster(16)
registerDoParallel(cl)
getDoParWorkers()

estimation <- function() {
  D <- simu1(alpha, beta, b, rho, tau, L, M, tps.final, pas)
  opt <- estim1(D, tau, L, M, tps.final, pas, optim.method = "SANN", guess = TRUE)
  res <- opt$par
  return(res)
}

vparms <- foreach(w = 1: B , .combine = cbind , .options.RNG =123) %dorng%
  estimation()
vparms = t(as.table(vparms))
stopCluster(cl)

res <- list(parms, vparms)

FicOut <- paste("output-cas", Id.Cas, ".Rd", sep = "")

save(res, file = FicOut)



