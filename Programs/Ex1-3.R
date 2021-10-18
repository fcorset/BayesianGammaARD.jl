##############################
### PREPARATION DU SCRIPTS ###
##############################

rm(list = ls())

source("simu1.R")
source("plot1.R")
source("Estim1.R")

##################
### PARAMETRES ###
##################

alpha <- 1
beta <- 1.25
b <- 1
rho <- 0.5
tau <- 1
L <- 5
M <- 10
tps.final <- 100
pas <- 0.01
parms <- c(alpha, 1, b, rho)

set.seed(123)

B <- 1000

# Code avec parallélisation

library(doParallel)
library(foreach)
library(doRNG)

NbCores <- detectCores()
cl <- makeCluster(NbCores)
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


#vparms <- log(vparms)


par(mfrow = c(2,2))
hist(x = vparms[,1], probability = TRUE, xlab = "alpha", main = "")
lines(density(vparms[,1]),lwd = 2, col = "red")
abline(v = alpha, lty = 2, col = "blue")
hist(x = vparms[,2], probability = TRUE, xlab = "beta", main = "")
lines(density(vparms[,2]),lwd = 2, col = "red")
abline(v = beta, lty = 2, col = "blue")
hist(x = vparms[,3], probability = TRUE, xlab = "b", main = "")
lines(density(vparms[,3]),lwd = 2, col = "red")
abline(v = b, lty = 2, col = "blue")
hist(x = vparms[,4], probability = TRUE, xlab = "rho", main = "")
lines(density(vparms[,4]),lwd = 2, col = "red")
abline(v = rho, lty = 2, col = "blue")

aux <- vparms

vparms <- apply(X = aux, MARGIN = 2, scale)

par(mfrow = c(2,2))
hist(x = vparms[,1], probability = TRUE, xlab = "alpha", main = "")
lines(density(vparms[,1]),lwd = 2, col = "red")
hist(x = vparms[,2], probability = TRUE, xlab = "beta", main = "")
lines(density(vparms[,2]),lwd = 2, col = "red")
hist(x = vparms[,3], probability = TRUE, xlab = "b", main = "")
lines(density(vparms[,3]),lwd = 2, col = "red")
hist(x = vparms[,4], probability = TRUE, xlab = "rho", main = "")
lines(density(vparms[,4]),lwd = 2, col = "red")



