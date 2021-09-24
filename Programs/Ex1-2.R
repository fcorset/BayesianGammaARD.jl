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
beta <- 1
b <- 1
rho <- 0.2
tau <- 1
L <- 5
M <- 10
tps.final <- 100
pas <- 0.01
parms <- c(alpha, 1, b, rho)

set.seed(123)

B <- 1000

vparms <- NULL

for (i in 1:B) {
  cat("Iteration :", i, "\n")
  D <- simu1(alpha, beta, b, rho, tau, L, M, tps.final, pas)
  opt <- estim1(D, tau, L, M, tps.final, pas, optim.method = "SANN")
  vparms <- rbind(vparms, opt$par)
}

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



