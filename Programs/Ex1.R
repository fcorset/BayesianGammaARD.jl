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

##############################
### SIMULATION DES DONNEES ###
##############################

set.seed(123)
D <- simu1(alpha, beta, b, rho, tau, L, M, tps.final, pas)
plot1(D$obs, D$GammaNM, D$GammaM)

#################################
### ESTIMATION DES PARAMETRES ###
#################################

parms <- c(alpha, 1, b, rho)


res.noguess <- estim1(D, tau, L, M, tps.final, pas, optim.method = "SANN")
print(res.noguess$par)

res.guess <- estim1(D, tau, L, M, tps.final, pas, optim.method = "SANN", guess = TRUE)
print(res.guess$par)


