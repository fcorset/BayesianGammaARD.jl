rm(list=ls())

tau <- 1 # intervalle inter-inspection
rho <- 0.3 # parametre ARDinf pour les maintenances efficaces avec proba p
rho_w <- 0.2 # parametre ARDinf pour les maintenances néfastes avec proba 1-p
p <- 0.8 # proba que la maintenance préventive soit efficace
L <- 5 # seuil pour MP
M <- 10 # seuil pout MC
tps.final <- 200 # fenêtre d'observation du processus
K<- 50 # nb iterations EM

nb.rep <- 1000
Output <- matrix(nrow = nb.rep, ncol = 6)

set.seed(123)

nr <- 1
while (nr <= nb.rep) {
  cat("Itération", nr, "sur", nb.rep, "\n")
  try(source("main2A.R"), silent = TRUE)
  if (!is.na(hat.theta[K+1, 1])) {
    Output[nr, ] <- hat.theta[K+1, ]
    nr <- nr+1
  }
}

# PLOT 
par(mfrow = c(2, 3))
hist(x = Output[, 1], probability = TRUE, main = "", xlab = expression(alpha))
lines(density(x = Output[, 1]), lwd = 2)
abline(v = alpha, col = "red", lwd = 2)
hist(x = Output[, 2], probability = TRUE, main = "", xlab = expression(beta))
lines(density(x = Output[, 2]), lwd = 2)
abline(v = beta, col = "red", lwd = 2)
hist(x = Output[, 3], probability = TRUE, main = "", xlab = expression(b))
lines(density(x = Output[, 3]), lwd = 2)
abline(v = b, col = "red", lwd = 2)
hist(x = Output[, 4], probability = TRUE, main = "", xlab = expression(rho))
lines(density(x = Output[, 4]), lwd = 2)
abline(v = rho, col = "red", lwd = 2)
hist(x = Output[, 5], probability = TRUE, main = "", xlab = expression(rho[w]))
lines(density(x = Output[, 5]), lwd = 2)
abline(v = rho_w, col = "red", lwd = 2)
hist(x = Output[, 6], probability = TRUE, main = "", xlab = expression(p))
lines(density(x = Output[, 6]), lwd = 2)
abline(v = p, col = "red", lwd = 2)

  


