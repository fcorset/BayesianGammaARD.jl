###################################################
###################################################
### OPTIMISATION DE LA POLITIQUE DE MAINTENANCE ###
###################################################
###################################################

#-----------------------------#
#--- PREPARATION DU SCRIPT ---#
#-----------------------------#

# Nettoyage de l'environnement 
rm(list = ls())

# Chargement des packages
library(readxl)
library(ggplot2)
library(tidyverse)

# Définition des différents paramètres
MatParam <-  read_excel("../Figures/res.xlsx")

#--------------------------#
#--- FONCTIONS INTERNES ---#
#--------------------------#

# Méthode des trapèzes
ntrap <- function(abs,ord){
  # Cette fonction renvoie un vecteur de taille nc (pour différentes valeurs de y)
  # abscisse abs : vecteur de taille nl
  # ordonnée ord : matrice de taille nc*nl
  # (abs_i+1-abs_i)*(f(abs_i,ord)+f(abs_i,ord)/2
  # (matrix(1,ncol=1,nrow=nc)%*%diff(abs)) = (x_2-x_1 x_3-x_2 ...
  #                                           x_2-x_1 x_3-x_2 ...)
  
  nl<- length(abs)
  if (is.matrix(ord)){
    nc <- nrow(ord)
    res <- rowSums((matrix(1,ncol=1,nrow=nc)%*%diff(abs))*(ord[,-ncol(ord)]+ord[,-1])/2)
  } else {
    nc <- 1
    res <- sum(diff(abs)*(ord[-length(ord)]+ord[-1])/2)
  }
  return(res)
}

# Calcul du coût asymptotique d'une politique de maitenance
MaintOpti <- function(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.2,
                      L = 40,  M = 50, C_I = 1, C_P = 10, C_C = 10, 
                      tps.final = 200, K= 50, nb.sim = 1e5, delta.grid = 0.01){
  # CALCUL DE LA LOI STATIONNAIRE
  level.x <- seq(from = delta.grid, by = delta.grid, to = 2*M)
  n.level.x <- length(level.x)
  
  ind.L <- which(level.x >= L)[1]
  ind.M <- which(level.x == M)
  
  level.y <- seq(from = delta.grid, by = delta.grid, to = 2*M)
  n.level.y <-length(level.y)
  
  mat.trans <- matrix(0, ncol = n.level.y, nrow = n.level.x)
  mat.trans.rho <- matrix(0, ncol = n.level.y, nrow = n.level.x)
  
  for(i in 1:(n.level.x-1)){
    mat.trans[i,(i+1):n.level.y] <- dgamma(x = level.y[(i+1):n.level.y]-level.x[i], scale = b,
                                           shape = alpha*tau^beta)
    mat.trans.rho[i,(i+1):n.level.y] <- dgamma(x = level.y[(i+1):n.level.y]-(1-rho)*level.x[i],
                                               scale = b,shape = alpha*tau^beta)
  }
  
  w <- matrix(0, nrow = K, ncol = n.level.y)
  w[1,] <- dgamma(x = level.y, scale = b, shape = alpha*tau^beta)
  
  pi <- list()
  pi[[1]] <- function(x) dgamma(x = ,scale = b, shape = alpha*tau^beta)

  cat("Calcul de la loi stationnaire (", K, " itérations) : ", 1, sep = "")
  for(k in 2:K){
    cat(" -", k)
    # Contribution à la première intégrale :
    Q1 <- ntrap(level.x[1:ind.L],t(mat.trans[1:ind.L,]) * matrix(rep(w[k-1,1:ind.L],n.level.y),byrow = T,ncol = ind.L)) 
    # Contribution à la deuxième intégrale :
    if(length(ind.M) == 0){
      Q2 <- ntrap(level.x[(ind.L+1):length(level.x)],t(mat.trans.rho[(ind.L+1):length(level.x),]) * matrix(rep(w[k-1,(ind.L+1):length(level.x)],n.level.y), byrow = TRUE, ncol = length(level.x)-ind.L))
    } else {
      Q2 <- ntrap(level.x[(ind.L+1):ind.M],t(mat.trans.rho[(ind.L+1):ind.M,]) * matrix(rep(w[k-1,(ind.L+1):ind.M],n.level.y), byrow = TRUE, ncol = ind.M-ind.L))
    }
    # Contribution à la troisième intégrale :
    if(length(ind.M) == 0){
      Q3 <- 0
    } else {
      Q3 <- dgamma(x = level.y, scale = b, shape = alpha*tau^beta) * ntrap(level.x[(ind.M+1):n.level.y],w[k-1,(ind.M+1):n.level.y])
    }
    
    aux <- w[k,] <- (Q1+Q2+Q3)/ntrap(level.x,Q1+Q2+Q3)
    fn_w <- function(x) {
      res <- splinefun(x = level.x, y = aux)(x)*((0 < x) & (x<=max(level.x)))
      return(res)
    }
    pi[[k]] <- fn_w
  }
  cat("- ENDED \n")
  # curve(pi[[K]](x), from = 0, to = max(level.x), add = TRUE,lwd = 2, col = col,ylab = "Stationary Law")
  
  xPi <- runif(n = nb.sim, min = 0, max = L)
  yPi <- runif(n = nb.sim, min = xPi, max = L)
  int.3 <- L*mean(dgamma(x = yPi-xPi, scale = b, shape = alpha*tau^beta)*pi[[K]](xPi)*(L-xPi))
  # print(paste("la proba que Y soit plus petit que L vaut ",int.3))
  
  xPi.1 <- runif(n = nb.sim, min = 0, max = L)
  yPi.1 <- runif(n = nb.sim, min = L, max = M)
  int.1.1 <- mean(dgamma(x = yPi.1-xPi.1, scale = b, shape = alpha*tau^beta)*pi[[K]](xPi.1)*(L *(M-L)))
  xPi.2 <- runif(n = nb.sim, min = L, max = M)
  yPi.2 <- runif(n = nb.sim, min = xPi.2, max = M)
  int.1.2 <- mean(dgamma(x = yPi.2-xPi.2, scale = b, shape = alpha*tau^beta)*pi[[K]](xPi.2)*(M-xPi.2)*(M-L))
  int.1 <- int.1.1+int.1.2
  # print(paste("la proba que Y soit entre L et M vaut ",int.1))
  
  xPi <- runif(n = nb.sim, min = 0, max = M)
  yPi <- runif(n = nb.sim, min = xPi, max = M)
  
  int.2 <- 1-M*mean(dgamma(x = yPi-xPi, scale = b, shape = alpha*tau^beta)*pi[[K]](xPi)*(M-xPi))
  # print(paste("la proba que Y soit plus grand que M vaut ",int.2))
  # print(paste("la somme des proba de Y vaut ",int.1+int.2+int.3))
  
  cout.moy <- (C_I+C_P*int.1+C_C*int.2)/tau
  res <- cout.moy
  return(res)
}

###################
###################
### MAIN SCRIPT ###
###################
###################

seq.L <-seq(from = 1,to = 45,by = .5)
#seq.L <-seq(from = 5,to = 45,by = 10)
cout.L <-vector(mode = "numeric", length = length(seq.L))

output <- list()
set.seed(123)

NumCas <- 40
for (i in 1:length(seq.L)){
  cat("Itération sur L : ",i, "sur ", length(seq.L),"\n")
  
  # res <- MaintOpti(tau = 10, alpha = 1, beta = 1, b = 1, rho = 0.2,
  #                  L =  seq.L[ii],  M = 50, C_I = 1, C_P = 10, C_C = 10, 
  #                  tps.final = 200, K = K, nb.sim = 1e5, delta.grid = 0.01)
  
  res <- MaintOpti(tau = MatParam$tau[NumCas], alpha = MatParam$alpha[NumCas], 
                   beta = MatParam$beta[NumCas], b = MatParam$b[NumCas], 
                   rho = MatParam$rho[NumCas], L =  seq.L[i],  
                   M = MatParam$M[NumCas], C_I = MatParam$C_I[NumCas], 
                   C_P = MatParam$C_P[NumCas], C_C = MatParam$C_C[NumCas])
  
  cout.L[i] <- res
}

Cout.Maintenance <- data.frame(seq.L, cout.L)
output <- list(Cout.Maintenance, MatParam[NumCas, ])
fic.name <- paste("Results-cas-", NumCas, ".Rd", sep = "")
save(output, file = fic.name)



