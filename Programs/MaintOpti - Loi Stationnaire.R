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
library(ggplot2)
library(tidyverse)
library(scales)

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
StationnaryDist <- function(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.2,
                            L = 40,  M = 50, tps.final = 200, K= 50, delta.grid = 0.01){
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
  res <- pi[[K]]
  return(res)
}

###################
###################
### MAIN SCRIPT ###
###################
###################

output <- list()
set.seed(123)

delta.grid <- 0.01
M = 50

# L fixé, tau varie
level.x <- seq(from = delta.grid, by = delta.grid, to = 2*M)

Loi.tau.1 <- StationnaryDist(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  40, M = M, delta.grid = delta.grid)
y.tau.1 <- Loi.tau.1(level.x)
rm(Loi.tau.1)

Loi.tau.2 <- StationnaryDist(tau = 5, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  40, M = M, delta.grid = delta.grid)
y.tau.2 <- Loi.tau.2(level.x)
rm(Loi.tau.2)

Loi.tau.3 <- StationnaryDist(tau = 10, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  40, M = M, delta.grid = delta.grid)
y.tau.3 <- Loi.tau.3(level.x)
rm(Loi.tau.3)

Df <- data.frame(level.x, y.tau.1, y.tau.2, y.tau.3)
Df2 <- Df %>% pivot_longer(cols = 2:4,
                           names_to = "Case",
                           values_to = "Value")

pdf("Stationnay_Law_tau.pdf")

Gris <- grey_pal()(3)
CLR <- c(y.tau.1 = Gris[1], y.tau.2 = Gris[2], y.tau.3 = Gris[3])
g.tau <- ggplot(Df2, aes(x = level.x, y = Value, group = Case, color = Case)) +
  geom_line(size = 1.2) +
  scale_color_manual(values = CLR, name = '', labels = expression(tau == 1, tau == 5, tau == 10))

g.tau + labs(title = "", x = "x", y = expression(w[K](x)))

dev.off()

# tau fixé, L varie
level.x <- seq(from = delta.grid, by = delta.grid, to = 2*M)

Loi.L.1 <- StationnaryDist(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  20, M = M, delta.grid = delta.grid)
y.L.1 <- Loi.L.1(level.x)
rm(Loi.L.1)

Loi.L.2 <- StationnaryDist(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  25, M = M, delta.grid = delta.grid)
y.L.2 <- Loi.L.2(level.x)
rm(Loi.L.2)

Loi.L.3 <- StationnaryDist(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  30, M = M, delta.grid = delta.grid)
y.L.3 <- Loi.L.3(level.x)
rm(Loi.L.3)

Loi.L.4 <- StationnaryDist(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  35, M = M, delta.grid = delta.grid)
y.L.4 <- Loi.L.4(level.x)
rm(Loi.L.4)

Loi.L.5 <- StationnaryDist(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  40, M = M, delta.grid = delta.grid)
y.L.5 <- Loi.L.5(level.x)
rm(Loi.L.5)

Loi.L.6 <- StationnaryDist(tau = 1, alpha = 1, beta = 1, b = 1, rho = 0.8, L =  45, M = M, delta.grid = delta.grid)
y.L.6 <- Loi.L.6(level.x)
rm(Loi.L.6)

Df <- data.frame(level.x, y.L.1, y.L.2, y.L.3, y.L.4, y.L.5, y.L.6)
Df2 <- Df %>% pivot_longer(cols = 2:7, 
                           names_to = "Case", 
                           values_to = "Value")

pdf("Stationnay_Law_L.pdf")

Gris <- grey_pal()(6)
CLR <- c(y.L.1 = Gris[1], y.L.2 = Gris[2], y.L.3 = Gris[3],
         y.L.4 = Gris[4], y.L.5 = Gris[5], y.L.6 = Gris[6])
g.tau <- ggplot(Df2, aes(x = level.x, y = Value, group = Case, color = Case)) +
  geom_line(size = 1.2) +
  scale_color_manual(values = CLR, name = '', labels = expression(L == 20, L == 25, L == 30,
                                                                  L == 35, L == 40, L == 45))

g.tau + labs(title = "", x = "x", y = expression(w[K](x)))

dev.off()

