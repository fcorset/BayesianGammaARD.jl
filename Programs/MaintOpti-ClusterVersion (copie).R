#############################################################################
#############################################################################
### PROGRAMME D'OPTIMISATION DE LA MAINTENANCE EN FONCTION DE TAU ET DE L ###
#############################################################################
#############################################################################

#############################
### PREPARATION DU SCRIPT ###
#############################

# Nettoyage
rm(list = ls())

# Chargement des packages
library(ggplot2)
library(tidyverse)
library(viridis)

##########################
### FONCTIONS INTERNES ###
##########################

#---  Calcul d'intégrale par la méthode des trapèzes ---#

ntrap <- function(abs, ord) {
  # Cette fonction renvoie un vecteur de taille nc (pour différentes valeurs de y)
  # abscisse abs : vecteur de taille nl
  # ordonnée ord : matrice de taille nc*nl
  # (abs_i+1-abs_i)*(f(abs_i,ord)+f(abs_i,ord)/2
  # (matrix(1,ncol=1,nrow=nc)%*%diff(abs)) = (x_2-x_1 x_3-x_2 ...
  #                                           x_2-x_1 x_3-x_2 ...)
  nl <- length(abs)
  if (is.matrix(ord)) {
    nc <- nrow(ord)
    res <- rowSums((matrix(1,ncol=1,nrow=nc)%*%diff(abs))*(ord[,-ncol(ord)]+ord[,-1])/2)
  } else {
    nc <- 1
    res <- sum(diff(abs)*(ord[-length(ord)]+ord[-1])/2)
  }
  return(res)
}

#--- Calcul de la loi stationnaire ---#

MaintOpti <- function(tau = 1, L = 40, M = 50, tps.final = 200,
                      rho = 0.2, alpha = 1, beta = 1, b = 1, 
                      C_I = 1, C_P = 10, C_C = 50) {
  # Paramètres
  pas.x <- 0.01
  pas.x <- 1
  pas.y <- pas.x
  
  # Vecteur de l'état x
  level.x <- seq(from = pas.x, by = pas.x, to = 2*M)
  n.level.x <-length(level.x)
  
  ind.L <- which(level.x >= L)[1]
  ind.M <- which(level.x == M)
  
  level.y <- seq(from = pas.y, by = pas.y, to = 2*M)
  n.level.y <- length(level.y)
  
  mat.trans <- matrix(data = 0, ncol = n.level.y, nrow = n.level.x)
  mat.trans.rho <- matrix(data = 0, ncol = n.level.y, nrow = n.level.x)
  
  for(i in 1:(n.level.x-1)) {
    mat.trans[i,(i+1):n.level.y] <- dgamma(x = level.y[(i+1):n.level.y]-level.x[i], scale = b,
                                           shape = alpha*tau^beta)
    mat.trans.rho[i,(i+1):n.level.y] <- dgamma(x = level.y[(i+1):n.level.y]-(1-rho)*level.x[i],
                                               scale = b,shape = alpha*tau^beta)
  }
  
  K <- 50
  # K <- 5
  w <- matrix(data = 0, nrow = K, ncol = n.level.y)
  w[1,] <- dgamma(x = level.y, scale = b, shape = alpha*tau^beta)
  
  pi <- list()
  pi[[1]] <- function(x) dgamma(x = x, scale = b, shape = alpha*tau^beta)

  for(k in 2:K) {
    # Contribution à la première intégrale 
    Q1 <- ntrap(level.x[1:ind.L],t(mat.trans[1:ind.L,]) * matrix(rep(w[k-1,1:ind.L],n.level.y), 
                                                                 byrow = TRUE, ncol = ind.L)) 
    # Contribution à la deuxième intégrale
    if (length(ind.M) == 0){
      Q2 <- ntrap(level.x[(ind.L+1):length(level.x)],t(mat.trans.rho[(ind.L+1):length(level.x),]) *
                    matrix(rep(w[k-1,(ind.L+1):length(level.x)],n.level.y),byrow = TRUE, ncol = length(level.x)-ind.L) )
    } else {
      Q2 <- ntrap(level.x[(ind.L+1):ind.M],t(mat.trans.rho[(ind.L+1):ind.M,]) *
                    matrix(rep(w[k-1,(ind.L+1):ind.M],n.level.y), byrow = TRUE, ncol = ind.M-ind.L) )
    }
    # Contribution à la troisième intégrale 
    if (length(ind.M) == 0) {
      Q3 <- 0
    } else {
      Q3 <- dgamma(x = level.y, scale = b, shape = alpha*tau^beta) * 
        ntrap(level.x[(ind.M+1):n.level.y], w[k-1,(ind.M+1):n.level.y])
    }
    
    aux <- w[k,] <- (Q1+Q2+Q3)/ntrap(level.x,Q1+Q2+Q3)
    fn_w <- function(x) {
      res <- splinefun(x = level.x, y = aux)(x)*((0 < x) & (x<=max(level.x)))
      return(res)
    }
    pi[[k]] <- fn_w
  }

  clr <- magma(K)
  y <- pi[[K]](level.x)
  eps <- 1e-2
  idx <- which(y < eps)
  max.level.x <- level.x[1:idx[1]]
  curve(expr = pi[[2]](x), from = 0, to = max(level.x), lwd = 2, col = clr[1], ylab = "Stationary Law")
  for (k in 3:K) {
    curve(expr = pi[[k]](x), from = 0, to = max(level.x), add = TRUE, lwd = 2, col = clr[k])
  }
  
  PI <- Vectorize(function(u, n = 10000) {
    if (u>max(level.x)) {
      res<-1
    }
    else {
      x <- runif(n,0,u)
      res <- mean(pi[[K]](x))*u
    }  
    return(res)
  })

  samplePi.aux <- function() {
    uu <- runif(1)
    res <- uniroot(function(u) PI(u) - uu,c(0,10000))$root
    return(res)
  }
  
  nb.sim <- 1e5
  xPi <- runif(n = nb.sim, min = 0, max = L)
  yPi <- runif(n = nb.sim, min = xPi, max = L)
  int.3 <- L*mean(dgamma(x = yPi-xPi, scale = b, shape = alpha*tau^beta)*pi[[K]](xPi)*(L-xPi))
  #print(paste("la proba que Y soit plus petit que L vaut ",int.3))

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
  
  res <- list(int.1 = int.1, int.2 = int.2, cout.moy = cout.moy, pi = pi)
  return(res)
}

outp <- MaintOpti(tau = 1, L = 1)

# seq.tau <- seq(from = 0.1, by = 0.5, to = 10)
seq.tau <- seq(from = 1, by = 1, to = 3)
#int.1 <-numeric(length(seq.tau))
#int.2 <-numeric(length(seq.tau))
cout.tau <- numeric(length(seq.tau))


for (ii in 1:length(seq.tau)) {
  cat("itération : ",ii, "\n")
 res <- MaintOpti(tau = seq.tau[ii], L = 1.35)
 int.1[ii]<-res$int.1
 int.2[ii]<-res$int.2
 cout.tau[ii]<-res$cout.moy
}

#plot(seq.tau,cout.tau, type = "l", lwd = 3)

#cout.tau.smooth <- loess(cout.tau~seq.tau,span=0.8)

#xfit <- seq(from=min(seq.tau),to=max(seq.tau),by = 0.01)
#yfit1 <- predict(cout.tau.smooth,newdata=xfit)
#lines(x = xfit, y = yfit1, col = "red", lwd = 3)
#idx <- which.min(yfit1)
#cat("Coût optimal : ", yfit1[idx], "\n")
#cat("tau optimal : ", xfit[idx], "\n")




seq.L<-seq(from = 1,to = 45,by = .1)
int.1 <-numeric(length(seq.L))
int.2 <-numeric(length(seq.L))
cout.L <-numeric(length(seq.L))

#set.seed(123)
for(ii in 1:length(seq.L)){
  res <- MaintOpti(tau = 5,L=seq.L[ii],col=ii)
  int.1[ii]<-res$int.1
  int.2[ii]<-res$int.2
  cout.L[ii]<-res$cout.moy
  
  cat("itération : ",ii, "sur ", length(seq.L), " \n")
  
  output <- list(seq.L, cout.L, pi[[K]])
  save(output, file = "Optim_cas_x.Rd")
}




# plot(seq.L, cout.L, type = "l", lwd = 3,xlab="L",ylab="Cost",main="Cost for (C_I,C_P,C_C)=(1,5,10) and tau = 10")
# 
# cout.L.smooth <- loess(cout.L~seq.L,span=0.8)
# 
# xfit <- seq(from=min(seq.L),to=max(seq.L),by = 0.01)
# yfit1 <- predict(cout.L.smooth,newdata=xfit)
# lines(x = xfit, y = yfit1, col = "red", lwd = 3,xlab="L")
# idx <- which.min(yfit1)
# cat("Coût optimal : ", yfit1[idx], "\n")
# cat("L optimal : ", xfit[idx], "\n")
# 
