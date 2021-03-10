library(ggplot2)
library(tidyverse)
tau = 0.5 # intervalle inter-inspection
rho = 0.8 # parametre ARD1
L=3 # seuil pour MP
M=6 # seuil pout MC
tps.final <-8 # fenêtre d'observation du processus

# Simulation d'un processus gamma
pas = 0.01 # pas de temps pour simuler le processus

temps = seq(from = 0,to = tps.final,by = pas)
alpha = 1 # paramètre de forme de Gamma a = alpha t
beta = 1 # paramètre d'échelle de Gamma
n=length(temps)
x=numeric(n) # processus Gamma simulé
y=numeric(n) # processus Gamma maintenu
x[1] = 0

nb.inspection<- floor(tps.final/tau)

obs<-numeric(nb.inspection)

for(i in 2:n){
  x[i] = x[i-1] + rgamma(1,scale = beta, shape = alpha*(temps[i]^beta)-temps[i-1]^beta)
}

plot(temps,x,type="l")

# avant la première inspection y=x
y[1:(tau/pas)]<-x[1:(tau/pas)]

# Décision à chaque inspection

ifelse(x[tau/pas+1]<L,y[tau/pas+1]<-x[tau/pas+1],ifelse(x[tau/pas+1]<M,y[tau/pas+1]<-(1-rho)*x[tau/pas+1],y[tau/pas+1]<-0))

obs[1]<-y[tau/pas+1]

j<-1 # indicateur inspection
while (obs[j]<M & j<nb.inspection) {
  y[(j*tau/pas+2):((j+1)*tau/pas+1)]<-y[j*tau/pas+1]+x[(j*tau/pas+2):((j+1)*tau/pas+1)]-x[j*tau/pas+1]
  ifelse(y[(j+1)*tau/pas+1]<L,print("no action"),ifelse(y[(j+1)*tau/pas+1]<M,y[(j+1)*tau/pas+1]<-y[j*tau/pas+1]+(1-rho)*(x[(j+1)*tau/pas+1]-x[j*tau/pas+1]),y[(j+1)*tau/pas+1]<-0))
  obs[j+1]<-y[(j+1)*tau/pas+1]
  j<-j+1
}

df <- data.frame(temps,x,y)
mygraph <- ggplot(df,aes(x = temps)) +  
  geom_line(aes(y = y), color = "darkred") +
  geom_line(aes(y = x), color="steelblue", linetype="twodash") +
  geom_abline(slope = 0,intercept = L,color="blue") +
  geom_abline(slope = 0,intercept = M,color="red")



plot(temps,x,col="red",type="l",ylim = c(0,max(x)),xlim = c(0,8))
par(new=T)
plot(temps,y,type="l",ylim=c(0,max(x)),xlim = c(0,8))
par(new=T)
plot(tau*(1:nb.inspection),numeric(length = nb.inspection),ylim=c(0,max(x)),type="p",xlim = c(0,8))
abline(h=L,col="blue")
abline(h=M,col="red")

#abline(v = tau*(1:nb.inspection))


#####################################
# Calcul de la loi stationnaire
#####################################

# se donner une grille pour les abscisses

K<- 20 # nb itérations de l'algo de point fixe
grid_abs <- seq(0.01,8,0.01)
nb_abs <- length(grid_abs)
w<-matrix(nrow = K,ncol = nb_abs )

pi<-list()
pi[[1]]<-function(x) dexp(x)

w[1,]<-pi[[1]](grid_abs)
curve(pi[[1]](x), from = 0, to = max(grid_abs), lwd = 2)

for(k in 2:K){
<<<<<<< ours
  for (j in 1:nb_abs){
    abs <- grid_abs[j]
    fn_aux1 <- function(x) {
      res <- pi[[k-1]](x)*dgamma(abs-x,rate=beta,shape=alpha*tau)
      return(res)
    }
    fn_aux2 <- function(x) {
      res <- pi[[k-1]](x)*dgamma(abs-(1-rho)*x,rate=beta,shape=alpha*tau)
      return(res)
    }
    fn_aux3 <- function(x) {
      res <- pi[[k-1]](x)
      return(res)
    }
    w[k,j] <- integrate(f = fn_aux1, lower = 0, upper = min(L,abs))$value 
      + integrate(f = fn_aux2, lower = L, upper = min(M,abs/(1-rho)))$value
      + integrate(f = fn_aux3, lower = M, upper = Inf)$value * dgamma(abs,rate=beta,shape=alpha*tau)
  }
  aux <- w[k,]
  fn_w <- function(x) {
    res <- splinefun(x = grid_abs, y = aux)(x)*((0 < x) & (x<=max(grid_abs)))
    return(res)
||||||| base
  for (abs in grid_abs){
    w[2,abs]<-integrate(function(x) pi[[k-1]](x)*dgamma(abs-x,rate=beta,shape=alpha*tau),0,min(c(L,abs))) + ifelse(abs<(1-rho)*L,0,integrate(function(x) pi[[k-1]](x)*dgamma(abs-(1-rho)*x,rate=beta,shape=alpha*tau),L,min(c(M,abs/(1-rho)))))
=======
  for (j in 1:nb_abs){
    abs <- grid_abs[j]
    fn_aux1 <- function(x) {
      res <- pi[[k-1]](x)*dgamma(abs-x,rate=beta,shape=alpha*tau)
      return(res)
    }
    fn_aux2 <- function(x) {
      res <- pi[[k-1]](x)*dgamma(abs-(1-rho)*x,rate=beta,shape=alpha*tau)
      return(res)
    }
    fn_aux3 <- function(x) {
      res <- pi[[k-1]](x)
      return(res)
    }
    w[k,j] <- integrate(f = fn_aux1, lower = 0, upper = min(L,abs))$value 
    + integrate(f = fn_aux2, lower = L, upper = min(M,abs/(1-rho)))$value
    + integrate(f = fn_aux3, lower = M, upper = Inf)$value * dgamma(abs,rate=beta,shape=alpha*tau)
  }
  aux <- w[k,]
  fn_w <- function(x) {
    res <- splinefun(x = grid_abs, y = aux)(x)*((0 < x) & (x<=max(grid_abs)))
    return(res)
>>>>>>> theirs
  }
  pi[[k]] <- fn_w
  curve(pi[[k]](x), from = 0, to = max(grid_abs), lwd = 2, add = TRUE, col = k)
}





# initialisation











