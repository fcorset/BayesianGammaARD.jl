#### Programme de simulation lorsqu'on observe uniquement la dégradation juste avant l'inspection


rm(list=ls())
library(ggplot2)
library(tidyverse)
tau = 0.2 # intervalle inter-inspection
rho = 0.5 # parametre ARDinf
L=2 # seuil pour MP
M=4 # seuil pout MC
tps.final <-10 # fenêtre d'observation du processus

id.newcycle <- FALSE # Initialisation du nb de cycle de renouvellement

set.seed(123)
# Simulation d'un processus gamma jusqu'au temps final
pas = 0.01 # pas de temps pour simuler le processus

alpha = 1.2 # paramètre de forme de Gamma a = alpha (t)^beta
beta = 1 # paramètre de forme  de Gamma
b=1   # paramètre d'échelle du Gamma
temps = seq(from = 0,to = tps.final,by = pas)

n=length(temps)

nb.inspections<- floor(tps.final/tau) # nombre d'inspections max pendant la fenêtre d'observation.

obs<-numeric(nb.inspections) # données observées (pendant les inspections, à t_j^-)

j<-1 # indicateur prochaine inspection
j.newcycle <- 0 # identifier le j où nouveau cycle
nb.cycles <- 1 # compteur de cycles

x=matrix(nrow=nb.inspections,ncol = n) # processus Gamma simulé, nb.lignes = nb.cycles
x[nb.cycles,1] = 0     # initialisation du processus Gamma = 0 à t=0
for(i in 2:n){
  x[nb.cycles,i] = x[nb.cycles,i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
}
y=numeric(n) # processus Gamma maintenu
y.tilde=numeric(n) # processus Gamma maintenu

y[1]<-0
y.tilde[1]<-0

while (j<=nb.inspections) {
  # boucle sur un cycle de renouvellement
  if (id.newcycle){
    # on resimule un x depuis 0
    x[nb.cycles,1] = 0     # initialisation du processus Gamma = 0 à t=0
    for(i in 2:n){
      x[nb.cycles,i] = x[nb.cycles,i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
    }
    id.newcycle <- FALSE
  }
  
  y.tilde[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[(j-1)*tau/pas+1]+ x[nb.cycles,((j-j.newcycle-1)*tau/pas+2):((j-j.newcycle)*(tau/pas)+1)]- x[nb.cycles,(j-j.newcycle-1)*tau/pas+1]

  obs[j]<-y[j*tau/pas+1] # état dégradation à l'inspection (t_j^-)
  
  if (obs[j] < L) {
    print("no action")
  } else {
    if (obs[j]<M) {
      y[j*tau/pas+1]<-(1-rho)*y[j*tau/pas+1] # on écrase (ARD infini)
    } else {
      y[j*tau/pas+1]<-0
      id.newcycle <- TRUE
      j.newcycle <- j
      nb.cycles <- nb.cycles+1
    }
  }
  j<-j+1
}


#df <- data.frame(temps,x,y)
#mygraph <- ggplot(df,aes(x = temps)) +  
#  geom_line(aes(y = y), color = "darkred") +
#  geom_line(aes(y = x), color="steelblue", linetype="twodash") +
#  geom_abline(slope = 0,intercept = L,color="blue") +
#  geom_abline(slope = 0,intercept = M,color="red")


temps.cycle<-c(0,tau*which(obs>M),tps.final)
indice.temps.cycle <- c(1,tau/pas*(which(obs>M))+1,length(temps))


for (k in 1:nb.cycles){
  plot(temps[indice.temps.cycle[k]:indice.temps.cycle[k+1]],x[k,1:(indice.temps.cycle[k+1]-indice.temps.cycle[k]+1)],col="red",type="l",ylim = c(0,max(y.tilde)),xlim = c(0,tps.final),xlab="",ylab="")
  par(new=T)
}

plot(temps,y.tilde,type="l",ylim=c(0,max(y.tilde)),xlim = c(0,tps.final),ylab="",xlab="")


par(new=T)
plot(tau*(1:nb.inspections),numeric(length = nb.inspections),ylim=c(0,max(y.tilde)),type="p",xlim = c(0,tps.final),xlab = "temps",ylab="Dégradation")
abline(h=L,col="blue")
abline(h=M,col="red")
abline(v=tau*(1:nb.inspections), lty =3, col = grey(0.1), lwd = 0.45)

#abline(v = tau*(1:nb.inspection))


#####################################
# Calcul de la loi stationnaire
#####################################

# se donner une grille pour les abscisses

K<- 20 # nb itérations de l'algo de point fixe
grid_abs <- seq(0.01,10,0.001)
nb_abs <- length(grid_abs)
w<-matrix(nrow = K,ncol = nb_abs )

pi<-list()
pi[[1]]<-function(x) dexp(x)


curve(pi[[1]](x), from = 0, to = max(grid_abs), lwd = 2)

for(k in 2:K){
  for (j in 1:nb_abs){
    abs <- grid_abs[j]
    fn_aux1 <- function(x) {
      res <- pi[[k-1]](x)*dgamma(abs-x,scale = b,shape=alpha*tau^beta)
      return(res)
    }
    fn_aux2 <- function(x) {
      res <- pi[[k-1]](x)*dgamma(abs-(1-rho)*x,scale=b,shape=alpha*tau^beta)
      return(res)
    }
    fn_aux3 <- function(x) {
      res <- pi[[k-1]](x)
      return(res)
    }
    w[k,j] <- integrate(f = fn_aux1, lower = 0, upper = min(L,abs))$value 
    + ifelse(min(abs/(1-rho),L)==min(abs/(1-rho),M),0,integrate(f = fn_aux2, lower = min(abs/(1-rho),L), upper = min(abs/(1-rho),M))$value)
    + integrate(f = fn_aux3, lower = M, upper = Inf)$value * dgamma(abs,scale=b,shape=alpha*tau^beta)
  }
  aux <- w[k,]
  fn_w <- function(x) {
    res <- splinefun(x = grid_abs, y = aux)(x)*((0 < x) & (x<=max(grid_abs)))
    return(res)
  }
  pi[[k]] <- fn_w
  curve(pi[[k]](x), from = 0, to = max(grid_abs), lwd = 2, add = TRUE, col = k)
  print(paste("L'intégrale de la", k,"-ième fonction vaut",integrate(fn_w,0,Inf)$value,sep=" "))
}










