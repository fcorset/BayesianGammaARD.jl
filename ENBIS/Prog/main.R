#### Programme de simulation lorsqu'on observe uniquement la dégradation juste avant l'inspection
#### avec deux effets de la maintenance :
#### ARD inf avec rho in [0,1] avec proba p
#### ARD inf avec rho' pour les réparations néfastes avec proba 1-p



rm(list=ls())
library(ggplot2)
library(tidyverse)

tau <- 1 # intervalle inter-inspection
rho <- 0.2 # parametre ARDinf pour les maintenances efficaces avec proba p
rho_w <- 0.4 # parametre ARDinf pour les maintenances néfastes avec proba 1-p
p <- 0.9 # proba que la maintenance préventive soit efficace


L <- 5 # seuil pour MP
M <- 10 # seuil pout MC
tps.final <- 100 # fenêtre d'observation du processus

id.newcycle <- FALSE # Initialisation du nb de cycle de renouvellement

#set.seed(123)
# Simulation d'un processus gamma jusqu'au temps final
pas = 0.01 # pas de temps pour simuler le processus

alpha = 1 # paramètre de forme de Gamma a = alpha (t)^beta
beta = 1 # paramètre de forme  de Gamma
b=1   # paramètre d'échelle du Gamma
temps = seq(from = 0,to = tps.final,by = pas)

theta <-c(alpha,beta,b,rho,rho_w,p)


n=length(temps)

nb.inspections<- floor(tps.final/tau) # nombre d'inspections max pendant la fenêtre d'observation.

temps.insp<-temps[tau/pas*(1:nb.inspections)+1] # temps des inspections (t_i dans le papier)

obs<-numeric(nb.inspections) # données observées (pendant les inspections, à t_j^-)
vect.b<-rep(1,nb.inspections) # données non observées =1 si PM efficace, 0 si worse
Delta.P <-rep(0,nb.inspections) # Données observées = 1 si renouvellement

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
      u<-runif(1)
      if (u>p) vect.b[j]<-0
      ifelse(u<=p,y[j*tau/pas+1]<-(1-rho)*y[j*tau/pas+1],y[j*tau/pas+1]<-(1+rho_w)*y[j*tau/pas+1])
       # on écrase (ARD infini)
    } else {
      y[j*tau/pas+1]<-0
      id.newcycle <- TRUE
      j.newcycle <- j
      nb.cycles <- nb.cycles+1
      Delta.P[j]<-1
    }
  }
  j<-j+1
}

vect.u <- (obs<M)*(obs>L) # indicatrice u_i


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
vect.b.fact <- factor(vect.b)
mescouleurs <- rainbow(length(levels(vect.b.fact)))
plot(tau*(1:nb.inspections),numeric(length = nb.inspections),ylim=c(0,max(y.tilde)),type="p",xlim = c(0,tps.final),xlab = "temps",ylab="Dégradation",col=mescouleurs[vect.b.fact])
abline(h=L,col="blue")
abline(h=M,col="red")
abline(v=tau*(1:nb.inspections), lty =3, col = grey(0.1), lwd = 0.45)

vect.b.fact <- factor(vect.b)
mescouleurs <- rainbow(length(levels(vect.b.fact)))
plot(1:nb.inspections,obs,type="p",xlim = c(0,tps.final),xlab = "temps",ylab="Dégradation",col=mescouleurs[vect.b.fact])
abline(h=L,col="blue")
abline(h=M,col="red")
abline(v=tau*(1:nb.inspections), lty =3, col = grey(0.1), lwd = 0.45)



#abline(v = tau*(1:nb.inspection))

#####################################
############## Algo EM ##############
#####################################

# initialisation des paramètres
K<-100

hat.theta<-matrix(nrow=K+1,ncol=6)
hat.theta[1,] <- c(2,1.5,1.5,0.5,0.5,0.5) # (alpha,beta,b,rho,rho^\prime,p)

delta <- hat.theta[1,1]*(temps.insp^hat.theta[1,2]-c(0,temps.insp[1:(nb.inspections-1)])^hat.theta[1,2])
p.tilde <- (hat.theta[1,6]*dgamma(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[1,4])*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3]))/(hat.theta[1,6]*dgamma(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[1,4])*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3])+(1-hat.theta[1,6])*dgamma(obs-(1-c(0,Delta.P[(nb.inspections-1)]))*(1+hat.theta[1,5])*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3]))

for (k in 2:(K+1)){
  hat.theta[k,6]<-mean(p.tilde) # mise à jour de p
  # Calcul des delta_i
  num.b<-sum(hat.theta[k-1,1]*(temps.insp^hat.theta[k-1,2]-(c(0,temps.insp[1:(nb.inspections-1)]))^hat.theta[k-1,2]))
  det.b<-sum(p.tilde*(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[k-1,4])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)]))+(1-p.tilde)*(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1+hat.theta[k-1,5])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)])))
  
  hat.theta[k,3]<-num.b/det.b # mise à jour de b
  
  # Pour la mise à jour de rho et rho_w, on utilise optim car uniroot ne marche pas... 
  # En effet, plusieurs valeurs annulent la dérivée...
  
  f.rho <- function(x){
    y.rho <- obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-x[1])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)])
    y.rhow <-obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1+x[2])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)])
    logy.rho<-ifelse(y.rho<=0,-Inf,log(y.rho))
    logy.rhow<-ifelse(y.rhow<=0,-Inf,log(y.rhow))
    return(-hat.theta[k,3]*(sum(p.tilde*y.rho+(1-p.tilde)*y.rhow))+sum(p.tilde*(delta-1)*logy.rho+(1-p.tilde)*(delta-1)*ifelse(p.tilde==1,0,logy.rhow)))
  }
  hat.theta[k,4:5]<-optim(hat.theta[k-1,4:5],f.rho,control=list(fnscale=-1))$par
  
  # Mise à jour de alpha et beta
  
  f.ab <- function(x){
    y.rho  <- obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[k,4])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)])
    y.rhow <- obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1+hat.theta[k,5])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)])
    logy.rho<-ifelse(y.rho<=0,-Inf,log(y.rho))
    logy.rhow<-ifelse(y.rhow<=0,-Inf,log(y.rhow))
    delta.n <- x[1]*(temps.insp^x[2]-c(0,temps.insp[1:(nb.inspections-1)])^x[2])
    return(sum(delta.n*log(hat.theta[k,3])-log(gamma(delta.n))+p.tilde*(delta.n-1)*logy.rho+(1-p.tilde)*(delta.n-1)*ifelse(p.tilde==1,0,logy.rhow)))
  }
  hat.theta[k,1:2]<-optim(hat.theta[k-1,1:2],f.ab,control=list(fnscale=-1))$par
  
  
  
  # Mise à jour des delta.i
  delta <- hat.theta[k,1]*(temps.insp^hat.theta[k,2]-c(0,temps.insp[1:(nb.inspections-1)])^hat.theta[k,2])
  
  # Mise à jour des p.tilde
  p.tilde <- (hat.theta[k,6]*dgamma(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[k,4])*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[k,3]))/(hat.theta[k,6]*dgamma(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[k,4])*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[k,3])+(1-hat.theta[k,6])*dgamma(obs-(1-c(0,Delta.P[(nb.inspections-1)]))*(1+hat.theta[k,5])*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[k,3]))
  
  print(paste("la valeur de alpha à l'étape ",k," est : ",hat.theta[k,1]))
  print(paste("la valeur de beta à l'étape ",k," est : ",hat.theta[k,2]))
  print(paste("la valeur de b à l'étape ",k," est : ",hat.theta[k,3]))
  print(paste("la valeur de rho à l'étape ",k," est : ",hat.theta[k,4]))
  print(paste("la valeur de rho_w à l'étape ",k," est : ",hat.theta[k,5]))
  print(paste("la valeur de p à l'étape ",k," est : ",hat.theta[k,6]))
  print("==============================================================")
}





#####################################
# Calcul de la loi stationnaire
#####################################


## update 20/05/2021
## version discrétisée pour le calcul de la loi stationnaire

# fonction ntrap : renvoie l'intégrale par la méthode des trapèzes
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





# vecteur de l'état x
level.x <- seq(0.01,M+2,0.01)
n.level.x <-length(level.x)

ind.L <- which( level.x == L)
ind.M <- which( level.x == M)


level.y <- seq(0.01,M+2,0.01)
n.level.y <-length(level.y)

mat.trans <- matrix(0,ncol=n.level.y,nrow = n.level.x)
mat.trans.rho <- matrix(0,ncol=n.level.y,nrow = n.level.x)

for(i in 1:(n.level.x-1)){
  mat.trans[i,(i+1):n.level.y]<-dgamma(level.y[(i+1):n.level.y]-level.x[i],scale=b,shape = alpha*tau^beta)
  mat.trans.rho[i,(i+1):n.level.y]<-dgamma(level.y[(i+1):n.level.y]-(1-rho)*level.x[i],scale=b,shape = alpha*tau^beta)
}

K<-100
w<-matrix(0,nrow=K,ncol=n.level.y)
w[1,]<-dgamma(level.y,scale = b,shape=alpha*tau^beta)


pi<-list()
pi[[1]]<-function(x) dgamma(x,scale = b,shape=alpha*tau^beta)
curve(pi[[1]](x), from = 0, to = max(level.x), lwd = 2, col = k)

for(k in 2:K){
  
  # Contribution à la première intégrale :
  
  Q1 <- ntrap(level.x[1:ind.L],t(mat.trans[1:ind.L,]) * matrix(rep(w[k-1,1:ind.L],n.level.y),byrow = T,ncol = ind.L)) 
  # Contribution à la deuxième intégrale :
  
  Q2 <- ntrap(level.x[(ind.L+1):ind.M],t(mat.trans.rho[(ind.L+1):ind.M,]) *matrix(rep(w[k-1,(ind.L+1):ind.M],n.level.y),byrow = T,ncol = ind.M-ind.L) )
  
  # Contribution à la troisième intégrale :
  
  Q3 <- dgamma(level.y,scale=b,shape=alpha*tau^beta) * ntrap(level.x[(ind.M+1):n.level.y],w[k-1,(ind.M+1):n.level.y])
  
  aux <- w[k,] <- (Q1+Q2+Q3)/ntrap(level.x,Q1+Q2+Q3)
  fn_w <- function(x) {
    res <- splinefun(x = level.x, y = aux)(x)*((0 < x) & (x<=max(level.x)))
    return(res)
  }
  pi[[k]] <- fn_w
  curve(pi[[k]](x), from = 0, to = max(level.x), lwd = 2, add = TRUE, col = k,ylab = "w_k")
  print(paste("L'intégrale de la", k,"-ième fonction vaut",integrate(fn_w,0,Inf)$value,sep=" "))
  #  scan()
}
######################
###   fin update   ###
###################### 


# 1ere intégrale \int_0^L \int_L^M f(y-x) pi(dx)

#PI <- Vectorize(function(u) integrate(pi[[K]],0,u)$value)

PI <- Vectorize(function(u,n=10000) {
  if (u>max(level.x)) {
    res<-1
  }
  else {
    #set.seed(123)
    x<-runif(n,0,u)
    res <- mean(pi[[K]](x))*u
  }  
  return(res)
})



# Tirer selon loi stationnaire :

# Tirer un u selon U[0,1]
samplePi.aux <- function() {
  uu<-runif(1)
  uniroot(function(u) PI(u) - uu,c(0,10000))$root
}



ech<-replicate(10000,samplePi.aux())
yPi <- runif(10000,L,M)
mean(dgamma(yPi-ech,scale=b,shape=alpha*tau^beta)*(M-L))*PI(M)


# Ne semble pas fonctionner quand on compare l'histo avec la densité ! 23/06/2021.

yPi <- runif(100000,L,M)
xPi <- runif(100000,0,M)

mean(dgamma(yPi-xPi,scale=b,shape=alpha*tau^beta)*pi[[K]](xPi))*(M-L)*M


# A re

xPi <- runif(100000,0,L)
yPi <- runif(100000,0,M)
1-mean(dgamma(yPi-xPi,scale=b,shape=alpha*tau^beta)*pi[[K]](xPi))*(M)*L

# Peut-être choisir une méthode plus efficace... car tous les x>y ne contribuent pas à l'intégrale


K<- 20 # nb itérations de l'algo de point fixe
grid_abs <- seq(0.01,10,0.01)
nb_abs <- length(grid_abs)
w<-matrix(nrow = K,ncol = nb_abs )

pi<-list()
pi[[1]]<-function(x) dgamma(x,scale = b,shape=alpha*tau^beta)


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










