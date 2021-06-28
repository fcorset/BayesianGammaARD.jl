# Programme d'optimisation de la maintenance en fonction de tau et de L
rm(list=ls())
library(ggplot2)
library(tidyverse)

# Définition des coûts d'inspections et  de maintenances.
C_I <- 5
C_P <- 50
C_C <- 100

MaintOpti <- function(tau,L=1){
  set.seed(123)
  rho = 0.8 # parametre ARDinf
  M=5 # seuil pour MC et renouvellement
  tps.final <-100 # fenêtre d'observation du processus
  id.newcycle <- FALSE # Initialisation du nb de cycle de renouvellement
  
  #set.seed(123)
  # Simulation d'un processus gamma jusqu'au temps final
  pas = 0.01 # pas de temps pour simuler le processus
  
  alpha = 1 # paramètre de forme de Gamma a = alpha (t)^beta
  beta = 1 # paramètre de forme  de Gamma
  b=1   # paramètre d'échelle du Gamma
  temps = seq(from = 0,to = tps.final,by = pas)
  
  n=length(temps)
  
  nb.inspections<- floor(tps.final/tau) # nombre d'inspections max pendant la fenêtre d'observation.
  
  obs<-numeric(nb.inspections) # données observées (pendant les inspections, à t_j^-)
  
  j<-1 # indicateur prochaine inspection
  j.newcycle <- 0 # identifier le j où nouveau cycle
  nb.cycles <- 1 # compteur de cycles
  
  x=matrix(nrow=nb.inspections+1,ncol = n) # processus Gamma simulé, nb.lignes = nb.cycles
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
      #print("no action")
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
  level.x <- seq(0.01,2*M,0.01)
  n.level.x <-length(level.x)
  
  ind.L <- which( level.x == L)
  ind.M <- which( level.x == M)
  
  
  level.y <- seq(0.01,2*M,0.01)
  n.level.y <-length(level.y)
  
  mat.trans <- matrix(0,ncol=n.level.y,nrow = n.level.x)
  mat.trans.rho <- matrix(0,ncol=n.level.y,nrow = n.level.x)
  
  for(i in 1:(n.level.x-1)){
    mat.trans[i,(i+1):n.level.y]<-dgamma(level.y[(i+1):n.level.y]-level.x[i],scale=b,shape = alpha*tau^beta)
    mat.trans.rho[i,(i+1):n.level.y]<-dgamma(level.y[(i+1):n.level.y]-(1-rho)*level.x[i],scale=b,shape = alpha*tau^beta)
  }
  
  K<-200
  w<-matrix(0,nrow=K,ncol=n.level.y)
  w[1,]<-dgamma(level.y,scale = b,shape=alpha*tau^beta)
  
  
  pi<-list()
  pi[[1]]<-function(x) dgamma(x,scale = b,shape=alpha*tau^beta)
  curve(pi[[1]](x), from = 0, to = max(level.x), lwd = 2, col = k)
  
  for(k in 2:K){
    
    # Contribution à la première intégrale :
    
    Q1 <- ntrap(level.x[1:ind.L],t(mat.trans[1:ind.L,]) * matrix(rep(w[k-1,1:ind.L],n.level.y),byrow = T,ncol = ind.L)) 
    # Contribution à la deuxième intégrale :
    
    if(length(ind.M) == 0){
      Q2 <- ntrap(level.x[(ind.L+1):length(level.x)],t(mat.trans.rho[(ind.L+1):length(level.x),]) *matrix(rep(w[k-1,(ind.L+1):length(level.x)],n.level.y),byrow = T,ncol = length(level.x)-ind.L) )
    } else {
      Q2 <- ntrap(level.x[(ind.L+1):ind.M],t(mat.trans.rho[(ind.L+1):ind.M,]) *matrix(rep(w[k-1,(ind.L+1):ind.M],n.level.y),byrow = T,ncol = ind.M-ind.L) )
    }
    
    
    # Contribution à la troisième intégrale :
    if(length(ind.M) == 0){
      Q3 <- 0
    } else {
      Q3 <- dgamma(level.y,scale=b,shape=alpha*tau^beta) * ntrap(level.x[(ind.M+1):n.level.y],w[k-1,(ind.M+1):n.level.y])
    }
    
    
    
    aux <- w[k,] <- (Q1+Q2+Q3)/ntrap(level.x,Q1+Q2+Q3)
    fn_w <- function(x) {
      res <- splinefun(x = level.x, y = aux)(x)*((0 < x) & (x<=max(level.x)))
      return(res)
    }
    pi[[k]] <- fn_w
    if(k%%10 == 0){
      curve(pi[[k]](x), from = 0, to = max(level.x), lwd = 2, add = TRUE, col = k,ylab = "w_k")
    }
    
    print(paste("L'intégrale de la", k,"-ième fonction vaut",integrate(fn_w,0,Inf)$value,sep=" "))
    #  scan()
  }
  ######################
  ###   fin update   ###
  ###################### 
  curve(pi[[K]](x), from = 0, to = max(level.x), add=T,lwd = 2, col = K)
  
  
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
  
  
  
  #ech<-replicate(10000,samplePi.aux())
  #yPi <- runif(10000,L,M)
  #mean(dgamma(yPi-ech,rate=beta,shape=alpha*tau)*(M-L))*PI(M)
  
  
  # Ne semble pas fonctionner quand on compare l'histo avec la densité ! 23/06/2021.
  
  yPi <- runif(100000,L,M)
  xPi <- runif(100000,0,M)
  
  int.1 <- mean(dgamma(yPi-xPi,scale=b,shape=alpha*tau^beta)*pi[[K]](xPi))*(M-L)*M
  print(paste("la proba que Y soit entre L et M vaut ",int.1))
  xPi <- rexp(100000,1)
  yPi <- runif(100000,0,M)
  int.2 <- 1-mean(dgamma(yPi-xPi,scale=b,shape=alpha*tau^beta)*pi[[K]](xPi)/dexp(xPi))*(M)
  print(paste("la proba que Y soit plus grand que M vaut ",int.2))
  
  res <- (C_I+C_P*int.1+C_C*int.2)/tau
  return(res)
}

#seq.tau<-seq(from = 0.1,to = 1,by = 0.01)
#cout.tau <-numeric(length(seq.tau))
cout.tau<-MaintOpti(0.2)


#for(ii in 1:length(seq.tau)){
#  cout.tau[ii] <- MaintOpti(tau = seq.tau[ii])
#}

#plot(seq.tau,cout.tau)


#seq.L<-seq(from = 0.5,to = 3.5,by = 0.1)
#cout.L <-numeric(length(seq.L))

#for(ii in 1:length(seq.L)){
#  cout.L[ii] <- MaintOpti(tau = 0.5,L=seq.L[ii])
#}

#plot(seq.L,cout.L)

