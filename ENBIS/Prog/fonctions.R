
#######################################################################################################
##########################       simuGP simule une Processus Gamma           ##########################  
#######################################################################################################
simuGP <- function(alpha=1,beta=1,b=1,rho=.2,rhow = 0.2,p=.9,tau=1,HT=100,L=5,M=10,pas=0.01){
  # alpha et beta sont pour le paramètre de forme du processus Gamma \alpha t^\beta
  # b est le paramètre de taux du processus Gamma
  # rho est le paramètre d'efficacité de la PM
  # rhow est le paramètre de nuisance de la PM
  # p est la probabilité qu'un PM soit efficace
  # tau est le temps inter inspection
  # HT est la fenêtre d'observation
  # L est le seuil pour déclencher une PM
  # M est le seuil pour déclencher une CM (supposée AGAN)
  # pas est le pas de temps pour simuler le processus Gamma
  
  # Cette fonction renvoie une liste :
  # df : data frame avec temps.inspections, obs, u, b
  # theta : vraie valeur des paramètres
  # HT : Horizon de temps
  # L = seuil de MP
  # M = seuil de MC (AGAN)
  # tau = temps inter inspections
  # nbCycles = nombre de cycles de renouvellement
  # simuGammaNM : matrice des copies de processus de Gamma non maintenus
  
  id.newcycle <- FALSE # Initialisation du nb de cycle de renouvellement
  
  temps = seq(from = 0,to = HT,by = pas)
  
  n=length(temps)
  
  nb.inspections<- floor(HT/tau) # nombre d'inspections max pendant la fenêtre d'observation.
  
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
      #print("no action")
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
  
  df<-data.frame(temps.insp,obs,vect.u,vect.b,Delta.P)
  
  return(list(df=df,theta=c(alpha,beta,b,rho,rhow,p),nbCycles=nb.cycles,HT=HT,L=L,M=M,tau=tau,simuGammaNM=x))
}

plotTrajectory <- function(df,tau=1,L=5,M=10,HT=100,pas=0.01){
  temps = seq(from = 0,to = HT,by = pas)
  temps.cycle<-c(0,tau*which(df$obs>M),HT)
  indice.temps.cycle <- c(1,tau/pas*(which(df$obs>M))+1,length(temps))
  
  
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
  
}

