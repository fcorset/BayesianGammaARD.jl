simuGP <- function(alpha=1,beta=1,b=1,rho=.2,rho_w = 0.2,p=.9,tau=1,HT=100,L=5,M=10,pas=0.01){
  
  # tau   : intervalle inter-inspection
  # rho   : parametre ARDinf pour les maintenances efficaces avec proba p
  # rho_w : parametre ARDinf pour les maintenances néfastes avec proba 1-p
  # p     : proba que la maintenance préventive soit efficace
  # L     : seuil pour MP
  # M     : seuil pout MC
  # HT    : fenêtre d'observation du processus
  # pas   : pas de temps pour simuler le processus
  # alpha : paramètre de forme de Gamma a = alpha (t)^beta
  # beta  : paramètre de forme  de Gamma
  # b     : paramètre de taux du Gamma
  
  # Renvoie une liste 
  # donnees est la data frame des données
  # ProcGamma renvoie la date frame :
  #   * x pour le processus non maintenu
  #   * y pour le processus maintenu
  
  id.newcycle <- FALSE # Initialisation du nb de cycle de renouvellement
  
  #set.seed(123)
  # Simulation d'un processus gamma jusqu'au temps final
  
  temps = seq(from = 0,to = HT,by = pas)
  theta <-c(alpha,beta,b,rho,rho_w,p)
  
  
  n=length(temps)
  
  nb.inspections<- floor(HT/tau) # nombre d'inspections max pendant la fenêtre d'observation.
  
  temps.insp<-temps[tau/pas*(1:nb.inspections)+1] # temps des inspections (t_i dans le papier) sans tenir compte des renouvellements
  
  
  obs<-numeric(nb.inspections) # données observées (pendant les inspections, à t_j^-)
  vect.b<-rep(1,nb.inspections) # données non observées =1 si PM efficace, 0 si worse
  Delta.P <-rep(0,nb.inspections) # Données observées = 1 si renouvellement
  
  j<-1 # indicateur prochaine inspection
  j.newcycle <- 0 # identifier le j où nouveau cycle
  nb.cycles <- 1 # compteur de cycles
  
  x=matrix(nrow=nb.inspections,ncol = n) # processus Gamma simulé, nb.lignes = nb.cycles
  x[nb.cycles,1] = 0     # initialisation du processus Gamma = 0 à t=0
  for(i in 2:n){
    x[nb.cycles,i] = x[nb.cycles,i-1] + rgamma(1,rate = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
  }
  y=numeric(n) # processus Gamma maintenu
  y.tilde=numeric(n) # processus Gamma maintenu
  y[1]<-0
  y.tilde[1]<-0
  
  while (j<=nb.inspections) {
    # boucle sur un cycle de renouvellement
    if (id.newcycle){
      # on resimule un x depuis 0
      x[nb.cycles,1:((j-1)*tau/pas+1)] = 0     # initialisation du processus Gamma = 0 à t=0
      for(i in ((j-1)*tau/pas+2):n){
        x[nb.cycles,i] = x[nb.cycles,i-1] + rgamma(1,rate = b, shape = alpha*(temps[i-(j-1)*tau/pas+1]^beta-temps[i-(j-1)*tau/pas]^beta))
      }
      id.newcycle <- FALSE
    }
    # Changement 21/07 : remise du temps à 0 
    #  y.tilde[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[(j-1)*tau/pas+1]+ x[nb.cycles,((j-j.newcycle-1)*tau/pas+2):((j-j.newcycle)*(tau/pas)+1)]- x[nb.cycles,(j-j.newcycle-1)*tau/pas+1]
    
    y.tilde[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[(j-1)*tau/pas+1]+ x[nb.cycles,((j-1)*tau/pas+2):((j)*(tau/pas)+1)]- x[nb.cycles,(j-1)*tau/pas+1]
    
    obs[j]<-y[j*tau/pas+1] # état dégradation à l'inspection (t_j^-)
    
    if (obs[j] < L) {
      # print("no action")
    } else {
      if (obs[j]<M) {
        u<-runif(1)
        if (u>p){
          vect.b[j]<-0
        }
        ifelse(u<=p,y[j*tau/pas+1]<-(1-rho)*y[j*tau/pas+1],y[j*tau/pas+1]<-(1+rho_w)*y[j*tau/pas+1])
        # on écrase (ARD infini)
      } else {
        y[j*tau/pas+1]<-0
        id.newcycle <- TRUE
        j.newcycle <- j
        if(j<nb.inspections){
          Delta.P[j]<-1
          nb.cycles <- nb.cycles+1
        } # 19/09/22 : Peu importe si la dernière inspection est une CM
        
        #     temps2[(j*tau/pas+1):n] <- temps[(j*tau/pas+1):n]-temps[j*tau/pas+1]
      }
    }
    j<-j+1
  }
  
  vect.u <- (obs<M)*(obs>L) # indicatrice u_i
  
  # On ne définit les p.tildes[i] que lorsque u_{i-1}=1 
  ind.u<-which(vect.u[1:(nb.inspections-1)]==1)
  
  # Indice intervenant dans les sommes (i.e tel que u_{i-1}=1)
  ind.i.u<-rep(0,nb.inspections)
  ind.i.u[ind.u+1] <- 1
  
  # indice i lorsque Delta_{i-1}=1 et i différent de nb.inspections
  ind.Delta<-which(Delta.P[1:(nb.inspections-1)]==1)
  
  ind.i.Delta<-c(1,rep(0,nb.inspections-1))
  ind.i.Delta[ind.Delta+1] <- 1   # Pb si la dernière inspection est > M (15/09/22)
  
  #Définition des y_{t_{i-1}}
  obs.pre<-c(0,obs[1:(nb.inspections-1)])*(1-ind.i.Delta)
  
  
  
  
  
  # Nouveaux temps d'inspection prennant en compte les renouvellements
  tps.insp.wR<-c(tau,rep(0,nb.inspections-1))
  tps.insp.pre.wR<-rep(0,nb.inspections)
  
  
  temps.insp.pre <- c(0,temps.insp[1:(nb.inspections-1)])
  
  
  temps.cycle<-c(0,tau*which(obs>M),HT)
  indice.temps.cycle <- c(1,tau/pas*(which(obs>M))+1,length(temps))
  
  
  for(i in 2:nb.inspections){
    if(obs[i-1]>=M){
      tps.insp.wR[i]<-tau
      tps.insp.pre.wR[i]<-0
    } else {
      tps.insp.wR[i]<-tps.insp.wR[i-1]+tau
      tps.insp.pre.wR[i]<-tps.insp.pre.wR[i-1]+tau
    }
  }
  
  for(i in 2:nb.inspections){
    if(ind.i.Delta[i]==1){
      tps.insp.wR[i]<-tau
      tps.insp.pre.wR[i]<-0
    } else {
      tps.insp.wR[i]<-tps.insp.wR[i-1]+tau
      tps.insp.pre.wR[i]<-tps.insp.pre.wR[i-1]+tau
    }
  }
  
  data <- data.frame(temps.insp,temps.insp.pre,obs,obs.pre,vect.u,vect.b,ind.i.u,Delta.P,ind.i.Delta,tps.insp.wR,tps.insp.pre.wR)
  
  
  return(list(donnees=data,ProcGammaNM=x,ProcGammaM=y,Nbcycles=nb.cycles))
  
  
}

EspLogLik.abc <- function(x,mydata){
  # Cette fonction calcule l'espérance de la logVraisemblance en fonction de alpha
  # beta et b uniquement pour les données où y_i < L, i.e u_{i-1}=0
  # x[1] : alpha
  # x[2] : beta
  # x[3] : b
  
  data0 <- filter(mydata,ind.i.u==0) # On ne garde que les y_i tq u_{i-1}=0
  
  # Calcul des delta_i
  delta <- x[1]*(data0$tps.insp.wR^x[2]-data0$tps.insp.pre.wR^x[2])

  # 
  return(log(x[3])*sum(delta) - sum(log(gamma(delta))) + sum((delta-1)*log(data0$obs-data0$obs.pre)) - x[3]*sum(data0$obs-data0$obs.pre))
}

EspLogLik.rho <- function(x,mydata,est.alpha,est.beta,est.b){
  # Cette fonction calcule l'espérance de la logVraisemblance en fonction de rho
  # uniquement pour u_{i-1}=1 et obs <obs.pre
  # est.alpha : estimation de  alpha
  # est.beta : estimation de beta
  # est.b : estimation de b
  # x : rho
  
  data1 <- filter(mydata,ind.i.u==1) # On ne garde que les y_i tq u_{i-1}=0
  
  data1.rho <- filter(data1,obs<obs.pre)
  # Calcul des delta_i
  delta <- est.alpha*(data1.rho$tps.insp.wR^est.beta-data1.rho$tps.insp.pre.wR^est.beta)
  
  return(sum((delta-1)*log(data1.rho$obs-(1-x)*data1.rho$obs.pre)) - est.b*sum(data1.rho$obs-(1-x)*data1.rho$obs.pre))
}





AlgoEM <- function(mydata,par.init,K=50,est.alpha,est.beta,est.b,est.rho){
  # mydata : data.frame issu de la fonction simuGP
  # est.alpha = estimation de alpha
  # est.beta = estimation de beta
  # est.b = estimation de b
  # est.rho = estimation de rho
  # K : nb itérations de EM
  # par.init : initialisation des paramètres pour rho_w et p.tilde
  
  # Déclaration de la matrice des estimations pour rho_w et p
  est.rhow.p<-matrix(nrow=K+1,ncol=2)
  
  # initialisation de l'Algo EM
  est.rhow.p[1,] <- par.init
  
  data1 <- filter(mydata,ind.i.u==1) # On ne garde que les y_i tq u_{i-1}=1

  # Calcul des delta_i
  delta <- est.alpha*(data1$tps.insp.wR^est.beta-data1$tps.insp.pre.wR^est.beta)
  
  # Calcul des p.tilde
  p.tilde <-rep(1,length(data1$obs))
  
  for(i in 1:length(data1$obs)){
    if(data1$obs[i]<data1$obs.pre[i]){
      p.tilde[i] <- 1
    } 
    else 
      {
        p.tilde[i] <- (est.rhow.p[1,2]*dgamma(data1$obs[i]-(1-est.rho)*data1$obs.pre[i],delta[i],rate=est.b))/(est.rhow.p[1,2]*dgamma(data1$obs[i]-(1-est.rho)*data1$obs.pre[i],delta[i],rate=est.b)+(1-est.rhow.p[1,2])*dgamma(data1$obs[i]-(1+est.rhow.p[1,1])*data1$obs.pre[i],delta[i],rate=est.b))
    }
  }
  
  data1.temp<- data.frame(data1,p.tilde,delta)
  data1.rhow <- filter(data1.temp,p.tilde<1)
  
  for (k in 2:(K+1)){
    print(k)
    
    est.rhow.p[k,2]<-mean(p.tilde) # mise à jour de p
    upper.rhow <- min(data1.rhow$obs/data1.rhow$obs.pre-1)
    
    EspLogLik.rhow <- Vectorize(function(x){
      sum(data1.rhow$p.tilde*(data1.rhow$delta-1)*log(data1.rhow$obs-(1-est.rho)*data1.rhow$obs.pre)+(1-data1.rhow$p.tilde)*(data1.rhow$delta-1)*log(data1.rhow$obs-(1+x)*data1.rhow$obs.pre))-est.b*sum(data1.rhow$p.tilde*(data1.rhow$obs-(1-est.rho)*data1.rhow$obs.pre)+(1-data1.rhow$p.tilde)*(data1.rhow$obs-(1+x)*data1.rhow$obs.pre))
    })
    
    est.rhow.p[k,1] <- ifelse(length(data1.rhow$obs)==0,est.rhow.p[k-1,1],optim(est.rhow.p[k-1,1],EspLogLik.rhow,method="Brent",lower=0,upper=upper.rhow,control = list(fnscale=-1))$par)
    
    # print(paste("Estimation de rhow : ",round(est.rhow.p[k,1],3)))
    # print(paste("Estimation de p : ",round(est.rhow.p[k,2],3)))
    
    # Mise à jour des p.tildes
    for(i in 1:length(data1$obs)){
      if(data1$obs[i]<data1$obs.pre[i]){
        p.tilde[i] <- 1
      } 
      else 
      {
        p.tilde[i] <- (est.rhow.p[k,2]*dgamma(data1$obs[i]-(1-est.rho)*data1$obs.pre[i],delta[i],rate=est.b))/(est.rhow.p[k,2]*dgamma(data1$obs[i]-(1-est.rho)*data1$obs.pre[i],delta[i],rate=est.b)+(1-est.rhow.p[k,2])*dgamma(data1$obs[i]-(1+est.rhow.p[k,1])*data1$obs.pre[i],delta[i],rate=est.b))
      }
    }
    data1.temp<- data.frame(data1,p.tilde,delta)
    data1.rhow <- filter(data1.temp,p.tilde<1)
    
  }
  return(list(estim = est.rhow.p[K+1,1:2],p = p.tilde))
}

plot1traj <- function(mydata,alpha=1,beta=1,b=1,rho=.2,rho_w = 0.2,p=.9,tau=1,HT=50,L=5,M=10,pas=0.01){
  # mydata est la sortie de simuGP
  
  data<-mydata$donnees # dataframe contenant les données
  TrajecNM <- mydata$ProcGammaNM
  TrajecM<- mydata$ProcGammaM
  Nbcycles<- mydata$Nbcycles
  
  temps = seq(from = 0,to = HT,by = pas)
  temps.cycle<-c(0,tau*which(data$obs>M),HT)
  indice.temps.cycle <- c(1,tau/pas*(which(data$obs>M))+1,length(temps))
  nb.cycles <- sum(data$obs>=M)+1
  nb.inspections<- floor(HT/tau) # nombre d'inspections max pendant la fenêtre d'observation.
  
  # Calcul du vecteur s (cf. papier)
  s<-temps[1:indice.temps.cycle[2]]
  for(k in 2:nb.cycles){
    s.aux <- temps[(indice.temps.cycle[k]+1):indice.temps.cycle[k+1]]-temps[indice.temps.cycle[k]]
    s<-c(s,s.aux)
  }
  
  temps.insp.2<-c(0,s[tau/pas*(1:nb.inspections)+1]) # Ajout du 20/09/2022 : A mettre dans la LogL
  
  dif.tps <- diff(temps.insp.2^beta)
  dif.tps[dif.tps<0]<-1  # vecteur de longueur n prenant en compte les renouvellements (23/09/2022)
  
  
  for (k in 1:nb.cycles){
    plot(temps[indice.temps.cycle[k]:indice.temps.cycle[k+1]],TrajecNM[k,indice.temps.cycle[k]:indice.temps.cycle[k+1]],col="red",type="l",ylim = c(0,max(TrajecM)),xlim = c(0,HT),xlab="",ylab="")
    par(new=T)
  }
  
  plot(temps,TrajecM,type="l",ylim=c(0,max(TrajecM)),xlim = c(0,HT),ylab="",xlab="")
  
  
  par(new=T)
  vect.b.fact <- factor(data$vect.b)
  mescouleurs <- rainbow(length(levels(vect.b.fact)))
  plot(tau*(1:nb.inspections),numeric(length = nb.inspections),ylim=c(0,max(TrajecM)),type="p",xlim = c(0,HT),xlab = "time",ylab="Degradation",col=mescouleurs[vect.b.fact])
  abline(h=L,col="blue")
  abline(h=M,col="red")
  abline(v=tau*(1:nb.inspections), lty =3, col = grey(0.1), lwd = 0.45)
  
  vect.b.fact <- factor(data$vect.b)
  mescouleurs <- rainbow(length(levels(vect.b.fact)))
  plot(1:nb.inspections,data$obs,type="p",xlim = c(0,HT),xlab = "time",ylab="Degradation",col=mescouleurs[vect.b.fact])
  abline(h=L,col="blue")
  abline(h=M,col="red")
  abline(v=tau*(1:nb.inspections), lty =3, col = grey(0.1), lwd = 0.45)
}




