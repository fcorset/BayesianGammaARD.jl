# Fonction qui permet de simuler selon les valeurs des paramètres
# 22 septembre 2021

# La fonction simu1 permet de simuler un processus gamma lorsqu'on observe
# uniquement les dégradations avant maintenance. Elle prend en entrées :
# alpha, beta : paramètres de forme du processus Gamma : a = alpha (t)^beta
# b : paramètre d'échelle du processus Gamma
# rho : paramètre de réduction de l'ARDinf
# tau : temps inter-inspection
# L : seuil pour faire une MP
# M : seuil pour remplacer le système
# tpsfinal : fenêtre d'observation du processus [0,tpsfinal]
# pas : pas pour effectuer la simulation

# La fonction renvoie :
# obs : les observations (processus maintenu au temps d'inspection) 
# GammaNM : Les processus Gamma Non maintenus (dans la matrice x)
# GammaM : Le processus Gamma Maintenu

simu1 <- function(alpha=1,beta=1,b=1,rho=0.2,tau=1,L=5,M=10,tps.final=100,pas=0.01){
  id.newcycle <- FALSE # Booléen pour nouveau cycle
  # Création du vecteur temps à chaque pas de simulation
  temps = seq(from = 0,to = tps.final,by = pas)
  n=length(temps) # taille du vecteur temps
  nb.insp.max<- floor(tps.final/tau) # nombre d'inspections max pendant la fenêtre d'observation.
  obs<-numeric(nb.insp.max) # données observées (pendant les inspections, à t_j^-)
  j<-1 # initialisation de l'indicateur de la prochaine inspection
  j.newcycle <- 0 # initialisation de l'identificateur du nouveau cycle
  nb.cycles <- 1 # initialisation du compteur de cycles
  x=matrix(nrow=1,ncol = n) # processus Gamma simulé, nb.lignes = nb.cycles 
  x[1,1] <- 0
  # Simulation du Gamma non maintenu sur la fenêtre d'observation
  for(i in 2:n){
    x[nb.cycles,i] = x[nb.cycles,i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
  }
  y=numeric(n) # processus Gamma maintenu en écrasant certaines valeurs lorsque maintenance
  y.tilde=numeric(n) # processus Gamma maintenu
  
  y[1]<-0
  y.tilde[1]<-0
  while (j<=nb.insp.max) {
    # boucle sur un cycle de renouvellement
    if (id.newcycle){
      # on resimule un x depuis 0 et on ajoute une ligne à la matrice x
      x=rbind(x,matrix(ncol=n))
      x[nb.cycles,1] = 0     # initialisation du processus Gamma = 0 à t=0
      for(i in 2:n){
        x[nb.cycles,i] = x[nb.cycles,i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
      }
      id.newcycle <- FALSE
    }
    
    y.tilde[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[(j-1)*tau/pas+1]+ x[nb.cycles,((j-j.newcycle-1)*tau/pas+2):((j-j.newcycle)*(tau/pas)+1)]- x[nb.cycles,(j-j.newcycle-1)*tau/pas+1]
    
    obs[j]<-y.tilde[j*tau/pas+1] # état dégradation à l'inspection (t_j^-)
    
    if (obs[j] < L) {
      # print("no action")
    } else {
      if (obs[j]<M) {
        y[j*tau/pas+1]<-(1-rho)*y[j*tau/pas+1] # on écrase (ARD infini)
      } else {
        y[j*tau/pas+1]<-0 # On écrase la valeur de y
        id.newcycle <- TRUE
        j.newcycle <- j
        nb.cycles <- nb.cycles+1
      }
    }
    j<-j+1
  }
  
  
  
return(list(obs=obs,GammaNM=x,GammaM=y,nb.cycles=nb.cycles))  
}
