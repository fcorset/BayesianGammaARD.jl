#### Programme de simulation lorsqu'on observe uniquement la dégradation juste avant l'inspection
#### avec deux effets de la maintenance :
#### ARD inf avec rho in [0,1] avec proba p
#### ARD inf avec rho' pour les réparations néfastes avec proba 1-p



rm(list=ls())
library(ggplot2)
library(tidyverse)

tau <- 1 # intervalle inter-inspection
rho <- 0.6 # parametre ARDinf pour les maintenances efficaces avec proba p
rho_w <- 0.8 # parametre ARDinf pour les maintenances néfastes avec proba 1-p
p <- 0.8 # proba que la maintenance préventive soit efficace


L <- 5 # seuil pour MP
M <- 10 # seuil pout MC
tps.final <- 500 # fenêtre d'observation du processus

id.newcycle <- FALSE # Initialisation du nb de cycle de renouvellement

#set.seed(123)
# Simulation d'un processus gamma jusqu'au temps final
pas = 0.01 # pas de temps pour simuler le processus

alpha = 1.5 # paramètre de forme de Gamma a = alpha (t)^beta
beta = 1.3 # paramètre de forme  de Gamma
b=1.2   # paramètre d'échelle du Gamma
temps = seq(from = 0,to = tps.final,by = pas)
temps2 <- temps

theta <-c(alpha,beta,b,rho,rho_w,p)


n=length(temps)

nb.inspections<- floor(tps.final/tau) # nombre d'inspections max pendant la fenêtre d'observation.

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
  x[nb.cycles,i] = x[nb.cycles,i-1] + rgamma(1,rate = b, shape = alpha*(temps2[i]^beta-temps2[i-1]^beta))
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
    print("no action")
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


#df <- data.frame(temps,x,y)
#mygraph <- ggplot(df,aes(x = temps)) +  
#  geom_line(aes(y = y), color = "darkred") +
#  geom_line(aes(y = x), color="steelblue", linetype="twodash") +
#  geom_abline(slope = 0,intercept = L,color="blue") +
#  geom_abline(slope = 0,intercept = M,color="red")


temps.cycle<-c(0,tau*which(obs>M),tps.final)
indice.temps.cycle <- c(1,tau/pas*(which(obs>M))+1,length(temps))


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
  plot(temps[indice.temps.cycle[k]:indice.temps.cycle[k+1]],x[k,indice.temps.cycle[k]:indice.temps.cycle[k+1]],col="red",type="l",ylim = c(0,max(y.tilde)),xlim = c(0,tps.final),xlab="",ylab="")
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
abline(v=81)




#abline(v = tau*(1:nb.inspection))

#####################################
############## Algo EM ##############
#####################################

# initialisation des paramètres
K<- 50

hat.theta<-matrix(nrow=K+1,ncol=6)
#hat.theta[1,] <- c(1.2,1.1,1.5,0.2,0.2,0.5) # (alpha,beta,b,rho,rho^\prime,p)
hat.theta[1,] <- theta

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

temps.insp.pre <- c(0,temps.insp[1:(nb.inspections-1)])


# Nouveaux temps d'inspection prennant en compte les renouvellements


tps.insp.wR<-c(tau,rep(0,nb.inspections-1))
tps.insp.pre.wR<-rep(0,nb.inspections)
                            
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



data1 <- filter(data,ind.i.u==1) # On ne garde que les y_i tq u_{i-1}=1
data0 <- filter(data,ind.i.u==0)



delta <- hat.theta[1,1]*(tps.insp.wR^hat.theta[1,2]-tps.insp.pre.wR^hat.theta[1,2])

  
#p.tilde <- (hat.theta[1,6]*dgamma(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[1,4])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3]))/(hat.theta[1,6]*dgamma(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[1,4])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3])+(1-hat.theta[1,6])*dgamma(obs-(1-c(0,Delta.P[(nb.inspections-1)]))*(1+hat.theta[1,5])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3]))

# p.tilde uniquement défini lorsque u_{i-1}=1

#p.tilde <- ifelse(data1$obs/data1$obs.pre<=1,1,(hat.theta[1,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[1,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[1,3]))/(hat.theta[1,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[1,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[1,3])+(1-hat.theta[1,6])*dgamma(data$obs[ind.i.u==1]-(1+hat.theta[1,5])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[1,3])))
p.tilde <- ifelse(data1$obs/data1$obs.pre<=1,1,(hat.theta[1,6]*dgamma(data1$obs-(1-hat.theta[1,4])*data1$obs.pre,delta[ind.i.u==1],rate=hat.theta[1,3]))/(hat.theta[1,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[1,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[1,3])+(1-hat.theta[1,6])*dgamma(data$obs[ind.i.u==1]-(1+hat.theta[1,5])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[1,3])))



data.temp<-data.frame(data1,p.tilde)
data1.rhow<-filter(data.temp,p.tilde<1)

# Définition de la borne inf pour rho
lower.rho <- max(1-data1$obs/data1$obs.pre)

# Définition de la borne sup pour rhow en enlevant les cas où p.tilde = 1
upper.rhow <- min(data1.rhow$obs/data1.rhow$obs.pre-1)




#data1.rho<-filter(data1,comp.p.tilde>0)

# forcer les p.tilde = 1 lorsque y_i/y_{i-1}<1 !

# p.tilde[which(data1$obs/data1$obs.pre<=1)] <- 1

for (k in 2:(K+1)){
  hat.theta[k,6]<-mean(p.tilde) # mise à jour de p
  # Calcul des delta_i
#  num.b<-sum(hat.theta[k-1,1]*(temps.insp^hat.theta[k-1,2]-(c(0,temps.insp[1:(nb.inspections-1)]))^hat.theta[k-1,2]))
 
  # Mise à jour du 23/09/2022
  # theta <-c(alpha,beta,b,rho,rho_w,p)
  
  esp.logL <- function(x){
    # x[1] : alpha
    # x[2] : beta
    # x[3] : b
    # x[4] : rho
    # x[5] : rho_w
    part.0 <- sum(log(dgamma(data0$obs-data0$obs.pre,shape =x[1]*(data0$tps.insp.wR^x[2]-data0$tps.insp.pre.wR^x[2]),rate = x[3])))
    part.1 <- sum(p.tilde*log(dgamma(x = data1$obs-(1-x[4])*data1$obs.pre,shape=x[1]*(data1$tps.insp.wR^x[2]-data1$tps.insp.pre.wR^x[2]),rate=x[3])))
    part.1.w <- sum((1-p.tilde)*log(dgamma(x = data1.rhow$obs-(1+x[5])*data1.rhow$obs.pre,shape=x[1]*(data1.rhow$tps.insp.wR^x[2]-data1.rhow$tps.insp.pre.wR^x[2]),rate=x[3])))
    return(part.0+part.1+part.1.w)
  }
    par <- c(hat.theta[k-1,1:5])
    hat.theta[k,1:5] <- optim(par,esp.logL,control = list(fnscale=-1))$par
#   num.b<-sum(hat.theta[k-1,1]*dif.tps)
  
#  det.b.1 <- sum(data$obs[ind.i.u==0]-data$obs.pre[ind.i.u==0])
#  det.b.2 <- sum(p.tilde*(data$obs[ind.i.u==1]-(1-hat.theta[k-1,4])*data$obs.pre[ind.i.u==1])+(1-p.tilde)*(data$obs[ind.i.u==1]-(1+hat.theta[k-1,5])*data$obs.pre[ind.i.u==1]))
 
#  det.b<-det.b.1+det.b.2  
#  hat.theta[k,3]<-num.b/det.b # mise à jour de b
  
  # Pour la mise à jour de rho et rho_w, on utilise optim car uniroot ne marche pas... 
  # En effet, plusieurs valeurs annulent la dérivée...
  
  f.rho <- Vectorize(function(x){
    y.rho <- data$obs[ind.i.u==1] - (1-x)*data$obs.pre[ind.i.u==1]
     
    logy.rho<-ifelse(y.rho<=0,-1e8,log(y.rho))
    
    return(-hat.theta[k,3]*sum(p.tilde*y.rho)+sum(p.tilde*(delta[ind.i.u==1]-1)*logy.rho))
  })
#  hat.theta[k,4]<-optimize(f.rho,c(lower.rho,1),maximum = T)$maximum
  
  f.rhow <- Vectorize(function(x){
    
    # UNE COUILLE ICI 
    # cond <- (p.tilde < 1) & (ind.i.u == 1)
    delta.rhow <- hat.theta[k-1,1]*(data1.rhow$temps.insp^hat.theta[k-1,2]-data1.rhow$temps.insp.pre^hat.theta[k-1,2])
    # 
    y.rhow <-data1.rhow$obs - (1+x)*data1.rhow$obs.pre
    # logy.rhow<-ifelse(y.rhow<=0,-Inf,log(y.rhow))
    logy.rhow<-ifelse(y.rhow<=0,-1e8,log(y.rhow))
    return(-hat.theta[k,3]*sum((1-data1.rhow$p.tilde)*y.rhow)+sum((1-data1.rhow$p.tilde)*(delta.rhow-1)*logy.rhow))
  })
  
  
#  hat.theta[k,5]<-optimize(f.rhow,c(0,upper.rhow),maximum = T)$maximum
  
  # Mise à jour de alpha et beta
  
#  y.u.0 <- data$obs[ind.i.u==0]-(1-Delta.P[ind.i.u==0])*data$obs.pre[ind.i.u==0]
#  y.rho  <- data$obs[ind.i.u==1]-(1-hat.theta[k,4])*data$obs.pre[ind.i.u==1]
#  y.rhow <- data$obs[ind.i.u==1]-(1+hat.theta[k,5])*data$obs.pre[ind.i.u==1]
#  logy.u.O <- ifelse(y.u.0<=0,-1e8,log(y.u.0))
#  logy.rho<-ifelse(y.rho<=0,-1e8,log(y.rho))
#  logy.rhow<-ifelse(y.rhow<=0,-1e8,log(y.rhow))
  
  
 # f.ab <- function(x){
#    delta.u.0 <- x[1]*(data$temps.insp[ind.i.u==0]^x[2]-data$temps.insp.pre[ind.i.u==0]^x[2])
#    delta.u.1 <- x[1]*(data$temps.insp[ind.i.u==1]^x[2]-data$temps.insp.pre[ind.i.u==1]^x[2])
#    delta.n   <- x[1]*(data$temps.insp^x[2]-data$temps.insp.pre^x[2]) # tous les delta_i
#  
#    return(sum(delta.n)*log(hat.theta[k,3])-sum(log(gamma(delta.n)))+sum((delta.u.0-1)*logy.u.O)+sum(p.tilde*(delta.u.1-1)*logy.rho+(1-p.tilde)*(delta.u.1-1)*ifelse(p.tilde==1,0,logy.rhow)))
#  }
#  hat.theta[k,1:2]<-optim(hat.theta[k-1,1:2],f.ab,control=list(fnscale=-1))$par
  
  
  
  # Mise à jour des delta.i
  #delta <- hat.theta[k,1]*(data$temps.insp^hat.theta[k,2]-data$temps.insp.pre^hat.theta[k,2])
  delta <- hat.theta[k,1]*(tps.insp.wR^hat.theta[k,2]-tps.insp.pre.wR^hat.theta[k,2])

  #dif.tps <- diff(temps.insp.2^hat.theta[k,2])
  #dif.tps[dif.tps<0]<-0
  
  #delta <- hat.theta[k,1]*dif.tps
  
  
  # Mise à jour des p.tilde
  #p.tilde <- ifelse(data1$obs/data1$obs.pre<=1,1,(hat.theta[k,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[k,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[k,3]))/(hat.theta[k,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[k,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[k,3])+(1-hat.theta[k,6])*dgamma(data$obs[ind.i.u==1]-(1+hat.theta[k,5])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[k,3])))
  p.tilde <- ifelse(data1$obs/data1$obs.pre<=1,1,(hat.theta[k,6]*dgamma(data1$obs-(1-hat.theta[k,4])*data1$obs.pre,delta[ind.i.u==1],rate=hat.theta[k,3]))/(hat.theta[k,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[k,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[k,3])+(1-hat.theta[k,6])*dgamma(data$obs[ind.i.u==1]-(1+hat.theta[k,5])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[k,3])))
  
  
  data.temp <- data.frame(data1,p.tilde)
  data1.rhow <- filter(data.temp,p.tilde<1)
  upper.rhow <- min(data1.rhow$obs/data1.rhow$obs.pre-1)
  
  
  print(paste("la valeur de alpha à l'étape ",k," est : ",hat.theta[k,1]))
  print(paste("la valeur de beta à l'étape ",k," est : ",hat.theta[k,2]))
  print(paste("la valeur de b à l'étape ",k," est : ",hat.theta[k,3]))
  print(paste("la valeur de rho à l'étape ",k," est : ",hat.theta[k,4]))
  print(paste("la valeur de rho_w à l'étape ",k," est : ",hat.theta[k,5]))
  print(paste("la valeur de p à l'étape ",k," est : ",hat.theta[k,6]))
  print("==============================================================")
}

# Identifier les p.tildes <1
ind.poor.maintenance <- which(data1$vect.b==0)

# Maintenance où yappartient à [L,M]
cbind(temps.insp[ind.u],vect.b[ind.u],1-p.tilde)

# PLOT 
par(mfrow = c(2, 3))
plot(hat.theta[, 1], type = "l", lwd = 2, ylim = c(0,2*alpha),xlab="k",ylab="alpha")
abline(h = alpha, col = "red", lwd = 2)^
plot(hat.theta[, 2], type = "l", lwd = 2, ylim = c(0,2*beta),xlab="k",ylab="beta")
abline(h = beta, col = "red", lwd = 2)
plot(hat.theta[, 3], type = "l", lwd = 2, ylim = c(0,2*b),xlab="k",ylab="b")
abline(h = b, col = "red", lwd = 2)
plot(hat.theta[, 4], type = "l", lwd = 2, ylim = c(0,1),xlab="k",ylab="rho")
abline(h = rho, col = "red", lwd = 2)
plot(hat.theta[, 5], type = "l", lwd = 2, ylim = c(0,1),xlab="k",ylab="rho_w")
abline(h = rho_w, col = "red", lwd = 2)
plot(hat.theta[, 6], type = "l", lwd = 2, ylim = c(0,1),xlab="k",ylab="p")
abline(h = p, col = "red", lwd = 2)




