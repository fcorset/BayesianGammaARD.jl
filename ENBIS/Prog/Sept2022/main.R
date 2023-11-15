rm(list=ls())
library(ggplot2)
library(tidyverse)
library(stringr)
library(dplyr)
setwd(dir = "/Users/fcorset/Documents/bureau/Papiers/encours/AnalyseSensibilite/degradation/ENBIS/Prog/Sept2022")
source("./fonctions.R")

#==================================
#      Paramètres du modèle :
#==================================
alpha = 1 # paramètre de forme de Gamma a = alpha (t)^beta
beta = 1 # paramètre de forme  de Gamma
b=1   # paramètre d'échelle du Gamma
rho <- 0.8 # parametre ARDinf pour les maintenances efficaces avec proba p
rho_w <- 0.5 # parametre ARDinf pour les maintenances néfastes avec proba 1-p
p <- 0.9 # proba que la maintenance préventive soit efficace
L <- 5 # seuil pour MP
M <- 10 # seuil pout MC
tau <- 1 # intervalle inter-inspection
HT <- 500 # fenêtre d'observation du processus
pas <- 0.01 # pas de temps pour simuler le processus

# Vraie valeur du paramètre
theta <-c(alpha,beta,b,rho,rho_w,p)

# Estimation des paramètres
#N<-200 # Nb de simulations

N<-1

hat.theta <- matrix(ncol=6,nrow=N)

for(kk in 1:N){
  
  print(paste("Numéro simulation : ",kk))
  
  # Simulation des données
  tmp <- simuGP(alpha = alpha,beta = beta,b = b,rho = rho,rho_w = rho_w,p = p,HT=HT)
  data<-tmp$donnees
  TrajecNM <- tmp$ProcGammaNM
  TrajecM<- tmp$ProcGammaM
  Nbcycles<- tmp$Nbcycles
  # 
  
  # Estimation de alpha, beta et b
  hat.theta[kk,1:3] <- optim(c(1,1,1),EspLogLik.abc,mydata=data,control = list(fnscale=-1))$par
  
  # Estimation de rho
  # Définition de la borne inf pour rho
  data1 <- filter(data,ind.i.u==1) # On ne garde que les y_i tq u_{i-1}=1
  lower.rho <- max(1-data1$obs/data1$obs.pre)
  
  hat.theta[kk,4] <- optim(0.5,EspLogLik.rho,mydata=data,est.alpha=hat.theta[kk,1],est.beta=hat.theta[kk,2],est.b=hat.theta[kk,3],method="Brent",lower=lower.rho,upper=1,control = list(fnscale=-1))$par
  
  # Algo EM
  resEM <- AlgoEM(mydata=data,par.init=c(.5,.5),K=50,est.alpha=hat.theta[kk,1],est.beta=hat.theta[kk,2],est.b=hat.theta[kk,3],est.rho=hat.theta[kk,4])
  hat.theta[kk,5:6] <- resEM$estim
  est.p.tilde <- resEM$p
  id.worseM <- data1$temps.insp[which(est.p.tilde<1)]-1
  data1<-data.frame(data1,est.p.tilde)
  plot1traj(tmp,alpha=alpha,beta=beta,b=b,rho=rho,rho_w = rho_w,p=p,tau=tau,HT=HT,L=L,M=M,pas=pas)
}

#txt <- str_remove_all(paste("./Results/1trajectoire/Concave/WorseCase/estim_",alpha,"_",beta,"_",b,"_",rho,"_",rho_w,"_",p,"_",HT)," ")
txt <- str_remove_all(paste("./Results/1trajectoire/Concave/WorseCase/estim_",alpha,"_",beta,"_",b,"_",rho,"_",rho_w,"_",p,"_50")," ")

save(hat.theta,file=str_remove_all(paste(txt,".Rdata")," "))
save(tmp,file="./Results/1trajectoire/Concave/WorseCase/donnees_50.Rdata")
save(data1,file="./Results/1trajectoire/Concave/WorseCase/data1_50.Rdata")
save(id.worseM,file="./Results/1trajectoire/Concave/WorseCase/idWM_50.Rdata")
