#rm(list=ls())
library(ggplot2)
library(tidyverse)
library(stringr)
library(dplyr)
#setwd(dir = "/Users/fcorset/Documents/bureau/Papiers/encours/AnalyseSensibilite/degradation/ENBIS/Prog/Sept2022")
source("./fonctions.R")
set.seed(10)
#==================================
#      Paramètres du modèle :
#==================================
 alpha = 2 # paramètre de forme de Gamma a = alpha (t)^beta
 beta = 1.2 # paramètre de forme  de Gamma
 b=2   # paramètre d'échelle du Gamma
 rho <- 0.8 # parametre ARDinf pour les maintenances efficaces avec proba p
 rho_w <- 0.5 # parametre ARDinf pour les maintenances néfastes avec proba 1-p
 p <- 0.9 # proba que la maintenance préventive soit efficace
L <- 5 # seuil pour MP
M <- 10 # seuil pout MC
tau <- 1 # intervalle inter-inspection
 HT <- 50 # fenêtre d'observation du processus
pas <- 0.01 # pas de temps pour simuler le processus

# Vraie valeur du paramètre
theta <-c(alpha,beta,b,rho,rho_w,p)

# Estimation des paramètres
#N<-200 # Nb de simulations

N<-200

hat.theta <- matrix(ncol=6,nrow=N)

for(kk in 1:N){

  print(paste("Numéro simulation : ",kk))

  # Simulation des données
  data <- simuGP(alpha = alpha,beta = beta,b = b,rho = rho,rho_w = rho_w,p = p,HT=HT)$donnees

  #

  # Estimation de alpha, beta et b
  hat.theta[kk,1:3] <- optim(c(1.5,1.5,1.5),EspLogLik.abc,mydata=data,control = list(fnscale=-1))$par

  # Estimation de rho
  # Définition de la borne infNo pour rho
  data1 <- filter(data,ind.i.u==1) # On ne garde que les y_i tq u_{i-1}=0
  lower.rho <- max(1-data1$obs/data1$obs.pre)

  hat.theta[kk,4] <- optim(0.5,EspLogLik.rho,mydata=data,est.alpha=hat.theta[kk,1],est.beta=hat.theta[kk,2],est.b=hat.theta[kk,3],method="Brent",lower=lower.rho,upper=1,control = list(fnscale=-1))$par

  # Algo EM

  hat.theta[kk,5:6] <- AlgoEM(mydata=data,par.init=c(.5,.5),K=50,est.alpha=hat.theta[kk,1],est.beta=hat.theta[kk,2],est.b=hat.theta[kk,3],est.rho=hat.theta[kk,4])$estim


}

txt <- str_remove_all(paste("./Results/Ntrajectoires//Concave/estim_",alpha,"_",beta,"_",b,"_",rho,"_",rho_w,"_",p,"_",HT)," ")

save(hat.theta,file=str_remove_all(paste(txt,".Rdata")," "))
