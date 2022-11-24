rm(list=ls())
library(ggplot2)
library(tidyverse)
library(stringr)
library(dplyr)
# Charger le fichier de données
# *.Rdata
alpha <- 1
beta <- 1
b <- 1
rho <- 0.8
rho_w <- 0.5
p <- 0.9
HT <- 500
theta<-c(alpha,beta,b,rho,rho_w,p)
txt <- str_remove_all(paste("./Results/estim_",alpha,"_",beta,"_",b,"_",rho,"_",rho_w,"_",p,"_",HT)," ")
load(str_remove_all(paste(txt,".Rdata")," "))




# Donner les vraies valeurs des paramètres
label.theta <-c("alpha","beta","b","rho","rho_w","p")
#theta <- c(alpha,beta,b,rho,rho_w,p)
for(k in 1:6){
  hist(hat.theta[,k],freq=F,nclass=20,main=paste("Histogram of",label.theta[k]))
  abline(v=theta[k],col="green")
  abline(v=mean(hat.theta[,k]),col="red")
}
