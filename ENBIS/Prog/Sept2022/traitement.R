rm(list=ls())
library(ggplot2)
library(tidyverse)
library(stringr)
library(dplyr)
# Charger le fichier de données
# *.Rdata
alpha <- 2
beta <- 1.2
b <- 2
rho <- 0.8
rho_w <- 0.1
p <- 0.9
HT <- 50
theta<-c(alpha,beta,b,rho,rho_w,p)
txt <- str_remove_all(paste("./Results/Ntrajectoires/Convexe/estim_",alpha,"_",beta,"_",b,"_",rho,"_",rho_w,"_",p,"_",HT)," ")
load(str_remove_all(paste(txt,".Rdata")," "))




# Donner les vraies valeurs des paramètres
label.theta <-c("alpha","beta","b","rho","rho_w","p")
#theta <- c(alpha,beta,b,rho,rho_w,p)

# Histogramme pour alpha
hist(hat.theta[,1],freq=F,nclass=20,xlab="",main=expression(paste("Histogram of ",alpha," (for ",alpha, "=2, ",beta,"=1.2, ",b,"=2, ",rho,"=0.8, ",rho[w],"=0.1, ",p,"=0.9, ",n,"=50)")))
abline(v=theta[1],col="green")
abline(v=mean(hat.theta[,1]),col="red")

# Histogramme pour beta
hist(hat.theta[,2],freq=F,nclass=20,xlab="",main=expression(paste("Histogram of ",beta," (for ",alpha, "=2, ",beta,"=1.2, ",b,"=2, ",rho,"=0.8, ",rho[w],"=0.1, ",p,"=0.9, ",n,"=50)")))
abline(v=theta[2],col="green")
abline(v=mean(hat.theta[,2]),col="red")


# Histogramme pour b
hist(hat.theta[,3],freq=F,nclass=20,xlab="",main=expression(paste("Histogram of ",b," (for ",alpha, "=2, ",beta,"=1.2, ",b,"=2, ",rho,"=0.8, ",rho[w],"=0.1, ",p,"=0.9, ",n,"=50)")))
abline(v=theta[3],col="green")
abline(v=mean(hat.theta[,3]),col="red")


# Histogramme pour rho
hist(hat.theta[,4],freq=F,nclass=20,xlab="",main=expression(paste("Histogram of ",rho," (for ",alpha, "=2, ",beta,"=1.2, ",b,"=2, ",rho,"=0.8, ",rho[w],"=0.1, ",p,"=0.9, ",n,"=50)")))
abline(v=theta[4],col="green")
abline(v=mean(hat.theta[,4]),col="red")


# histogramme pour rho_w
hist(hat.theta[,5],freq=F,nclass=20,xlab="",main=expression(paste("Histogram of ",rho[w]," (for ",alpha, "=2, ",beta,"=1.2, ",b,"=2, ",rho,"=0.8, ",rho[w],"=0.1, ",p,"=0.9, ",n,"=50)")))
abline(v=theta[5],col="green")
abline(v=mean(hat.theta[,5]),col="red")


# Histogramme pour p
hist(hat.theta[,6],freq=F,nclass=20,xlab="",main=expression(paste("Histogram of ",p," (for ",alpha, "=2, ",beta,"=1.2, ",b,"=2, ",rho,"=0.8, ",rho[w],"=0.1, ",p,"=0.9, ",n,"=50)")))
abline(v=theta[6],col="green")
abline(v=mean(hat.theta[,6]),col="red")






#for(k in 1:6){
#  hist(hat.theta[,k],freq=F,nclass=20,xlab="",main=expression(paste("Histogram of",label.theta[k]," for ",alpha, "=1, ",beta,"=1, ",b,"=1")))
#  abline(v=theta[k],col="green")
#  abline(v=mean(hat.theta[,k]),col="red")
#}

# Traitement pour une trajectoire 

temps = seq(from = 0,to = HT,by = 0.01)


temps.cycle<-c(0,tau*which(data$obs>M),HT)
indice.temps.cycle <- c(1,tau/pas*(which(data$obs>M))+1,length(temps))
nb.cycles <- sum(data$obs>=M)+1

# Calcul du vecteur s (cf. papier)
s<-temps[1:indice.temps.cycle[2]]
for(k in 2:nb.cycles){
  s.aux <- temps[(indice.temps.cycle[k]+1):indice.temps.cycle[k+1]]-temps[indice.temps.cycle[k]]
  s<-c(s,s.aux)
}
nb.inspections<- floor(HT/tau) 
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




