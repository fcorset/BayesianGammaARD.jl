#### Programme de simulation lorsqu'on observe uniquement la dégradation juste avant l'inspection
#### avec deux effets de la maintenance :
#### ARD inf avec rho in [0,1] avec proba p
#### ARD inf avec rho' pour les réparations néfastes avec proba 1-p

#############################
#############################
### PREPARATION DU SCRIPT ###
#############################
#############################

# Nettoyage
rm(list = ls())

# Chargement des packages
library(ggplot2)
library(tidyverse)

##################
##################
### PARAMETRES ###
##################
##################

# Intervalle inter-inspection
tau <- 1 
# Paramètre ARDinf pour les maintenances efficaces (avec proba p)
rho <- 0.4 
# Paramètre ARDinf pour les maintenances néfastes (avec proba 1-p)
rho_w <- 0.3 
# Proba que la maintenance préventive soit efficace
p <- 0.8 
# Seuil pour MP
L <- 5 
# Seuil pour MC
M <- 10 

# Paramètre de forme de Gamma a = alpha (t)^beta
alpha <- 2
# Paramètre de forme  de Gamma
beta <- 1 
# Paramètre d'échelle du Gamma
b <- 1.5   

# Vecteur des paramètres à estimer
theta <-c(alpha, beta, b, rho, rho_w, p)

###################################################################
###################################################################
### SIMULATION D'UN PROCESSUS JUSQU'A UN HORIZON DE TEMPS DONNE ###
###################################################################
###################################################################

# Fenêtre d'observation du processus
tps.final <- 500 
# Pas de temps pour la simulation
pas <- 0.01 
# Vecteur des instants de simulation
temps <- seq(from = 0,to = tps.final,by = pas)
n <- length(temps)

# Initialisation du nb de cycle de renouvellement
id.newcycle <- FALSE 

# Lombre d'inspections max pendant la fenêtre d'observation
nb.inspections <- floor(tps.final/tau) 

# Temps des inspections (t_i dans le papier)
temps.insp <-temps[tau/pas*(1:nb.inspections)+1] 

# Données observées (pendant les inspections, à t_j^-)
obs <- vector(mode = "numeric", length = nb.inspections)
# Données non observées : 1 si PM efficace, 0 si worse
vect.b <- rep(x = 1, times = nb.inspections) 
# Données observées : 1 si renouvellement
Delta.P <-rep(x = 0, times = nb.inspections) 

# Indicateur prochaine inspection
j <- 1 
# Identifier le j où nouveau cycle
j.newcycle <- 0
# Compteur de cycles
nb.cycles <- 1 

# Processus Gamma simulé : nb.lignes = nb.cycles
x <- matrix(nrow = nb.inspections, ncol = n) 
# Initialisation du processus Gamma = 0 à t = 0
x[nb.cycles,1] <- 0     
for(i in 2:n){
  x[nb.cycles, i] = x[nb.cycles, i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
}
# Processus Gamma maintenu
y <- vector(mode = "numeric", length = n) 
# Processus Gamma maintenu
y.tilde <- vector(mode = "numeric", length = n) 
y[1] <- 0
y.tilde[1] <- 0

while (j <= nb.inspections) {
  # Boucle sur un cycle de renouvellement
  if (id.newcycle){
    # On resimule un x depuis 0
    # Initialisation du processus Gamma = 0 à t=0
    x[nb.cycles,1] = 0     
    for(i in 2:n){
      x[nb.cycles,i] = x[nb.cycles,i-1] + 
        rgamma(n = 1, scale = b, shape = alpha*(temps[i]^beta-temps[i-1]^beta))
    }
    id.newcycle <- FALSE
  }
  
  y.tilde[((j-1)*tau/pas+2):(j*(tau/pas)+1)] <-
    y[((j-1)*tau/pas+2):(j*(tau/pas)+1)]<-y[(j-1)*tau/pas+1] + 
    x[nb.cycles,((j-j.newcycle-1)*tau/pas+2):((j-j.newcycle)*(tau/pas)+1)] - 
    x[nb.cycles,(j-j.newcycle-1)*tau/pas+1]
  
  # Etat de dégradation à l'inspection (t_j^-)
  obs[j] <- y[j*tau/pas+1] 
  
  if (obs[j] < L) {
    print("no action")
  } else {
    if (obs[j]<M) {
      u <- runif(n = 1)
      if (u>p) vect.b[j] <- 0
      ifelse(u<=p, y[j*tau/pas+1] <- (1-rho)*y[j*tau/pas+1], y[j*tau/pas+1] <- (1+rho_w)*y[j*tau/pas+1])
       # on écrase (ARD infini)
    } else {
      y[j*tau/pas+1] <- 0
      id.newcycle <- TRUE
      j.newcycle <- j
      nb.cycles <- nb.cycles+1
      Delta.P[j] <- 1
    }
  }
  j <- j+1
}

# Indicatrice u_i
vect.u <- (obs<M)*(obs>L) 

#df <- data.frame(temps,x,y)
#mygraph <- ggplot(df,aes(x = temps)) +  
#  geom_line(aes(y = y), color = "darkred") +
#  geom_line(aes(y = x), color="steelblue", linetype="twodash") +
#  geom_abline(slope = 0,intercept = L,color="blue") +
#  geom_abline(slope = 0,intercept = M,color="red")

temps.cycle <- c(0, tau*which(obs>M), tps.final)
indice.temps.cycle <- c(1, tau/pas*(which(obs>M))+1, length(temps))

for (k in 1:nb.cycles){
  plot(x = temps[indice.temps.cycle[k]:indice.temps.cycle[k+1]], 
       y = x[k,1:(indice.temps.cycle[k+1]-indice.temps.cycle[k]+1)],
       col = "red", type = "l", ylim = c(0,max(y.tilde)), xlim = c(0,tps.final), xlab = "", ylab = "")
  par(new = TRUE)
}
plot(x = temps, y = y.tilde, type = "l", ylim = c(0,max(y.tilde)), xlim = c(0,tps.final), ylab = "",
     xlab = "")

par(new = TRUE)
vect.b.fact <- factor(vect.b)
mescouleurs <- rainbow(length(levels(vect.b.fact)))
plot(x = tau*(1:nb.inspections), y = numeric(length = nb.inspections), ylim = c(0,max(y.tilde)),
     type = "p", xlim = c(0,tps.final), xlab = "Temps", ylab = "Dégradation",
     col = mescouleurs[vect.b.fact])
abline(h = L, col = "blue")
abline(h = M, col = "red")
abline(v = tau*(1:nb.inspections), lty = 3, col = grey(0.1), lwd = 0.45)

vect.b.fact <- factor(vect.b)
mescouleurs <- rainbow(length(levels(vect.b.fact)))
plot(x = 1:nb.inspections, y = obs, type = "p", xlim = c(0,tps.final), xlab = "Temps",
     ylab = "Dégradation", col = mescouleurs[vect.b.fact])
abline(h = L, col = "blue")
abline(h = M, col = "red")
abline(v = tau*(1:nb.inspections), lty =3, col = grey(0.1), lwd = 0.45)
abline(v = 81)

###########################################
###########################################
### ESTIMATION DES PARAMETRES : ALGO EM ###
###########################################
###########################################

# Nombre d'itérations
K <- 50

# Initialisation des paramètres
hat.theta<-matrix(nrow = K+1, ncol = 6)
#hat.theta[1,] <- c(1.2,1.1,1.5,0.2,0.2,0.5) # (alpha,beta,b,rho,rho^\prime,p)
hat.theta[1,] <- theta

# On ne définit les p.tildes[i] que lorsque u_{i-1}=1 
ind.u <- which(vect.u[1:(nb.inspections-1)] == 1)

# Indice intervenant dans les sommes (i.e tel que u_{i-1}=1)
ind.i.u <- rep(x = 0, times = nb.inspections)
ind.i.u[ind.u+1] <- 1

# Indice i lorsque Delta_{i-1}=1
ind.Delta <- which(Delta.P == 1)

ind.i.Delta <- rep(x = 0, times = nb.inspections)
ind.i.Delta[ind.Delta+1] <- 1

# Définition des y_{t_{i-1}}
obs.pre <- c(0, obs[1:(nb.inspections-1)])*(1-ind.i.Delta)

temps.insp.pre <- c(0, temps.insp[1:(nb.inspections-1)])

data <- data.frame(temps.insp, temps.insp.pre, obs, obs.pre, vect.u, vect.b, ind.i.u, Delta.P, ind.i.Delta)

# On ne garde que les y_i tq u_{i-1}=1
data1 <- filter(data, ind.i.u == 1) 
#data0 <- filter(data,ind.i.u==0)

delta <- hat.theta[1,1]*(data$temps.insp^hat.theta[1,2]-data$temps.insp.pre^hat.theta[1,2])

#p.tilde <- (hat.theta[1,6]*dgamma(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[1,4])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3]))/(hat.theta[1,6]*dgamma(obs-(1-c(0,Delta.P[1:(nb.inspections-1)]))*(1-hat.theta[1,4])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3])+(1-hat.theta[1,6])*dgamma(obs-(1-c(0,Delta.P[(nb.inspections-1)]))*(1+hat.theta[1,5])^(c(0,vect.u[1:(nb.inspections-1)]))*c(0,obs[1:(nb.inspections-1)]),delta,rate=hat.theta[1,3]))
# p.tilde uniquement défini lorsque u_{i-1}=1
p.tilde <- ifelse(data1$obs/data1$obs.pre<=1, 1, 
                  (hat.theta[1,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[1,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[1,3]))/(hat.theta[1,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[1,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[1,3])+(1-hat.theta[1,6])*dgamma(data$obs[ind.i.u==1]-(1+hat.theta[1,5])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[1,3])))

data.temp <- data.frame(data1, p.tilde)
data1.rhow <- filter(data.temp, p.tilde<1)

# Définition de la borne inf pour rho
lower.rho <- max(1-data1$obs/data1$obs.pre)

# Définition de la borne sup pour rhow en enlevant les cas où p.tilde = 1
upper.rhow <- min(data1.rhow$obs/data1.rhow$obs.pre-1)

#data1.rho<-filter(data1,comp.p.tilde>0)

# forcer les p.tilde = 1 lorsque y_i/y_{i-1}<1 !

# p.tilde[which(data1$obs/data1$obs.pre<=1)] <- 1

for (k in 2:(K+1)){
  # Mise à jour de p
  hat.theta[k,6] <- mean(p.tilde)
  # Calcul des delta_i
  # Mise à jour de b
  num.b <- sum(hat.theta[k-1,1]*(temps.insp^hat.theta[k-1,2]-(c(0,temps.insp[1:(nb.inspections-1)]))^hat.theta[k-1,2]))
  det.b.1 <- sum(data$obs[ind.i.u==0]-data$obs.pre[ind.i.u==0])
  det.b.2 <- sum(p.tilde*(data$obs[ind.i.u==1]-(1-hat.theta[k-1,4])*data$obs.pre[ind.i.u==1])+(1-p.tilde)*(data$obs[ind.i.u==1]-(1+hat.theta[k-1,5])*data$obs.pre[ind.i.u==1]))
  det.b <- det.b.1+det.b.2  
  hat.theta[k,3] <- num.b/det.b 
  # Pour la mise à jour de rho et rho_w, on utilise optim car uniroot ne marche pas... 
  # En effet, plusieurs valeurs annulent la dérivée...
  f.rho <- Vectorize(function(x){
    y.rho <- data$obs[ind.i.u==1] - (1-x)*data$obs.pre[ind.i.u==1]
    logy.rho <- ifelse(y.rho<=0,-1e8,log(y.rho))
    res <- -hat.theta[k,3]*sum(p.tilde*y.rho)+sum(p.tilde*(delta[ind.i.u==1]-1)*logy.rho)
    return(res)
  })
  hat.theta[k,4] <- optimize(f = f.rho, interval = c(lower.rho,1), maximum = TRUE)$maximum
  f.rhow <- Vectorize(function(x){
    # UNE COUILLE ICI 
    # cond <- (p.tilde < 1) & (ind.i.u == 1)
    delta.rhow <- hat.theta[k-1,1]*(data1.rhow$temps.insp^hat.theta[k-1,2]-data1.rhow$temps.insp.pre^hat.theta[k-1,2])
    # 
    y.rhow <-data1.rhow$obs - (1+x)*data1.rhow$obs.pre
    # logy.rhow<-ifelse(y.rhow<=0,-Inf,log(y.rhow))
    logy.rhow <- ifelse(y.rhow<=0,- 1e8, log(y.rhow))
    res <- -hat.theta[k,3]*sum((1-data1.rhow$p.tilde)*y.rhow)+sum((1-data1.rhow$p.tilde)*(delta.rhow-1)*logy.rhow)
    return(res)
  })
  
  hat.theta[k,5] <- optimize(f = f.rhow, interval = c(0,upper.rhow), maximum = TRUE)$maximum
  # Mise à jour de alpha et beta
  y.u.0 <- data$obs[ind.i.u==0]-(1-Delta.P[ind.i.u==0])*data$obs.pre[ind.i.u==0]
  y.rho  <- data$obs[ind.i.u==1]-(1-hat.theta[k,4])*data$obs.pre[ind.i.u==1]
  y.rhow <- data$obs[ind.i.u==1]-(1+hat.theta[k,5])*data$obs.pre[ind.i.u==1]
  logy.u.O <- ifelse(y.u.0<=0, -1e8, log(y.u.0))
  logy.rho <-ifelse(y.rho<=0, -1e8, log(y.rho))
  #logy.rhow <- ifelse(y.rhow<=0, -1e8, log(y.rhow))
  logy.rhow <- y.rhow
  idx <- which(y.rhow<=0)
  logy.rhow[idx] <- -1e8
  logy.rhow[-idx] <- -log(y.rhow[-idx])
  
  f.ab <- Vectorize(function(x){
    delta.u.0 <- x[1]*(data$temps.insp[ind.i.u==0]^x[2]-data$temps.insp.pre[ind.i.u==0]^x[2])
    delta.u.1 <- x[1]*(data$temps.insp[ind.i.u==1]^x[2]-data$temps.insp.pre[ind.i.u==1]^x[2])
    delta.n   <- x[1]*(data$temps.insp^x[2]-data$temps.insp.pre^x[2]) # tous les delta_i
    res <- sum(delta.n)*log(hat.theta[k,3])-sum(log(gamma(delta.n)))+sum((delta.u.0-1)*logy.u.O)+sum(p.tilde*(delta.u.1-1)*logy.rho+(1-p.tilde)*(delta.u.1-1)*ifelse(p.tilde==1,0,logy.rhow))
    return(res)
  })
  hat.theta[k,1:2] <- optim(par = hat.theta[k-1,1:2], fn = f.ab, control = list(fnscale=-1))$par
  # Mise à jour des delta.i
  delta <- hat.theta[k,1]*(data$temps.insp^hat.theta[k,2]-data$temps.insp.pre^hat.theta[k,2])
  # Mise à jour des p.tilde
  p.tilde <- ifelse(data1$obs/data1$obs.pre<=1, 1, (hat.theta[k,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[k,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[k,3]))/(hat.theta[k,6]*dgamma(data$obs[ind.i.u==1]-(1-hat.theta[k,4])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[k,3])+(1-hat.theta[k,6])*dgamma(data$obs[ind.i.u==1]-(1+hat.theta[k,5])*data$obs.pre[ind.i.u==1],delta[ind.i.u==1],rate=hat.theta[k,3])))

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




