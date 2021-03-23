library(ggplot2)
library(tidyverse)
tau = 0.5 # intervalle inter-inspection
rho = 0.5 # parametre ARD1
L=3 # seuil pour MP
M=6 # seuil pout MC
tps.final <-8 # fenêtre d'observation du processus

# Simulation d'un processus gamma
pas = 0.01 # pas de temps pour simuler le processus

temps = seq(from = 0,to = tps.final,by = pas)
alpha = 1 # paramètre de forme de Gamma a = alpha t
beta = 1.5 # paramètre d'échelle de Gamma
b=1
n=length(temps)
x=numeric(n) # processus Gamma simulé
y=numeric(n) # processus Gamma maintenu
x[1] = 0

nb.inspection<- floor(tps.final/tau)

obs<-numeric(nb.inspection)

for(i in 2:n){
  x[i] = x[i-1] + rgamma(1,scale = b, shape = alpha*(temps[i]^beta)-temps[i-1]^beta)
}

plot(temps,x,type="l")

# avant la première inspection y=x
y[1:(tau/pas)]<-x[1:(tau/pas)]

# Décision à chaque inspection

ifelse(x[tau/pas+1]<L,y[tau/pas+1]<-x[tau/pas+1],ifelse(x[tau/pas+1]<M,y[tau/pas+1]<-(1-rho)*x[tau/pas+1],^))

obs[1]<-y[tau/pas+1]

j<-1 # indicateur inspection
while (obs[j]<M & j<nb.inspection) {
  y[(j*tau/pas+2):((j+1)*tau/pas+1)]<-y[j*tau/pas+1]+x[(j*tau/pas+2):((j+1)*tau/pas+1)]-x[j*tau/pas+1]
  ifelse(y[(j+1)*tau/pas+1]<L,print("no action"),ifelse(y[(j+1)*tau/pas+1]<M,y[(j+1)*tau/pas+1]<-(1-rho)*y[(j+1)*tau/pas+1],y[(j+1)*tau/pas+1]<-0))
  obs[j+1]<-y[(j+1)*tau/pas+1]
  j<-j+1
}

df <- data.frame(temps,x,y)
mygraph <- ggplot(df,aes(x = temps)) +  
  geom_line(aes(y = y), color = "darkred") +
  geom_line(aes(y = x), color="steelblue", linetype="twodash") +
  geom_abline(slope = 0,intercept = L,color="blue") +
  geom_abline(slope = 0,intercept = M,color="red")



plot(temps,x,col="red",type="l",ylim = c(0,max(x)),xlim = c(0,8))
par(new=T)
plot(temps,y,type="l",ylim=c(0,max(x)),xlim = c(0,8))
par(new=T)
plot(tau*(1:nb.inspection),numeric(length = nb.inspection),ylim=c(0,max(x)),type="p",xlim = c(0,8))
abline(h=L,col="blue")
abline(h=M,col="red")

##################
### ESTIMATION ###
##################

# CAS A : NIVEAU DE DEGRADATION AVANT MAINTENANCE

LogLik <- function(parms) {
  alpha <- parms[1]
  beta  <- parms[2]
  b <- parms[3]
  rho <- parms[4]
  
  # Y_{t_j^-} - Y_{t_{j-1}^+} suit une loi Gamma scale = b et shape = alpha*(t_j^beta - t_{j-1}^beta)
  # pbm : estimer rho ??? EM ???
  # DeltaL_j = 1_{L < Y_{t_j^-} < M}
  # DeltaM_j = 1_{M < Y_{t_j^-} }
  # Y_{t_{j+1}^-} - Y_{t_j^+} suit une loi Gamma scale = b et shape = alpha*(t_{j+1}^beta - t_j^beta)
  # Si DeltaL_j = 1 
  # contrib = pgamma(Y_{t_{j+1}^-} - Y_{t_j^+}, scale = b, shape = alpha*(t_{j+1}^beta - t_j^beta))
  # Y_{t_{j+1}^-} - (1-rho)Y_{t_j^-} suit une loi Gamma
  # contrib = pgamma(Y_{t_{j+1}^-} - (1-rho)Y_{t_j^-}, scale = b, shape = alpha*(t_{j+1}^beta - t_j^beta))
  # Si DeltaM_j = 1 : on repart de zéro ---> comme si nouveau système
    
  
}

