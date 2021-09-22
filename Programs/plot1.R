# Fonction qui sert à tracer les processus gammas
# dans le cas où on observe uniquement les dégradations
# avant les inspections

plot1 <- function(obs,x,y,L=5,M=10,tau=1,pas=0.01,tps.final=100){
  
  DeltaImperf <- ((L <= obs) & (obs < M))
  DeltaPerf <- (obs >= M)
  temps = seq(from = 0,to = tps.final,by = pas)
  
  temps.cycle<-c(0,tau*which(obs>M),tps.final)
  indice.temps.cycle <- c(1,tau/pas*(which(obs>M))+1,length(temps))
  nb.insp.max<- floor(tps.final/tau) # nombre d'inspections max pendant la fenêtre d'observation.
  nb.cycles=nrow(x)
  
  for (k in 1:nb.cycles){
    plot(temps[indice.temps.cycle[k]:indice.temps.cycle[k+1]],x[k,1:(indice.temps.cycle[k+1]-indice.temps.cycle[k]+1)],col="red",type="l",ylim = c(0,2*M),xlim = c(0,tps.final),xlab="",ylab="")
    par(new=T)
  }
  
  plot(temps,y,type="l",ylim=c(0,2*M),xlim = c(0,tps.final),ylab="",xlab="")
  
  
  par(new=T)
  plot(tau*(1:nb.insp.max),numeric(length = nb.insp.max),ylim=c(0,2*M),type="p",xlim = c(0,tps.final),xlab = "temps",ylab="Dégradation")
  abline(h=L,col="blue")
  abline(h=M,col="red")
  abline(v=tau*(1:nb.insp.max), lty =3, col = grey(0.1), lwd = 0.45)
  
  
}
