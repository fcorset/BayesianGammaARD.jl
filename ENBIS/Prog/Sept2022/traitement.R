# Charger le fichier de données

# 
label.theta <-c("alpha","beta","b","rho","rho_w","p")

for(k in 1:6){
  hist(hat.theta[,k],freq=F,nclass=20,main=paste("Histogram of",label.theta[k]))
  abline(v=theta[k],col="green")
  abline(v=mean(hat.theta[,k]),col="red")
}
