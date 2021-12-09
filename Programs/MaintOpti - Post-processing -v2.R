#rm(list=ls())

library(readxl)
library(writexl)


LF <- list.files()
idx <- grep(pattern = "Results-cas", x = LF)
LF <- LF[idx]
nb.files <- length(LF)

load(file = LF[1])
Costs <- output[[1]]

for (i in 2:nb.files) {
  load(file = LF[i])
  Costs <- cbind.data.frame(Costs, output[[1]]$cout.L)
}


# which(Costs == max(Costs), arr.ind = TRUE)
Costs <- Costs[, -c(47, 48, 49)]
nb.files <- nb.files - 3

ymin = min(Costs[, -1])
ymax = max(Costs[, -1])

plot(x = Costs[, 1], y = Costs[, 2], type = "l", lwd = 2, ylim = c(ymin, ymax))
for (i in 2:nb.files) {
  lines(x = Costs[, 1], y = Costs[, i+1], type = "l", lwd = 2)
}

library(kml)

traj <- t(Costs[, -1])
myCld <- clusterLongData(
  traj = traj,
  idAll = as.character(1:nb.files),
  time = Costs[, 1],
  varNames="P",
  maxNA=3
)

nb.classes <- 3
kml(object = myCld, nbClusters = nb.classes, nbRedrawing = 3, toPlot = 'traj')
cls <- getClusters(xCld = myCld, nbCluster = nb.classes, clusterRank = 2)
cls <- as.character(cls)
for (i in 1:nb.classes) {
  idx <- which(cls == LETTERS[i])
  cls[idx] <- i
}
cls <- as.numeric(cls)

for (i in 1:nb.classes) {
  idx <- which(cls == i)
  nb.idx <- length(idx)
  aux <- Costs[, c(1, idx)]
  ymin = min(aux[, -1])
  ymax = max(aux[, -1])
  plot(x = aux[, 1], y = aux[, 2], type = "l", lwd = 2, ylim = c(ymin, ymax))
  for (j in 2:nb.idx) {
    lines(x = aux[, 1], y = aux[, j+1], type = "l", lwd = 2)
  }
}

wlibrary(funHDDC)


L = max(output[[1]]$seq.L)
cls = sample(x = 1:nb.classes, size = nb.files, replace = TRUE)
basis<- create.bspline.basis(c(0,L), nbasis=25)

y = as.matrix(Costs[, -1])
var1<-smooth.basis(argvals = Costs[, 1], y = y, fdParobj = basis)$fd
res.uni<-funHDDC(var1,K=nb.classes,model="AkBkQkDk",init="kmeans",threshold=0.2)
table(cls,res.uni$class,dnn=c("True clusters","FunHDDC clusters"))
plot(var1,col=res.uni$class)
