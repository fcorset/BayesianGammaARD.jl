alpha <- rep(x = 1, times = 9)
beta <- c(1, 1, 1, 0.75, 0.75, 0.75, 1.2, 1.2, 1.2)
b <- rep(x = 1, times = 9)
rho <- c(0.2, 0.5, 0.8, 0.2, 0.5, 0.8, 0.2, 0.5, 0.8)
TabCas <- cbind(alpha, beta, b, rho)

ListFic <- list.files()
idx <- grep(pattern = "Rd", x = ListFic)
ListFic <- ListFic[idx]

NbFic <- length(x = ListFic)

MSE <- matrix(data = rep(x = 0, times = 4*NbFic), nrow = NbFic, ncol = 4)

for (i in 1:NbFic) {
  Fic <- ListFic[i]
  load(file = Fic)
  NumCas <- as.integer(strsplit(Fic,"")[[1]][11])

  vparms <- res[[2]]
  
  idx <- which((vparms[, 1] < 0) | (vparms[, 2] < 0) | (vparms[, 3] < 0) | (vparms[, 4] < 0) | (vparms[, 4] > 1))
  if (length(idx)>0) {
    vparms <- vparms[-idx, ]
  }
  parms <- matrix(data = rep(x = res[[1]], each = nrow(vparms)), nrow = nrow(vparms))
  MSE[i, ] <- colMeans(x = (vparms - parms)^2)
  
}

round(x = MSE, digits = 6)

MSE.1 <- colMeans(MSE[1:3, ])
MSE.2 <- colMeans(MSE[4:6, ])
MSE.3 <- colMeans(MSE[7:9, ])

MSE.1 == pmin(MSE.1, MSE.2)
MSE.2 == pmin(MSE.2, MSE.3)
MSE.1 == pmin(MSE.1, MSE.3)

